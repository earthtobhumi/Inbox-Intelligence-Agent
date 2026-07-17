"""
Digest generation and sending.

Design decision, consistent with the rest of this project: digest content
is generated DETERMINISTICALLY from already-ranked email data -- no LLM
call. A scheduled background job is exactly where you don't want a new
LLM-shaped failure point (model unavailable, malformed output, slow
response holding up a Celery worker). The digest lists counts and the
top-10 highest-priority emails from the period, with their existing
score/reasons from priority.py -- informative without needing generation.

This is also the first place in the project that actually sends email.
Deliberately different from draft_reply, which never sends: a digest is
the assistant notifying the user about their own inbox, not replying to
someone on the user's behalf on their account. That distinction is why
this needed its own explicit design sign-off rather than reusing draft
infrastructure.
"""
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

FREQUENCY_TO_TIMEDELTA = {
    "daily": timedelta(days=1),
    "weekly": timedelta(days=7),
}

TOP_N_IN_DIGEST = 10


def is_digest_due(frequency: str, last_sent_at: str | None, now: datetime | None = None) -> bool:
    """Pure function: given a user's frequency setting and when their last
    digest went out (ISO string or None), decide if one is due now."""
    now = now or datetime.now(timezone.utc)
    if last_sent_at is None:
        return True
    interval = FREQUENCY_TO_TIMEDELTA.get(frequency)
    if interval is None:
        # Unknown frequency value -- fail safe by NOT sending rather than
        # guessing. A malformed config should never cause unexpected spam.
        return False
    last_sent = datetime.fromisoformat(last_sent_at)
    if last_sent.tzinfo is None:
        last_sent = last_sent.replace(tzinfo=timezone.utc)
    return (now - last_sent) >= interval


def build_digest_content(ranked_emails: list, frequency: str) -> tuple[str, str]:
    """Pure function: given already-ranked emails (priority.rank_emails
    output) for the period, returns (subject, body) for the digest email.
    Does not touch IMAP, sqlite, or an LLM -- fully unit-testable."""
    period = "day" if frequency == "daily" else "week"
    count = len(ranked_emails)
    high_priority_count = sum(1 for e in ranked_emails if e["score"] >= 50)

    subject = f"Your {frequency} digest: {count} unread email(s)"

    if count == 0:
        body = f"Your inbox is clear -- no unread emails in the past {period}."
        return subject, body

    lines = [
        f"{count} unread email(s) in the past {period}, "
        f"{high_priority_count} flagged as high priority.",
        "",
    ]

    top = ranked_emails[:TOP_N_IN_DIGEST]
    for i, e in enumerate(top, start=1):
        lines.append(f"{i}. [{e['score']}] {e.get('subject') or '(no subject)'}")
        lines.append(f"   From: {e.get('from', 'unknown')}")
        if e.get("reasons"):
            lines.append(f"   Why: {', '.join(e['reasons'])}")
        lines.append("")

    remaining = count - len(top)
    if remaining > 0:
        lines.append(f"...and {remaining} more.")

    body = "\n".join(lines)
    return subject, body


def send_digest_email(smtp_host: str, smtp_port: int, smtp_user: str, smtp_pass: str,
                       recipient: str, subject: str, body: str) -> None:
    """Sends a plain-text email via SMTP with STARTTLS. Raises on failure --
    callers (the Celery task) are responsible for catching and logging/
    retrying, since a background job must never crash the whole worker
    process on one user's SMTP failure."""
    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, [recipient], msg.as_string())
