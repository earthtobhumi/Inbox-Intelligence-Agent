"""
Priority scoring for unread emails.

Design note (raw IMAP, not Gmail API): there is no built-in "sender
importance" signal to query. Sender familiarity here is a locally-learned
proxy -- an interaction count stored in sqlite, incremented each time an
email from that sender is scored. It starts cold for a fresh install and
gets more accurate with usage. This is a known simplification, not a
substitute for real contact-frequency data from a mail provider API.
"""
import re
import sqlite3
from datetime import datetime, timezone
from email.utils import parseaddr

DB_PATH = "sender_stats.sqlite"

# Sender familiarity count is capped so one extremely frequent sender
# (e.g. a noreply/newsletter address) can't permanently dominate the score.
FAMILIARITY_CAP = 20

URGENT_KEYWORDS = [
    r"\burgent\b", r"\basap\b", r"\bdeadline\b", r"\baction required\b",
    r"\bresponse needed\b", r"\bimportant\b", r"\btime[- ]sensitive\b",
    r"\bby (today|tomorrow|eod|cob)\b", r"\bfinal notice\b", r"\breminder\b",
]
_URGENT_PATTERN = re.compile("|".join(URGENT_KEYWORDS), re.IGNORECASE)

# Weights must sum to 1.0 -- kept as named constants so the rubric is
# easy to explain and easy to tune later.
WEIGHT_URGENCY = 0.35
WEIGHT_DIRECTNESS = 0.20
WEIGHT_FAMILIARITY = 0.20
WEIGHT_RECENCY = 0.25


# Patterns that identify automated/transactional senders (no-reply,
# notifications, security alerts, etc). Frequency from these senders means
# "this system emails you a lot," not "this relationship matters" -- the
# opposite of what familiarity is meant to capture. Detected from real
# output: Google security alerts were outranking a personal email purely
# because Google emails more often than a friend does.
AUTOMATED_SENDER_PATTERNS = [
    r"no-?reply", r"do-?not-?reply", r"^notifications?@", r"^alerts?@",
    r"mailer-daemon", r"^security@", r"^accounts@", r"^automated@",
    r"^system@", r"^support@.*\.(google|apple|microsoft|amazon)\.com$",
]
_AUTOMATED_PATTERN = re.compile("|".join(AUTOMATED_SENDER_PATTERNS), re.IGNORECASE)


def is_automated_sender(sender_email: str) -> bool:
    """Best-effort heuristic, not exhaustive -- pattern-matches common
    no-reply/notification/security-alert address conventions."""
    if not sender_email:
        return False
    addr = parseaddr(sender_email)[1].lower()
    return bool(_AUTOMATED_PATTERN.search(addr))


def urgency_keyword_score(subject: str) -> float:
    """1.0 if subject contains urgency language, else 0.0.
    Binary rather than graded -- a subject either signals urgency or it
    doesn't; partial credit here would just be noise."""
    if not subject:
        return 0.0
    return 1.0 if _URGENT_PATTERN.search(subject) else 0.0


def directness_score(to_addrs, cc_addrs, user_email: str) -> float:
    """1.0 if the user is a direct To: recipient, 0.5 if only Cc'd,
    0.0 if neither (e.g. Bcc, mailing list, or header mismatch)."""
    user_email = (user_email or "").lower().strip()
    to_list = [parseaddr(a)[1].lower() for a in (to_addrs or [])]
    cc_list = [parseaddr(a)[1].lower() for a in (cc_addrs or [])]

    if user_email and user_email in to_list:
        return 1.0
    if user_email and user_email in cc_list:
        return 0.5
    return 0.0


def recency_score(email_date, now=None) -> float:
    """1.0 for an email received in the last hour, decaying linearly to
    0.0 at 7 days old. Older than 7 days floors at 0.0."""
    if email_date is None:
        return 0.0
    now = now or datetime.now(timezone.utc)
    if email_date.tzinfo is None:
        email_date = email_date.replace(tzinfo=timezone.utc)
    age_hours = (now - email_date).total_seconds() / 3600
    if age_hours <= 1:
        return 1.0
    if age_hours >= 24 * 7:
        return 0.0
    return max(0.0, 1.0 - (age_hours / (24 * 7)))


def familiarity_score(interaction_count: int) -> float:
    """Scales 0.0 (never seen this sender) to 1.0 (seen FAMILIARITY_CAP+
    times). A cold-start sender scores 0.0, not a penalty -- just no
    signal yet."""
    if interaction_count <= 0:
        return 0.0
    return min(interaction_count, FAMILIARITY_CAP) / FAMILIARITY_CAP


def combine_score(urgency: float, directness: float, familiarity: float, recency: float) -> float:
    """Weighted sum, returned as 0-100 for readability in the UI/API."""
    raw = (
        urgency * WEIGHT_URGENCY
        + directness * WEIGHT_DIRECTNESS
        + familiarity * WEIGHT_FAMILIARITY
        + recency * WEIGHT_RECENCY
    )
    return round(raw * 100, 1)


class SenderStats:
    """Sqlite-backed interaction counter, keyed by sender email address."""

    def __init__(self, db_path: str = DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._setup()

    def _setup(self):
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS sender_interactions (
                sender_email TEXT PRIMARY KEY,
                count INTEGER NOT NULL DEFAULT 0
            )"""
        )
        self.conn.commit()

    def get_count(self, sender_email: str) -> int:
        sender_email = parseaddr(sender_email or "")[1].lower()
        row = self.conn.execute(
            "SELECT count FROM sender_interactions WHERE sender_email = ?",
            (sender_email,),
        ).fetchone()
        return row[0] if row else 0

    def record(self, sender_email: str) -> int:
        """Increment and return the new count for this sender."""
        sender_email = parseaddr(sender_email or "")[1].lower()
        if not sender_email:
            return 0
        self.conn.execute(
            """INSERT INTO sender_interactions (sender_email, count) VALUES (?, 1)
               ON CONFLICT(sender_email) DO UPDATE SET count = count + 1""",
            (sender_email,),
        )
        self.conn.commit()
        return self.get_count(sender_email)

    def close(self):
        self.conn.close()


def score_email(email: dict, user_email: str, sender_stats: SenderStats, record_interaction: bool = True) -> dict:
    """
    email: dict with keys uid, subject, from, to, cc, date (datetime)
    Returns email dict augmented with score, score_breakdown, reasons.
    """
    sender = email.get("from", "")
    automated = is_automated_sender(sender)

    if record_interaction:
        count = sender_stats.record(sender)
    else:
        count = sender_stats.get_count(sender)

    u = urgency_keyword_score(email.get("subject", ""))
    d = directness_score(email.get("to", []), email.get("cc", []), user_email)
    # Automated/transactional senders don't get a familiarity boost --
    # high frequency from a bot is noise, not a signal of importance.
    f = 0.0 if automated else familiarity_score(count)
    r = recency_score(email.get("date"))

    score = combine_score(u, d, f, r)

    reasons = []
    if u:
        reasons.append("urgency language in subject")
    if d == 1.0:
        reasons.append("directly addressed to you")
    elif d == 0.5:
        reasons.append("you were cc'd")
    if automated:
        reasons.append("automated/no-reply sender (familiarity suppressed)")
    elif f > 0:
        reasons.append(f"seen {count}x from this sender before")
    if r > 0.75:
        reasons.append("received recently")

    return {
        **email,
        "score": score,
        "score_breakdown": {"urgency": u, "directness": d, "familiarity": f, "recency": r},
        "reasons": reasons or ["no strong signals"],
    }


def rank_emails(emails: list, user_email: str, sender_stats: SenderStats) -> list:
    """Score a list of email dicts and return them sorted by score, descending."""
    scored = [score_email(e, user_email, sender_stats) for e in emails]
    return sorted(scored, key=lambda e: e["score"], reverse=True)
