"""
Celery tasks for scheduled digest sending.

Two tasks:
- send_digest_for_user(username): does the actual work for one user --
  fetch, score, compose, send, mark sent. Called individually so one
  user's failure (bad SMTP creds, IMAP down) doesn't block others.
- check_and_send_due_digests(): the periodic task Celery Beat triggers
  hourly. Iterates all digest-enabled users, checks who's actually due
  via digest.is_digest_due(), and dispatches send_digest_for_user for
  each one as a separate task (not inline), so a slow/stuck user doesn't
  hold up the check itself.
"""
import logging

from celery_app import celery_app
from imap_tools import AND
from imap_tools.errors import MailboxLoginError

import auth
import digest
import main as agent
import priority

logger = logging.getLogger(__name__)


@celery_app.task(name="tasks.send_digest_for_user", bind=True, max_retries=2, default_retry_delay=300)
def send_digest_for_user(self, username: str):
    """Fetch this user's unread emails, score them, compose a digest,
    send it via their configured SMTP, and mark it sent. Raises are
    caught here and logged rather than propagated -- one user's failure
    must not crash the worker or block other users' digests. SMTP
    failures get retried (transient network issues); everything else
    is logged and given up on for this run."""
    store = auth.UserStore()
    user = store.get_user(username)
    if user is None:
        logger.warning(f"send_digest_for_user: user '{username}' no longer exists, skipping.")
        return {"status": "skipped", "reason": "user not found"}

    if not user["smtp_host"]:
        logger.warning(f"send_digest_for_user: user '{username}' has no smtp_host configured, skipping.")
        return {"status": "skipped", "reason": "no smtp_host configured"}

    try:
        imap_pass = auth.decrypt_secret(user["imap_pass_encrypted"])
        with agent._open_mailbox(user["imap_host"], user["imap_user"], imap_pass, user.get("inbox", "INBOX")) as mailbox:
            raw_emails = list(
                mailbox.fetch(criteria=AND(seen=False), headers_only=True, mark_seen=False)
            )
    except MailboxLoginError as e:
        logger.error(f"send_digest_for_user: IMAP login failed for '{username}': {e}")
        return {"status": "failed", "reason": "imap login failed"}
    except (RuntimeError, ConnectionError, OSError) as e:
        logger.error(f"send_digest_for_user: IMAP connection failed for '{username}': {e}")
        return {"status": "failed", "reason": f"imap connection failed: {e}"}

    email_dicts = [
        {
            "uid": mail.uid, "date": mail.date, "subject": mail.subject,
            "from": mail.from_, "to": list(mail.to or []), "cc": list(mail.cc or []),
        }
        for mail in raw_emails
    ]

    sender_stats = priority.SenderStats(db_path=f"data/{username}_sender_stats.sqlite")
    # Digest scoring should NOT record interactions -- viewing a digest
    # summary shouldn't count as the user having engaged with a sender
    # the way opening their live inbox does (see digest.py docstring).
    ranked = [
        priority.score_email(e, user["user_email"], sender_stats, record_interaction=False)
        for e in email_dicts
    ]
    ranked.sort(key=lambda e: e["score"], reverse=True)

    subject, body = digest.build_digest_content(ranked, user["digest_frequency"])

    try:
        digest.send_digest_email(
            smtp_host=user["smtp_host"], smtp_port=user["smtp_port"],
            smtp_user=user["imap_user"], smtp_pass=imap_pass,
            recipient=user["digest_recipient"] or user["user_email"],
            subject=subject, body=body,
        )
    except Exception as e:
        logger.error(f"send_digest_for_user: SMTP send failed for '{username}': {e}")
        raise self.retry(exc=e)

    store.mark_digest_sent(username)
    logger.info(f"send_digest_for_user: digest sent successfully for '{username}' ({len(ranked)} emails).")
    return {"status": "sent", "email_count": len(ranked)}


@celery_app.task(name="tasks.check_and_send_due_digests")
def check_and_send_due_digests():
    """Periodic task (Celery Beat, hourly). Checks every digest-enabled
    user and dispatches send_digest_for_user for anyone whose digest is
    due, based on their individual frequency + last_digest_sent_at."""
    store = auth.UserStore()
    enabled_users = store.get_digest_enabled_users()

    dispatched = []
    for user in enabled_users:
        if digest.is_digest_due(user["digest_frequency"], user["last_digest_sent_at"]):
            send_digest_for_user.delay(user["username"])
            dispatched.append(user["username"])

    logger.info(f"check_and_send_due_digests: dispatched digests for {dispatched}")
    return {"dispatched": dispatched, "checked": len(enabled_users)}
