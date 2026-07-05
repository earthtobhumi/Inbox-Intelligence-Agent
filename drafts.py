"""
Draft reply generation.

Design principle: drafts are generated and stored for human review. Nothing
in this module sends email. `DraftStore` exists so a draft survives process
restarts and can later be listed/approved/edited via the FastAPI `/drafts`
endpoint, without needing the LLM to regenerate it each time it's viewed.
"""
import re
import sqlite3
import uuid
from datetime import datetime, timezone

DB_PATH = "drafts.sqlite"

# Matches common bracketed name placeholders a model might leave in a
# sign-off: [Your Name], [User's Name], [Name], [Sender Name], [Full Name],
# with or without smart quotes, any bracket style.
_PLACEHOLDER_PATTERN = re.compile(
    r"[\[\(]\s*(your|user'?s?|sender'?s?|full|my)?\s*name\s*[\]\)]",
    re.IGNORECASE,
)


def strip_placeholder_signoff(draft_text: str) -> str:
    """Deterministically remove bracketed name placeholders from a draft.

    Added after the prompt instruction alone failed to stop this three
    times in live testing against qwen2.5:7b -- a 7B model will not
    reliably follow a 'don't do X' instruction 100% of the time, so this
    can't be prompt-only. This runs on every draft regardless of what the
    model produced, giving a hard guarantee instead of a soft nudge.
    """
    if not draft_text:
        return draft_text
    cleaned = _PLACEHOLDER_PATTERN.sub("", draft_text)
    # Collapse resulting trailing whitespace/blank lines left behind
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)  # trailing spaces before newline
    cleaned = cleaned.rstrip()
    return cleaned


DRAFT_SYSTEM_PROMPT = (
    "You are drafting a reply to an email on behalf of the user. Rules:\n"
    "- Only reference facts that actually appear in the original email. "
    "Do not invent names, dates, commitments, or details not present.\n"
    "- Keep the tone professional but not stiff, matching the tone of the "
    "original email where reasonable.\n"
    "- Keep it concise -- a few sentences unless the original clearly needs more.\n"
    "- Do not include a subject line. End with a generic closing like "
    "'Best,' or 'Thanks,' followed by nothing else -- no name, no "
    "placeholder text, no brackets like '[Your Name]' or '[User's Name]'. "
    "Just the closing word and comma, then stop. The user signs their own name.\n"
    "- This draft will be reviewed by the user before sending. If the email "
    "doesn't clearly need a reply (e.g. it's a notification or FYI), say so "
    "instead of forcing a reply."
)


def build_draft_prompt(subject: str, sender: str, body: str) -> str:
    return (
        f"{DRAFT_SYSTEM_PROMPT}\n\n"
        f"--- Original email ---\n"
        f"From: {sender}\n"
        f"Subject: {subject}\n\n"
        f"{body}\n"
        f"--- End of original email ---\n\n"
        f"Draft a reply:"
    )


class DraftStore:
    """Sqlite-backed storage for generated drafts, keyed by a generated
    draft_id (not email uid, since a single email could get re-drafted)."""

    def __init__(self, db_path: str = DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._setup()

    def _setup(self):
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS drafts (
                draft_id TEXT PRIMARY KEY,
                email_uid TEXT NOT NULL,
                subject TEXT,
                sender TEXT,
                draft_text TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending_review',
                created_at TEXT NOT NULL
            )"""
        )
        self.conn.commit()

    def save(self, email_uid: str, subject: str, sender: str, draft_text: str) -> str:
        draft_id = str(uuid.uuid4())[:8]
        self.conn.execute(
            """INSERT INTO drafts (draft_id, email_uid, subject, sender, draft_text, status, created_at)
               VALUES (?, ?, ?, ?, ?, 'pending_review', ?)""",
            (draft_id, email_uid, subject, sender, draft_text, datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()
        return draft_id

    def get(self, draft_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT draft_id, email_uid, subject, sender, draft_text, status, created_at FROM drafts WHERE draft_id = ?",
            (draft_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "draft_id": row[0], "email_uid": row[1], "subject": row[2],
            "sender": row[3], "draft_text": row[4], "status": row[5], "created_at": row[6],
        }

    def list_all(self, status: str = None) -> list:
        if status:
            rows = self.conn.execute(
                "SELECT draft_id, email_uid, subject, sender, draft_text, status, created_at FROM drafts WHERE status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT draft_id, email_uid, subject, sender, draft_text, status, created_at FROM drafts ORDER BY created_at DESC"
            ).fetchall()
        return [
            {
                "draft_id": r[0], "email_uid": r[1], "subject": r[2],
                "sender": r[3], "draft_text": r[4], "status": r[5], "created_at": r[6],
            }
            for r in rows
        ]

    def update_status(self, draft_id: str, status: str) -> bool:
        cur = self.conn.execute(
            "UPDATE drafts SET status = ? WHERE draft_id = ?", (status, draft_id)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def close(self):
        self.conn.close()
