"""
Auth: password hashing, JWT issuance/validation, and per-user account
storage -- including each user's own IMAP credentials, encrypted at rest.

Design decision, worth being upfront about: genuine multi-user support
means each user has their own mailbox to connect to, and their own
priority-scoring/draft history that must not leak into another user's
view. This module stores per-user IMAP credentials (encrypted with
Fernet, not plaintext) and api.py gives each user their own SenderStats/
DraftStore sqlite files, keyed by username. Without this, "multi-user"
would just be a login screen in front of one shared mailbox and one
shared draft list -- technically gated, not actually multi-user.

Known simplification: IMAP credentials are NOT verified against the real
mail server at registration time. Registration just stores what you give
it; a wrong password only surfaces later as a 401 from /inbox/priority
etc. Verifying live at registration would be better UX but adds a live
IMAP round-trip (and its own failure modes) to what should be a simple
account-creation step. Worth fixing if this becomes a real product.
"""
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import bcrypt
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from jose import jwt, JWTError

# Loaded here explicitly rather than relying on main.py (or anything else)
# having already called load_dotenv() first. auth.py previously read
# FERNET_KEY/JWT_SECRET_KEY from os.environ at import time without this,
# which worked by accident whenever main.py happened to import first --
# a real fragility, since e.g. tasks.py imports auth before main.
load_dotenv()

DB_PATH = "users.sqlite"

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-insecure-secret-change-me-in-production")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24h -- reasonable for a demo/portfolio project

# Fernet key for encrypting stored IMAP app passwords at rest. Generate one
# with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
# and put it in .env as FERNET_KEY. In a real product this would live in a
# secrets manager, not an env var -- acceptable tradeoff for this project.
_FERNET_KEY = os.getenv("FERNET_KEY")
_fernet = Fernet(_FERNET_KEY.encode()) if _FERNET_KEY else None

# bcrypt has a hard 72-byte input limit and raises on longer input.
# Validate explicitly with a clear error rather than let a mysterious
# ValueError surface from inside the library.
MAX_PASSWORD_BYTES = 72


class PasswordTooLongError(ValueError):
    pass


def hash_password(password: str) -> str:
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise PasswordTooLongError(f"Password must be at most {MAX_PASSWORD_BYTES} bytes.")
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        return False
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def encrypt_secret(value: str) -> str:
    if _fernet is None:
        raise RuntimeError(
            "FERNET_KEY not set in environment -- cannot store credentials securely. "
            "Generate one and add it to .env (see .env.example)."
        )
    return _fernet.encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt_secret(value: str) -> str:
    if _fernet is None:
        raise RuntimeError("FERNET_KEY not set in environment -- cannot decrypt stored credentials.")
    return _fernet.decrypt(value.encode("utf-8")).decode("utf-8")


def create_access_token(username: str, expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> str:
    """Returns the username from the 'sub' claim. Raises JWTError on an
    invalid, expired, or malformed token -- callers translate that to a
    401 at the API boundary."""
    payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    username = payload.get("sub")
    if username is None:
        raise JWTError("Token missing 'sub' claim")
    return username


class UserStore:
    """Sqlite-backed user accounts, including each user's own IMAP
    connection details and digest scheduling config. imap_pass is stored
    encrypted, never in plaintext."""

    def __init__(self, db_path: str = DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._setup()

    def _setup(self):
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                hashed_password TEXT NOT NULL,
                user_email TEXT NOT NULL,
                imap_host TEXT NOT NULL,
                imap_user TEXT NOT NULL,
                imap_pass_encrypted TEXT NOT NULL,
                inbox TEXT NOT NULL DEFAULT 'INBOX',
                smtp_host TEXT,
                smtp_port INTEGER NOT NULL DEFAULT 587,
                digest_frequency TEXT NOT NULL DEFAULT 'daily',
                digest_recipient TEXT,
                digest_enabled INTEGER NOT NULL DEFAULT 1,
                last_digest_sent_at TEXT,
                created_at TEXT NOT NULL
            )"""
        )
        self.conn.commit()

    def create_user(self, username: str, password: str, user_email: str,
                     imap_host: str, imap_user: str, imap_pass: str, inbox: str = "INBOX",
                     smtp_host: str | None = None, smtp_port: int = 587,
                     digest_frequency: str = "daily", digest_recipient: str | None = None,
                     digest_enabled: bool = True) -> None:
        if self.get_user(username) is not None:
            raise ValueError(f"User '{username}' already exists")
        hashed = hash_password(password)
        encrypted_pass = encrypt_secret(imap_pass)
        if smtp_host is None and imap_host.startswith("imap."):
            smtp_host = "smtp." + imap_host[len("imap."):]
        recipient = digest_recipient or user_email
        self.conn.execute(
            """INSERT INTO users
               (username, hashed_password, user_email, imap_host, imap_user, imap_pass_encrypted,
                inbox, smtp_host, smtp_port, digest_frequency, digest_recipient, digest_enabled, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (username, hashed, user_email, imap_host, imap_user, encrypted_pass, inbox,
             smtp_host, smtp_port, digest_frequency, recipient, int(digest_enabled),
             datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    def get_user(self, username: str) -> dict | None:
        row = self.conn.execute(
            """SELECT username, hashed_password, user_email, imap_host, imap_user, imap_pass_encrypted,
                      inbox, smtp_host, smtp_port, digest_frequency, digest_recipient, digest_enabled,
                      last_digest_sent_at
               FROM users WHERE username = ?""",
            (username,),
        ).fetchone()
        if row is None:
            return None
        return {
            "username": row[0], "hashed_password": row[1], "user_email": row[2],
            "imap_host": row[3], "imap_user": row[4], "imap_pass_encrypted": row[5], "inbox": row[6],
            "smtp_host": row[7], "smtp_port": row[8], "digest_frequency": row[9],
            "digest_recipient": row[10], "digest_enabled": bool(row[11]), "last_digest_sent_at": row[12],
        }

    def update_digest_config(self, username: str, frequency: str | None = None,
                              recipient: str | None = None, enabled: bool | None = None,
                              smtp_host: str | None = None, smtp_port: int | None = None) -> bool:
        user = self.get_user(username)
        if user is None:
            return False
        self.conn.execute(
            """UPDATE users SET
                digest_frequency = ?, digest_recipient = ?, digest_enabled = ?,
                smtp_host = ?, smtp_port = ?
               WHERE username = ?""",
            (
                frequency if frequency is not None else user["digest_frequency"],
                recipient if recipient is not None else user["digest_recipient"],
                int(enabled) if enabled is not None else int(user["digest_enabled"]),
                smtp_host if smtp_host is not None else user["smtp_host"],
                smtp_port if smtp_port is not None else user["smtp_port"],
                username,
            ),
        )
        self.conn.commit()
        return True

    def mark_digest_sent(self, username: str, sent_at: datetime | None = None) -> None:
        sent_at = sent_at or datetime.now(timezone.utc)
        self.conn.execute(
            "UPDATE users SET last_digest_sent_at = ? WHERE username = ?",
            (sent_at.isoformat(), username),
        )
        self.conn.commit()

    def get_all_users(self) -> list:
        rows = self.conn.execute("SELECT username FROM users").fetchall()
        return [self.get_user(r[0]) for r in rows]

    def get_digest_enabled_users(self) -> list:
        rows = self.conn.execute("SELECT username FROM users WHERE digest_enabled = 1").fetchall()
        return [self.get_user(r[0]) for r in rows]

    def authenticate(self, username: str, password: str) -> dict | None:
        """Returns the user dict on success, None on bad username or password."""
        user = self.get_user(username)
        if user is None:
            return None
        if not verify_password(password, user["hashed_password"]):
            return None
        return user

    def close(self):
        self.conn.close()
