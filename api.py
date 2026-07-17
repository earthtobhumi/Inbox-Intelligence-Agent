"""
FastAPI layer over the email agent, with JWT auth and genuine per-user
isolation.

Design decisions worth knowing:

1. /inbox/priority and GET /drafts do NOT go through the LangGraph agent
   or an LLM call -- they call the same deterministic Python
   (priority.rank_emails, DraftStore) the LangGraph tools already use,
   directly. The LLM is only invoked where generation actually happens
   (POST /drafts/generate). See main.py/README for the full rationale.

2. Multi-user is NOT just a login screen in front of one shared mailbox.
   Each account stores its own IMAP credentials (encrypted at rest via
   auth.py) and gets its own SenderStats/DraftStore sqlite files, keyed
   by username, under ./data/. Two users hitting the same endpoints see
   completely independent inboxes and draft histories.

3. Registration does NOT verify IMAP credentials against a live server --
   a wrong password only surfaces later as a 401 from a route that
   actually connects. See auth.py docstring for the tradeoff.
"""
import os

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from pydantic import BaseModel
from imap_tools import AND
from imap_tools.errors import MailboxLoginError

import main as agent
import priority
import drafts
import auth

app = FastAPI(title="Email Agent API", version="0.2.0")
security = HTTPBearer()

DATA_DIR = "data"


# ---------- response/request models ----------

class RegisterRequest(BaseModel):
    username: str
    password: str
    user_email: str
    imap_host: str
    imap_user: str
    imap_pass: str
    inbox: str = "INBOX"
    smtp_host: str | None = None  # best-effort derived from imap_host if not given
    smtp_port: int = 587
    digest_frequency: str = "daily"
    digest_recipient: str | None = None  # defaults to user_email if not given
    digest_enabled: bool = True


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class DraftOut(BaseModel):
    draft_id: str
    email_uid: str
    subject: str | None
    sender: str | None
    draft_text: str
    status: str
    created_at: str


class GenerateDraftRequest(BaseModel):
    uid: str


class UpdateDraftStatusRequest(BaseModel):
    status: str


class DigestConfigOut(BaseModel):
    digest_frequency: str
    digest_recipient: str | None
    digest_enabled: bool
    smtp_host: str | None
    smtp_port: int
    last_digest_sent_at: str | None


class UpdateDigestConfigRequest(BaseModel):
    frequency: str | None = None
    recipient: str | None = None
    enabled: bool | None = None
    smtp_host: str | None = None
    smtp_port: int | None = None


# ---------- per-user store singletons ----------
# Each user gets their own sqlite files, not shared global state. Cached
# per-username so we don't reopen a sqlite connection on every request.

_user_sender_stats_cache: dict = {}
_user_draft_store_cache: dict = {}
_user_store_singleton = None


def get_user_store() -> auth.UserStore:
    global _user_store_singleton
    if _user_store_singleton is None:
        _user_store_singleton = auth.UserStore()
    return _user_store_singleton


def get_user_sender_stats(username: str) -> priority.SenderStats:
    if username not in _user_sender_stats_cache:
        os.makedirs(DATA_DIR, exist_ok=True)
        _user_sender_stats_cache[username] = priority.SenderStats(
            db_path=os.path.join(DATA_DIR, f"{username}_sender_stats.sqlite")
        )
    return _user_sender_stats_cache[username]


def get_user_draft_store(username: str) -> drafts.DraftStore:
    if username not in _user_draft_store_cache:
        os.makedirs(DATA_DIR, exist_ok=True)
        _user_draft_store_cache[username] = drafts.DraftStore(
            db_path=os.path.join(DATA_DIR, f"{username}_drafts.sqlite")
        )
    return _user_draft_store_cache[username]


# ---------- auth dependency ----------

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    token = credentials.credentials
    try:
        username = auth.decode_access_token(token)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = get_user_store().get_user(username)
    if user is None:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return user


# ---------- IMAP helpers (per-user) ----------

def _connect_for_user(user: dict):
    imap_pass = auth.decrypt_secret(user["imap_pass_encrypted"])
    return agent._open_mailbox(user["imap_host"], user["imap_user"], imap_pass, user.get("inbox", "INBOX"))


def _fetch_unread_with_headers(user: dict):
    with _connect_for_user(user) as mailbox:
        return list(
            mailbox.fetch(criteria=AND(seen=False), headers_only=True, mark_seen=False)
        )


def _imap_error_to_http(e: Exception) -> HTTPException:
    if isinstance(e, MailboxLoginError):
        return HTTPException(
            status_code=401,
            detail="IMAP login failed for this account's stored credentials (use an app password).",
        )
    return HTTPException(status_code=503, detail=f"Could not connect to mail server: {e}")


# ---------- auth routes ----------

@app.post("/auth/register", response_model=TokenResponse)
def register(body: RegisterRequest):
    if body.digest_frequency not in ("daily", "weekly"):
        raise HTTPException(status_code=422, detail="digest_frequency must be 'daily' or 'weekly'")

    store = get_user_store()
    try:
        store.create_user(
            username=body.username, password=body.password, user_email=body.user_email,
            imap_host=body.imap_host, imap_user=body.imap_user, imap_pass=body.imap_pass,
            inbox=body.inbox, smtp_host=body.smtp_host, smtp_port=body.smtp_port,
            digest_frequency=body.digest_frequency, digest_recipient=body.digest_recipient,
            digest_enabled=body.digest_enabled,
        )
    except auth.PasswordTooLongError as e:
        # Must be caught before ValueError below -- PasswordTooLongError
        # is itself a ValueError subclass, so if this were listed second
        # it would never be reached.
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    token = auth.create_access_token(body.username)
    return TokenResponse(access_token=token)


@app.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest):
    store = get_user_store()
    user = store.authenticate(body.username, body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = auth.create_access_token(body.username)
    return TokenResponse(access_token=token)


# ---------- protected routes ----------

@app.get("/inbox/priority")
def get_inbox_priority(current_user: dict = Depends(get_current_user)):
    """Rank this user's unread emails by priority. Pure Python + sqlite --
    no LLM call. Uses this user's own stored IMAP credentials and their
    own sender-familiarity history, not shared with any other account."""
    try:
        raw_emails = _fetch_unread_with_headers(current_user)
    except (MailboxLoginError, RuntimeError, ConnectionError, OSError) as e:
        raise _imap_error_to_http(e)

    if not raw_emails:
        return []

    email_dicts = [
        {
            "uid": mail.uid,
            "date": mail.date,
            "subject": mail.subject,
            "from": mail.from_,
            "to": list(mail.to or []),
            "cc": list(mail.cc or []),
        }
        for mail in raw_emails
    ]

    sender_stats = get_user_sender_stats(current_user["username"])
    ranked = priority.rank_emails(email_dicts, current_user["user_email"], sender_stats)
    for e in ranked:
        e["date"] = e["date"].strftime("%Y-%m-%d %H:%M:%S") if e["date"] else None

    return ranked


@app.get("/drafts", response_model=list[DraftOut])
def list_drafts(status: str | None = Query(default="pending_review"),
                 current_user: dict = Depends(get_current_user)):
    """List this user's drafts, filtered by status. status=all for everything."""
    filter_status = None if status == "all" else status
    store = get_user_draft_store(current_user["username"])
    return store.list_all(status=filter_status)


@app.get("/drafts/{draft_id}", response_model=DraftOut)
def get_draft(draft_id: str, current_user: dict = Depends(get_current_user)):
    store = get_user_draft_store(current_user["username"])
    draft = store.get(draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail=f"No draft found with id {draft_id}")
    return draft


@app.patch("/drafts/{draft_id}", response_model=DraftOut)
def update_draft_status(draft_id: str, body: UpdateDraftStatusRequest,
                          current_user: dict = Depends(get_current_user)):
    """Update a draft's status. Does NOT send email -- only updates the
    local record, e.g. after the user has sent the reply themselves."""
    allowed_statuses = {"pending_review", "approved", "discarded", "sent"}
    if body.status not in allowed_statuses:
        raise HTTPException(status_code=422, detail=f"status must be one of {sorted(allowed_statuses)}")

    store = get_user_draft_store(current_user["username"])
    updated = store.update_status(draft_id, body.status)
    if not updated:
        raise HTTPException(status_code=404, detail=f"No draft found with id {draft_id}")
    return store.get(draft_id)


@app.post("/drafts/generate", response_model=DraftOut)
def generate_draft(body: GenerateDraftRequest, current_user: dict = Depends(get_current_user)):
    """Generate a new draft reply for this user's email UID. Calls the LLM
    directly (bypassing the LangGraph tool router -- one grounded
    generation call needs no multi-step reasoning)."""
    uid = body.uid
    try:
        with _connect_for_user(current_user) as mailbox:
            email_message = next(mailbox.fetch(AND(uid=[uid]), mark_seen=False), None)
    except (MailboxLoginError, RuntimeError, ConnectionError, OSError) as e:
        raise _imap_error_to_http(e)
    except TypeError:
        raise HTTPException(status_code=422, detail=f"'{uid}' is not a valid email UID.")

    if email_message is None:
        raise HTTPException(status_code=404, detail=f"No email found with UID {uid}")

    body_text = agent._clean_body(email_message.text, email_message.html)
    prompt = drafts.build_draft_prompt(email_message.subject, email_message.from_, body_text)

    try:
        draft_text = agent.raw_llm.invoke(prompt).content
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"LLM call failed ({e}). Is Ollama running with {agent.CHAT_MODEL} pulled?",
        )

    draft_text = drafts.strip_placeholder_signoff(draft_text)

    store = get_user_draft_store(current_user["username"])
    draft_id = store.save(
        email_uid=uid, subject=email_message.subject, sender=email_message.from_, draft_text=draft_text
    )
    return store.get(draft_id)


@app.get("/digest/config", response_model=DigestConfigOut)
def get_digest_config(current_user: dict = Depends(get_current_user)):
    return DigestConfigOut(
        digest_frequency=current_user["digest_frequency"],
        digest_recipient=current_user["digest_recipient"],
        digest_enabled=current_user["digest_enabled"],
        smtp_host=current_user["smtp_host"],
        smtp_port=current_user["smtp_port"],
        last_digest_sent_at=current_user["last_digest_sent_at"],
    )


@app.patch("/digest/config", response_model=DigestConfigOut)
def update_digest_config(body: UpdateDigestConfigRequest, current_user: dict = Depends(get_current_user)):
    if body.frequency is not None and body.frequency not in ("daily", "weekly"):
        raise HTTPException(status_code=422, detail="frequency must be 'daily' or 'weekly'")

    store = get_user_store()
    store.update_digest_config(
        current_user["username"], frequency=body.frequency, recipient=body.recipient,
        enabled=body.enabled, smtp_host=body.smtp_host, smtp_port=body.smtp_port,
    )
    updated_user = store.get_user(current_user["username"])
    return DigestConfigOut(
        digest_frequency=updated_user["digest_frequency"],
        digest_recipient=updated_user["digest_recipient"],
        digest_enabled=updated_user["digest_enabled"],
        smtp_host=updated_user["smtp_host"],
        smtp_port=updated_user["smtp_port"],
        last_digest_sent_at=updated_user["last_digest_sent_at"],
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": agent.CHAT_MODEL}
