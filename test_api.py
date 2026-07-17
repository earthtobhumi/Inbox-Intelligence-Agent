"""
Tests for api.py using FastAPI's TestClient with mocked IMAP and LLM.
Every protected route now requires a Bearer token, so most tests go
through a real register -> login -> authorized-request flow rather than
mocking auth away, to actually exercise the auth wiring end to end.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from imap_tools.errors import MailboxLoginError
from langchain_core.messages import AIMessage
from cryptography.fernet import Fernet

import api
import auth
import main
import priority


@pytest.fixture(autouse=True)
def _isolated_stores(tmp_path, monkeypatch):
    """Give every test a throwaway UserStore and clear the per-user
    sender-stats/draft-store caches, so tests don't leak state into each
    other or touch real files on disk."""
    monkeypatch.setattr(api, "_user_store_singleton", None)
    monkeypatch.setattr(api, "_user_sender_stats_cache", {})
    monkeypatch.setattr(api, "_user_draft_store_cache", {})
    monkeypatch.setattr(api, "DATA_DIR", str(tmp_path / "data"))

    test_user_store = auth.UserStore(db_path=str(tmp_path / "users.sqlite"))
    monkeypatch.setattr(api, "_user_store_singleton", test_user_store)

    monkeypatch.setattr(auth, "_fernet", Fernet(Fernet.generate_key()))

    yield
    test_user_store.close()


@pytest.fixture
def client():
    return TestClient(api.app)


def _register(client, username="alice", password="alice-password-123",
              user_email="alice@example.com", imap_host="imap.gmail.com",
              imap_user="alice@gmail.com", imap_pass="app-pw-123"):
    resp = client.post("/auth/register", json={
        "username": username, "password": password, "user_email": user_email,
        "imap_host": imap_host, "imap_user": imap_user, "imap_pass": imap_pass,
    })
    return resp


def _auth_headers(client, **kwargs) -> dict:
    resp = _register(client, **kwargs)
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _fake_mail(uid, subject, from_, to=None, cc=None, date=None):
    m = MagicMock()
    m.uid = uid
    m.subject = subject
    m.from_ = from_
    m.to = to or []
    m.cc = cc or []
    m.date = date or datetime.now(timezone.utc)
    return m


def _fake_mailbox(fetch_return):
    fb = MagicMock()
    fb.__enter__.return_value = fb
    fb.__exit__.return_value = False
    fb.fetch.return_value = fetch_return
    return fb


# ---------- GET /health (unauthenticated) ----------

def test_health_returns_ok_without_auth(client):
    resp = client.get("/health")
    assert resp.status_code == 200


# ---------- auth: register ----------

def test_register_returns_access_token(client):
    resp = _register(client)
    assert resp.status_code == 200
    assert "access_token" in resp.json()
    assert resp.json()["token_type"] == "bearer"


def test_register_duplicate_username_returns_409(client):
    _register(client, username="alice")
    resp = _register(client, username="alice")
    assert resp.status_code == 409


def test_register_overly_long_password_returns_422(client):
    resp = _register(client, password="a" * 100)
    assert resp.status_code == 422


# ---------- auth: login ----------

def test_login_with_correct_credentials_returns_token(client):
    _register(client, username="alice", password="correct-password")
    resp = client.post("/auth/login", json={"username": "alice", "password": "correct-password"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_login_with_wrong_password_returns_401(client):
    _register(client, username="alice", password="correct-password")
    resp = client.post("/auth/login", json={"username": "alice", "password": "wrong-password"})
    assert resp.status_code == 401


def test_login_unknown_username_returns_401(client):
    resp = client.post("/auth/login", json={"username": "nobody", "password": "anything"})
    assert resp.status_code == 401


# ---------- auth enforcement on protected routes ----------

def test_inbox_priority_without_token_returns_401():
    """HTTPBearer in this FastAPI version returns 401 for a missing
    Authorization header entirely (older/other versions sometimes return
    403 here -- verified against the actually-installed version rather
    than assumed)."""
    client = TestClient(api.app)
    resp = client.get("/inbox/priority")
    assert resp.status_code == 401


def test_inbox_priority_with_garbage_token_returns_401(client):
    resp = client.get("/inbox/priority", headers={"Authorization": "Bearer not-a-real-token"})
    assert resp.status_code == 401


def test_drafts_without_token_returns_401():
    client = TestClient(api.app)
    resp = client.get("/drafts")
    assert resp.status_code == 401


# ---------- GET /inbox/priority (authenticated) ----------

def test_inbox_priority_empty_inbox(client):
    headers = _auth_headers(client)
    with patch("main._open_mailbox", return_value=_fake_mailbox([])):
        resp = client.get("/inbox/priority", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == []


def test_inbox_priority_ranks_and_returns_json(client):
    headers = _auth_headers(client, username="alice", user_email="alice@example.com")
    now = datetime.now(timezone.utc)
    urgent = _fake_mail("1", "URGENT: need this today", "boss@example.com",
                         to=["alice@example.com"], date=now)
    old_newsletter = _fake_mail("2", "weekly digest", "newsletter@example.com",
                                to=["someone-else@example.com"], date=now - timedelta(days=10))

    with patch("main._open_mailbox", return_value=_fake_mailbox([old_newsletter, urgent])):
        resp = client.get("/inbox/priority", headers=headers)

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert body[0]["uid"] == "1"
    assert body[0]["score"] > body[1]["score"]


def test_inbox_priority_login_failure_returns_401(client):
    headers = _auth_headers(client)
    with patch("main._open_mailbox", side_effect=MailboxLoginError("bad creds", b"")):
        resp = client.get("/inbox/priority", headers=headers)
    assert resp.status_code == 401


def test_inbox_priority_connection_failure_returns_503(client):
    headers = _auth_headers(client)
    with patch("main._open_mailbox", side_effect=ConnectionError("unreachable")):
        resp = client.get("/inbox/priority", headers=headers)
    assert resp.status_code == 503


# ---------- per-user isolation (the core multi-user requirement) ----------

def test_two_users_see_independent_drafts(client):
    """Regression-style test for the actual point of multi-user support:
    Alice's drafts must not appear in Bob's /drafts response and vice versa."""
    alice_headers = _auth_headers(client, username="alice", user_email="alice@example.com",
                                    imap_user="alice@gmail.com")
    bob_headers = _auth_headers(client, username="bob", user_email="bob@example.com",
                                  imap_user="bob@outlook.com", imap_host="imap.outlook.com")

    fake_mail = _fake_mail("6", "Subject", "friend@example.com")
    fake_mail.text = "Some content"
    fake_mail.html = ""
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="A reply.\n\nBest,")

    with patch("main._open_mailbox", return_value=_fake_mailbox(iter([fake_mail]))), \
         patch("main.raw_llm", fake_llm):
        gen_resp = client.post("/drafts/generate", json={"uid": "6"}, headers=alice_headers)
    assert gen_resp.status_code == 200

    alice_drafts = client.get("/drafts?status=all", headers=alice_headers)
    bob_drafts = client.get("/drafts?status=all", headers=bob_headers)

    assert len(alice_drafts.json()) == 1
    assert len(bob_drafts.json()) == 0  # Bob must not see Alice's draft


def test_two_users_have_independent_sender_familiarity(client):
    """Alice viewing an email from a sender 5 times should not inflate
    how that same sender scores the first time Bob sees them. (Scoring
    always records the current view, so a brand-new sender for Bob
    still shows familiarity 0.05 for seeing them *this once* -- the
    isolation point is that it's not 6/20 from Alice's history leaking in.)"""
    alice_headers = _auth_headers(client, username="alice", user_email="alice@example.com")
    bob_headers = _auth_headers(client, username="bob", user_email="bob@example.com",
                                  imap_user="bob@outlook.com", imap_host="imap.outlook.com")

    now = datetime.now(timezone.utc)
    mail_from_shared_sender = _fake_mail("1", "Hello", "shared@example.com",
                                          to=["alice@example.com"], date=now)

    # Alice sees this sender 5 times
    with patch("main._open_mailbox", return_value=_fake_mailbox([mail_from_shared_sender] * 5)):
        alice_resp = client.get("/inbox/priority", headers=alice_headers)
    alice_familiarity = alice_resp.json()[0]["score_breakdown"]["familiarity"]

    # Bob sees this sender for the first time in HIS inbox
    mail_for_bob = _fake_mail("1", "Hello", "shared@example.com", to=["bob@example.com"], date=now)
    with patch("main._open_mailbox", return_value=_fake_mailbox([mail_for_bob])):
        bob_resp = client.get("/inbox/priority", headers=bob_headers)
    bob_familiarity = bob_resp.json()[0]["score_breakdown"]["familiarity"]

    # Bob's familiarity reflects only his own first view (1/20 = 0.05),
    # not Alice's 5 prior views leaking into his account
    assert bob_familiarity == pytest.approx(1 / priority.FAMILIARITY_CAP)
    assert alice_familiarity > bob_familiarity


# ---------- GET /drafts (authenticated) ----------

def test_list_drafts_defaults_to_pending_review(client):
    headers = _auth_headers(client)
    store = api.get_user_draft_store("alice")
    d1 = store.save("uid1", "Subject 1", "a@example.com", "Draft 1")
    store.save("uid2", "Subject 2", "b@example.com", "Draft 2")
    store.update_status(d1, "sent")

    resp = client.get("/drafts", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["email_uid"] == "uid2"


def test_list_drafts_status_all_returns_everything(client):
    headers = _auth_headers(client)
    store = api.get_user_draft_store("alice")
    d1 = store.save("uid1", "Subject 1", "a@example.com", "Draft 1")
    store.save("uid2", "Subject 2", "b@example.com", "Draft 2")
    store.update_status(d1, "sent")

    resp = client.get("/drafts?status=all", headers=headers)
    assert resp.status_code == 200
    assert len(resp.json()) == 2


# ---------- GET /drafts/{draft_id} ----------

def test_get_draft_by_id_success(client):
    headers = _auth_headers(client)
    store = api.get_user_draft_store("alice")
    draft_id = store.save("uid1", "Subject", "a@example.com", "Draft text")

    resp = client.get(f"/drafts/{draft_id}", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["draft_text"] == "Draft text"


def test_get_draft_by_id_not_found_returns_404(client):
    headers = _auth_headers(client)
    resp = client.get("/drafts/does-not-exist", headers=headers)
    assert resp.status_code == 404


# ---------- PATCH /drafts/{draft_id} ----------

def test_update_draft_status_success(client):
    headers = _auth_headers(client)
    store = api.get_user_draft_store("alice")
    draft_id = store.save("uid1", "Subject", "a@example.com", "Draft text")

    resp = client.patch(f"/drafts/{draft_id}", json={"status": "approved"}, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "approved"


def test_update_draft_status_invalid_status_returns_422(client):
    headers = _auth_headers(client)
    store = api.get_user_draft_store("alice")
    draft_id = store.save("uid1", "Subject", "a@example.com", "Draft text")

    resp = client.patch(f"/drafts/{draft_id}", json={"status": "not_a_real_status"}, headers=headers)
    assert resp.status_code == 422


def test_update_draft_status_not_found_returns_404(client):
    headers = _auth_headers(client)
    resp = client.patch("/drafts/does-not-exist", json={"status": "approved"}, headers=headers)
    assert resp.status_code == 404


# ---------- POST /drafts/generate ----------

def test_generate_draft_success(client):
    headers = _auth_headers(client)
    fake_mail = _fake_mail("6", "About your faction", "friend@example.com")
    fake_mail.text = "Welcome to the Dauntless faction!"
    fake_mail.html = ""

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="Thank you for the welcome.\n\nBest,")

    with patch("main._open_mailbox", return_value=_fake_mailbox(iter([fake_mail]))), \
         patch("main.raw_llm", fake_llm):
        resp = client.post("/drafts/generate", json={"uid": "6"}, headers=headers)

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pending_review"
    assert "Thank you" in body["draft_text"]


def test_generate_draft_strips_placeholder_signoff(client):
    headers = _auth_headers(client)
    fake_mail = _fake_mail("6", "About your faction", "friend@example.com")
    fake_mail.text = "Welcome to the Dauntless faction!"
    fake_mail.html = ""

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(
        content="Thank you for the welcome.\n\nBest,   \n[User's Name]"
    )

    with patch("main._open_mailbox", return_value=_fake_mailbox(iter([fake_mail]))), \
         patch("main.raw_llm", fake_llm):
        resp = client.post("/drafts/generate", json={"uid": "6"}, headers=headers)

    assert resp.status_code == 200
    assert "[User's Name]" not in resp.json()["draft_text"]


def test_generate_draft_uid_not_found_returns_404(client):
    headers = _auth_headers(client)
    with patch("main._open_mailbox", return_value=_fake_mailbox(iter([]))):
        resp = client.post("/drafts/generate", json={"uid": "999"}, headers=headers)
    assert resp.status_code == 404


def test_generate_draft_llm_failure_returns_502(client):
    headers = _auth_headers(client)
    fake_mail = _fake_mail("1", "Subject", "sender@example.com")
    fake_mail.text = "body"
    fake_mail.html = ""

    broken_llm = MagicMock()
    broken_llm.invoke.side_effect = Exception("connection refused")

    with patch("main._open_mailbox", return_value=_fake_mailbox(iter([fake_mail]))), \
         patch("main.raw_llm", broken_llm):
        resp = client.post("/drafts/generate", json={"uid": "1"}, headers=headers)

    assert resp.status_code == 502


def test_generate_draft_login_failure_returns_401(client):
    headers = _auth_headers(client)
    with patch("main._open_mailbox", side_effect=MailboxLoginError("bad creds", b"")):
        resp = client.post("/drafts/generate", json={"uid": "1"}, headers=headers)
    assert resp.status_code == 401


# ---------- GET/PATCH /digest/config ----------

def test_get_digest_config_returns_defaults(client):
    headers = _auth_headers(client, username="alice", imap_host="imap.gmail.com")
    resp = client.get("/digest/config", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["digest_frequency"] == "daily"
    assert body["digest_enabled"] is True
    assert body["smtp_host"] == "smtp.gmail.com"  # derived from imap.gmail.com
    assert body["last_digest_sent_at"] is None


def test_get_digest_config_requires_auth():
    client = TestClient(api.app)
    resp = client.get("/digest/config")
    assert resp.status_code == 401


def test_update_digest_config_changes_frequency(client):
    headers = _auth_headers(client)
    resp = client.patch("/digest/config", json={"frequency": "weekly"}, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["digest_frequency"] == "weekly"


def test_update_digest_config_disable(client):
    headers = _auth_headers(client)
    resp = client.patch("/digest/config", json={"enabled": False}, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["digest_enabled"] is False


def test_update_digest_config_invalid_frequency_returns_422(client):
    headers = _auth_headers(client)
    resp = client.patch("/digest/config", json={"frequency": "hourly"}, headers=headers)
    assert resp.status_code == 422


def test_update_digest_config_partial_update_preserves_other_fields(client):
    headers = _auth_headers(client)
    client.patch("/digest/config", json={"frequency": "weekly"}, headers=headers)
    resp = client.patch("/digest/config", json={"recipient": "other@example.com"}, headers=headers)
    body = resp.json()
    assert body["digest_recipient"] == "other@example.com"
    assert body["digest_frequency"] == "weekly"  # untouched by second call


def test_register_with_custom_digest_settings(client):
    resp = _register(client, username="custom_user")
    # Re-register a different user with explicit digest overrides
    resp2 = client.post("/auth/register", json={
        "username": "custom_user2", "password": "pw1234567", "user_email": "u2@example.com",
        "imap_host": "imap.gmail.com", "imap_user": "u2@gmail.com", "imap_pass": "app-pw",
        "digest_frequency": "weekly", "digest_enabled": False,
    })
    assert resp2.status_code == 200
    token = resp2.json()["access_token"]
    config_resp = client.get("/digest/config", headers={"Authorization": f"Bearer {token}"})
    assert config_resp.json()["digest_frequency"] == "weekly"
    assert config_resp.json()["digest_enabled"] is False


def test_register_invalid_digest_frequency_returns_422(client):
    resp = client.post("/auth/register", json={
        "username": "bad_freq_user", "password": "pw1234567", "user_email": "u@example.com",
        "imap_host": "imap.gmail.com", "imap_user": "u@gmail.com", "imap_pass": "app-pw",
        "digest_frequency": "hourly",
    })
    assert resp.status_code == 422


def test_two_users_have_independent_digest_config(client):
    alice_headers = _auth_headers(client, username="alice", user_email="alice@example.com")
    bob_headers = _auth_headers(client, username="bob", user_email="bob@example.com",
                                  imap_user="bob@outlook.com", imap_host="imap.outlook.com")

    client.patch("/digest/config", json={"frequency": "weekly"}, headers=alice_headers)

    alice_config = client.get("/digest/config", headers=alice_headers).json()
    bob_config = client.get("/digest/config", headers=bob_headers).json()

    assert alice_config["digest_frequency"] == "weekly"
    assert bob_config["digest_frequency"] == "daily"  # unaffected by Alice's change
