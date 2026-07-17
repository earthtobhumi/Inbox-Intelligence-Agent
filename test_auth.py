"""
Tests for auth.py: password hashing, JWT create/decode, Fernet encryption
of stored credentials, and UserStore CRUD/authentication.
"""
import os
import tempfile
import time

import pytest
from cryptography.fernet import Fernet
from jose import JWTError

import auth


# ---------- password hashing ----------

def test_hash_password_and_verify_correct():
    hashed = auth.hash_password("correct-password-123")
    assert auth.verify_password("correct-password-123", hashed) is True


def test_verify_password_rejects_wrong_password():
    hashed = auth.hash_password("correct-password-123")
    assert auth.verify_password("wrong-password", hashed) is False


def test_hash_password_rejects_overly_long_password():
    too_long = "a" * 100  # bcrypt's hard limit is 72 bytes
    with pytest.raises(auth.PasswordTooLongError):
        auth.hash_password(too_long)


def test_verify_password_returns_false_not_raises_for_overly_long_input():
    hashed = auth.hash_password("short-password")
    assert auth.verify_password("a" * 100, hashed) is False


def test_two_hashes_of_same_password_differ():
    """bcrypt salts each hash -- confirms we're not accidentally using a
    fixed salt or storing anything reversible."""
    h1 = auth.hash_password("same-password")
    h2 = auth.hash_password("same-password")
    assert h1 != h2
    assert auth.verify_password("same-password", h1)
    assert auth.verify_password("same-password", h2)


# ---------- JWT ----------

@pytest.fixture(autouse=True)
def _fernet_key(monkeypatch):
    """Ensure encryption tests have a real key regardless of environment."""
    monkeypatch.setattr(auth, "_fernet", Fernet(Fernet.generate_key()))


def test_create_and_decode_access_token_roundtrip():
    token = auth.create_access_token("alice")
    username = auth.decode_access_token(token)
    assert username == "alice"


def test_decode_access_token_rejects_garbage_token():
    with pytest.raises(JWTError):
        auth.decode_access_token("not.a.real.token")


def test_decode_access_token_rejects_expired_token():
    # expires_minutes as a negative number -> already expired
    token = auth.create_access_token("alice", expires_minutes=-1)
    with pytest.raises(JWTError):
        auth.decode_access_token(token)


def test_decode_access_token_rejects_token_signed_with_different_secret():
    from jose import jwt as jose_jwt
    bad_token = jose_jwt.encode({"sub": "alice"}, "a-completely-different-secret", algorithm="HS256")
    with pytest.raises(JWTError):
        auth.decode_access_token(bad_token)


# ---------- encryption ----------

def test_encrypt_decrypt_roundtrip():
    encrypted = auth.encrypt_secret("my-imap-app-password")
    assert encrypted != "my-imap-app-password"  # confirms it's actually encrypted, not passthrough
    decrypted = auth.decrypt_secret(encrypted)
    assert decrypted == "my-imap-app-password"


def test_encrypt_raises_without_fernet_key(monkeypatch):
    monkeypatch.setattr(auth, "_fernet", None)
    with pytest.raises(RuntimeError):
        auth.encrypt_secret("anything")


# ---------- UserStore ----------

@pytest.fixture
def user_store():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    store = auth.UserStore(db_path=path)
    yield store
    store.close()
    os.remove(path)


def test_create_and_get_user(user_store):
    user_store.create_user(
        username="alice", password="pw12345", user_email="alice@example.com",
        imap_host="imap.gmail.com", imap_user="alice@gmail.com", imap_pass="app-password-123",
    )
    user = user_store.get_user("alice")
    assert user is not None
    assert user["user_email"] == "alice@example.com"
    assert user["imap_host"] == "imap.gmail.com"
    # confirm the stored password is encrypted, not plaintext
    assert user["imap_pass_encrypted"] != "app-password-123"


def test_create_user_duplicate_username_raises(user_store):
    user_store.create_user(
        username="alice", password="pw12345", user_email="alice@example.com",
        imap_host="imap.gmail.com", imap_user="alice@gmail.com", imap_pass="pw",
    )
    with pytest.raises(ValueError):
        user_store.create_user(
            username="alice", password="different-pw", user_email="alice2@example.com",
            imap_host="imap.gmail.com", imap_user="alice2@gmail.com", imap_pass="pw2",
        )


def test_get_user_returns_none_for_unknown_username(user_store):
    assert user_store.get_user("nobody") is None


def test_authenticate_success(user_store):
    user_store.create_user(
        username="alice", password="correct-password", user_email="alice@example.com",
        imap_host="imap.gmail.com", imap_user="alice@gmail.com", imap_pass="pw",
    )
    result = user_store.authenticate("alice", "correct-password")
    assert result is not None
    assert result["username"] == "alice"


def test_authenticate_wrong_password_returns_none(user_store):
    user_store.create_user(
        username="alice", password="correct-password", user_email="alice@example.com",
        imap_host="imap.gmail.com", imap_user="alice@gmail.com", imap_pass="pw",
    )
    assert user_store.authenticate("alice", "wrong-password") is None


def test_authenticate_unknown_user_returns_none(user_store):
    assert user_store.authenticate("nobody", "any-password") is None


def test_users_persist_across_reopen():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        s1 = auth.UserStore(db_path=path)
        s1.create_user(
            username="alice", password="pw12345", user_email="alice@example.com",
            imap_host="imap.gmail.com", imap_user="alice@gmail.com", imap_pass="pw",
        )
        s1.close()

        s2 = auth.UserStore(db_path=path)
        assert s2.authenticate("alice", "pw12345") is not None
        s2.close()
    finally:
        os.remove(path)


def test_two_users_are_fully_independent(user_store):
    """Regression-style test for the core multi-user requirement: two
    accounts must have independent credentials and not collide."""
    user_store.create_user(
        username="alice", password="alice-pw", user_email="alice@example.com",
        imap_host="imap.gmail.com", imap_user="alice@gmail.com", imap_pass="alice-imap-pw",
    )
    user_store.create_user(
        username="bob", password="bob-pw", user_email="bob@example.com",
        imap_host="imap.outlook.com", imap_user="bob@outlook.com", imap_pass="bob-imap-pw",
    )

    assert user_store.authenticate("alice", "bob-pw") is None
    assert user_store.authenticate("bob", "alice-pw") is None
    assert user_store.authenticate("alice", "alice-pw")["user_email"] == "alice@example.com"
    assert user_store.authenticate("bob", "bob-pw")["user_email"] == "bob@example.com"


# ---------- digest config ----------

def test_create_user_defaults_digest_config(user_store):
    user_store.create_user(
        username="alice", password="pw12345", user_email="alice@example.com",
        imap_host="imap.gmail.com", imap_user="alice@gmail.com", imap_pass="pw",
    )
    user = user_store.get_user("alice")
    assert user["digest_frequency"] == "daily"
    assert user["digest_enabled"] is True
    assert user["digest_recipient"] == "alice@example.com"  # defaults to user_email
    assert user["last_digest_sent_at"] is None


def test_create_user_derives_smtp_host_from_imap_host(user_store):
    user_store.create_user(
        username="alice", password="pw12345", user_email="alice@example.com",
        imap_host="imap.gmail.com", imap_user="alice@gmail.com", imap_pass="pw",
    )
    assert user_store.get_user("alice")["smtp_host"] == "smtp.gmail.com"


def test_create_user_explicit_smtp_host_overrides_derivation(user_store):
    user_store.create_user(
        username="alice", password="pw12345", user_email="alice@example.com",
        imap_host="imap.gmail.com", imap_user="alice@gmail.com", imap_pass="pw",
        smtp_host="custom-smtp.example.com",
    )
    assert user_store.get_user("alice")["smtp_host"] == "custom-smtp.example.com"


def test_create_user_unrecognized_imap_host_leaves_smtp_host_none(user_store):
    user_store.create_user(
        username="alice", password="pw12345", user_email="alice@example.com",
        imap_host="mail.somehost.example.com", imap_user="alice@example.com", imap_pass="pw",
    )
    assert user_store.get_user("alice")["smtp_host"] is None


def test_update_digest_config_partial_update(user_store):
    user_store.create_user(
        username="alice", password="pw12345", user_email="alice@example.com",
        imap_host="imap.gmail.com", imap_user="alice@gmail.com", imap_pass="pw",
    )
    ok = user_store.update_digest_config("alice", frequency="weekly")
    assert ok is True
    user = user_store.get_user("alice")
    assert user["digest_frequency"] == "weekly"
    assert user["digest_enabled"] is True  # untouched fields preserved


def test_update_digest_config_disable(user_store):
    user_store.create_user(
        username="alice", password="pw12345", user_email="alice@example.com",
        imap_host="imap.gmail.com", imap_user="alice@gmail.com", imap_pass="pw",
    )
    user_store.update_digest_config("alice", enabled=False)
    assert user_store.get_user("alice")["digest_enabled"] is False


def test_update_digest_config_unknown_user_returns_false(user_store):
    assert user_store.update_digest_config("nobody", frequency="weekly") is False


def test_mark_digest_sent_updates_timestamp(user_store):
    user_store.create_user(
        username="alice", password="pw12345", user_email="alice@example.com",
        imap_host="imap.gmail.com", imap_user="alice@gmail.com", imap_pass="pw",
    )
    assert user_store.get_user("alice")["last_digest_sent_at"] is None
    user_store.mark_digest_sent("alice")
    assert user_store.get_user("alice")["last_digest_sent_at"] is not None


def test_get_digest_enabled_users_excludes_disabled(user_store):
    user_store.create_user(
        username="alice", password="pw1", user_email="alice@example.com",
        imap_host="imap.gmail.com", imap_user="alice@gmail.com", imap_pass="pw",
    )
    user_store.create_user(
        username="bob", password="pw2", user_email="bob@example.com",
        imap_host="imap.gmail.com", imap_user="bob@gmail.com", imap_pass="pw",
        digest_enabled=False,
    )
    enabled = user_store.get_digest_enabled_users()
    assert len(enabled) == 1
    assert enabled[0]["username"] == "alice"


def test_get_all_users_returns_everyone_regardless_of_digest_status(user_store):
    user_store.create_user(
        username="alice", password="pw1", user_email="alice@example.com",
        imap_host="imap.gmail.com", imap_user="alice@gmail.com", imap_pass="pw",
    )
    user_store.create_user(
        username="bob", password="pw2", user_email="bob@example.com",
        imap_host="imap.gmail.com", imap_user="bob@gmail.com", imap_pass="pw",
        digest_enabled=False,
    )
    assert len(user_store.get_all_users()) == 2
