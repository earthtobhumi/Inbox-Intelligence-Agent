"""
Tests for digest.py: due-date calculation, digest content generation
(pure functions, fully testable without I/O), and SMTP sending with a
mocked smtplib.SMTP.
"""
import smtplib
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

import digest


# ---------- is_digest_due ----------

def test_is_digest_due_true_when_never_sent():
    assert digest.is_digest_due("daily", None) is True


def test_is_digest_due_false_when_sent_recently_daily():
    now = datetime.now(timezone.utc)
    last_sent = (now - timedelta(hours=2)).isoformat()
    assert digest.is_digest_due("daily", last_sent, now=now) is False


def test_is_digest_due_true_when_a_full_day_has_passed():
    now = datetime.now(timezone.utc)
    last_sent = (now - timedelta(days=1, minutes=1)).isoformat()
    assert digest.is_digest_due("daily", last_sent, now=now) is True


def test_is_digest_due_false_when_sent_recently_weekly():
    now = datetime.now(timezone.utc)
    last_sent = (now - timedelta(days=3)).isoformat()
    assert digest.is_digest_due("weekly", last_sent, now=now) is False


def test_is_digest_due_true_when_a_full_week_has_passed():
    now = datetime.now(timezone.utc)
    last_sent = (now - timedelta(days=7, minutes=1)).isoformat()
    assert digest.is_digest_due("weekly", last_sent, now=now) is True


def test_is_digest_due_handles_naive_datetime_string():
    now = datetime.now(timezone.utc)
    naive_last_sent = (now - timedelta(days=2)).replace(tzinfo=None).isoformat()
    # Should not raise a tz-aware/naive comparison error
    assert digest.is_digest_due("daily", naive_last_sent, now=now) is True


def test_is_digest_due_fails_safe_on_unknown_frequency():
    """An unrecognized frequency value should not send rather than guess --
    a malformed config shouldn't spam the user."""
    assert digest.is_digest_due("hourly", None) is True  # never sent -> always due regardless
    now = datetime.now(timezone.utc)
    last_sent = (now - timedelta(days=100)).isoformat()
    assert digest.is_digest_due("hourly", last_sent, now=now) is False


# ---------- build_digest_content ----------

def _ranked_email(score, subject, from_, reasons=None):
    return {
        "uid": "1", "score": score, "subject": subject, "from": from_,
        "reasons": reasons or [], "score_breakdown": {}, "date": "2026-01-01 00:00:00",
    }


def test_build_digest_content_empty_inbox():
    subject, body = digest.build_digest_content([], "daily")
    assert "0 unread" in subject
    assert "clear" in body.lower()


def test_build_digest_content_includes_count_and_subjects():
    emails = [
        _ranked_email(80, "Urgent thing", "boss@example.com", ["urgency language"]),
        _ranked_email(20, "Newsletter", "news@example.com", []),
    ]
    subject, body = digest.build_digest_content(emails, "daily")
    assert "2 unread" in subject
    assert "Urgent thing" in body
    assert "Newsletter" in body
    assert "boss@example.com" in body


def test_build_digest_content_counts_high_priority_correctly():
    emails = [
        _ranked_email(80, "High 1", "a@example.com"),
        _ranked_email(60, "High 2", "b@example.com"),
        _ranked_email(30, "Low", "c@example.com"),
    ]
    _, body = digest.build_digest_content(emails, "daily")
    assert "2 flagged as high priority" in body


def test_build_digest_content_truncates_to_top_ten_with_overflow_note():
    emails = [_ranked_email(50 - i, f"Email {i}", "a@example.com") for i in range(15)]
    _, body = digest.build_digest_content(emails, "daily")
    assert "Email 0" in body
    assert "Email 9" in body
    assert "Email 10" not in body
    assert "5 more" in body


def test_build_digest_content_weekly_uses_week_wording():
    subject, body = digest.build_digest_content([], "weekly")
    assert "weekly" in subject
    assert "week" in body.lower()


# ---------- send_digest_email ----------

def test_send_digest_email_calls_smtp_correctly():
    mock_server = MagicMock()
    mock_server.__enter__.return_value = mock_server
    mock_server.__exit__.return_value = False

    with patch("smtplib.SMTP", return_value=mock_server) as mock_smtp_cls:
        digest.send_digest_email(
            smtp_host="smtp.gmail.com", smtp_port=587,
            smtp_user="me@gmail.com", smtp_pass="app-pw",
            recipient="me@gmail.com", subject="Test digest", body="Body text",
        )

    mock_smtp_cls.assert_called_once_with("smtp.gmail.com", 587, timeout=30)
    mock_server.starttls.assert_called_once()
    mock_server.login.assert_called_once_with("me@gmail.com", "app-pw")
    assert mock_server.sendmail.call_count == 1
    call_args = mock_server.sendmail.call_args[0]
    assert call_args[0] == "me@gmail.com"
    assert call_args[1] == ["me@gmail.com"]
    assert "Test digest" in call_args[2]
    assert "Body text" in call_args[2]


def test_send_digest_email_propagates_smtp_failure():
    """Sending failures must propagate, not be silently swallowed --
    the Celery task layer is responsible for catching/logging/retrying."""
    mock_server = MagicMock()
    mock_server.__enter__.return_value = mock_server
    mock_server.__exit__.return_value = False
    mock_server.login.side_effect = smtplib.SMTPAuthenticationError(535, b"bad credentials")

    with patch("smtplib.SMTP", return_value=mock_server):
        with pytest.raises(smtplib.SMTPAuthenticationError):
            digest.send_digest_email(
                smtp_host="smtp.gmail.com", smtp_port=587,
                smtp_user="me@gmail.com", smtp_pass="wrong-pw",
                recipient="me@gmail.com", subject="Test", body="Body",
            )
