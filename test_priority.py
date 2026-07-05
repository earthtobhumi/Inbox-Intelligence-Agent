"""
Tests for priority.py: scoring signal functions, combination weights,
and the sqlite-backed sender familiarity store.
"""
import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

import priority


# ---------- urgency_keyword_score ----------

@pytest.mark.parametrize("subject", [
    "URGENT: need your response",
    "Action required by EOD",
    "Reminder: deadline tomorrow",
    "asap please review",
])
def test_urgency_keyword_score_detects_urgent_subjects(subject):
    assert priority.urgency_keyword_score(subject) == 1.0


@pytest.mark.parametrize("subject", [
    "Weekly newsletter",
    "Your order has shipped",
    "",
    None,
])
def test_urgency_keyword_score_ignores_non_urgent_subjects(subject):
    assert priority.urgency_keyword_score(subject) == 0.0


# ---------- directness_score ----------

def test_directness_score_direct_recipient():
    score = priority.directness_score(["me@example.com"], [], "me@example.com")
    assert score == 1.0


def test_directness_score_cc_only():
    score = priority.directness_score(["other@example.com"], ["me@example.com"], "me@example.com")
    assert score == 0.5


def test_directness_score_neither():
    score = priority.directness_score(["other@example.com"], [], "me@example.com")
    assert score == 0.0


def test_directness_score_handles_display_names():
    # Real headers often look like "Jane Doe <me@example.com>"
    score = priority.directness_score(["Jane Doe <me@example.com>"], [], "me@example.com")
    assert score == 1.0


# ---------- recency_score ----------

def test_recency_score_very_recent_email():
    now = datetime.now(timezone.utc)
    score = priority.recency_score(now - timedelta(minutes=10), now=now)
    assert score == 1.0


def test_recency_score_old_email_floors_at_zero():
    now = datetime.now(timezone.utc)
    score = priority.recency_score(now - timedelta(days=30), now=now)
    assert score == 0.0


def test_recency_score_decays_between_bounds():
    now = datetime.now(timezone.utc)
    score = priority.recency_score(now - timedelta(days=3), now=now)
    assert 0.0 < score < 1.0


def test_recency_score_handles_naive_datetime():
    # imap_tools may return naive datetimes depending on the message;
    # this should not raise a tz-aware/naive comparison error.
    now = datetime.now(timezone.utc)
    naive_recent = datetime.now().replace(tzinfo=None)
    score = priority.recency_score(naive_recent, now=now)
    assert 0.0 <= score <= 1.0


# ---------- familiarity_score ----------

def test_familiarity_score_zero_for_unseen_sender():
    assert priority.familiarity_score(0) == 0.0


def test_familiarity_score_scales_with_count():
    assert priority.familiarity_score(5) == 5 / priority.FAMILIARITY_CAP


def test_familiarity_score_caps_at_one():
    assert priority.familiarity_score(priority.FAMILIARITY_CAP * 10) == 1.0


# ---------- combine_score ----------

def test_combine_score_all_signals_present_gives_high_score():
    score = priority.combine_score(urgency=1.0, directness=1.0, familiarity=1.0, recency=1.0)
    assert score == 100.0


def test_combine_score_no_signals_gives_zero():
    score = priority.combine_score(urgency=0.0, directness=0.0, familiarity=0.0, recency=0.0)
    assert score == 0.0


def test_combine_score_weights_sum_to_one():
    total = (
        priority.WEIGHT_URGENCY
        + priority.WEIGHT_DIRECTNESS
        + priority.WEIGHT_FAMILIARITY
        + priority.WEIGHT_RECENCY
    )
    assert abs(total - 1.0) < 1e-9


# ---------- SenderStats (real sqlite, temp file) ----------

@pytest.fixture
def sender_stats():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    stats = priority.SenderStats(db_path=path)
    yield stats
    stats.close()
    os.remove(path)


def test_sender_stats_starts_at_zero_for_unseen_sender(sender_stats):
    assert sender_stats.get_count("nobody@example.com") == 0


def test_sender_stats_increments_on_record(sender_stats):
    sender_stats.record("boss@example.com")
    sender_stats.record("boss@example.com")
    assert sender_stats.get_count("boss@example.com") == 2


def test_sender_stats_normalizes_display_name_format(sender_stats):
    sender_stats.record("Boss Person <boss@example.com>")
    # Should be counted under the bare address regardless of display name
    assert sender_stats.get_count("boss@example.com") == 1
    assert sender_stats.get_count("Boss Person <boss@example.com>") == 1


def test_sender_stats_persists_across_reopen():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        s1 = priority.SenderStats(db_path=path)
        s1.record("a@example.com")
        s1.close()

        s2 = priority.SenderStats(db_path=path)
        assert s2.get_count("a@example.com") == 1
        s2.close()
    finally:
        os.remove(path)


# ---------- is_automated_sender ----------

@pytest.mark.parametrize("sender", [
    "no-reply@google.com",
    "noreply@accounts.google.com",
    "do-not-reply@example.com",
    "notifications@github.com",
    "security@paypal.com",
    "MAILER-DAEMON@mail.example.com",
])
def test_is_automated_sender_detects_common_patterns(sender):
    assert priority.is_automated_sender(sender) is True


@pytest.mark.parametrize("sender", [
    "friend@gmail.com",
    "boss@company.com",
    "",
    None,
])
def test_is_automated_sender_ignores_human_senders(sender):
    assert priority.is_automated_sender(sender) is False


# ---------- familiarity suppression for automated senders ----------

def test_score_email_suppresses_familiarity_for_automated_sender(sender_stats):
    email = _make_email(from_="no-reply@accounts.google.com")
    # Simulate having "seen" this sender many times already
    for _ in range(10):
        sender_stats.record("no-reply@accounts.google.com")

    result = priority.score_email(email, "me@example.com", sender_stats, record_interaction=False)
    assert result["score_breakdown"]["familiarity"] == 0.0
    assert any("automated" in r.lower() for r in result["reasons"])


def test_score_email_does_not_suppress_familiarity_for_human_sender(sender_stats):
    email = _make_email(from_="friend@example.com")
    for _ in range(5):
        sender_stats.record("friend@example.com")

    result = priority.score_email(email, "me@example.com", sender_stats, record_interaction=False)
    assert result["score_breakdown"]["familiarity"] > 0.0


def test_rank_emails_personal_email_outranks_frequent_automated_sender(sender_stats):
    """Regression test for the real-world case that surfaced this bug:
    a Google security alert (seen many times) was outranking a one-off
    personal email, purely due to sender frequency."""
    now = datetime.now(timezone.utc)

    # Simulate a Google no-reply sender seen many times before
    for _ in range(15):
        sender_stats.record("no-reply@accounts.google.com")

    google_alert = _make_email(
        uid="1", subject="Security alert", from_="no-reply@accounts.google.com",
        to=["me@example.com"], date=now,
    )
    personal_email = _make_email(
        uid="2", subject="About your project", from_="friend@example.com",
        to=["me@example.com"], date=now,
    )

    ranked = priority.rank_emails([google_alert, personal_email], "me@example.com", sender_stats)

    assert ranked[0]["uid"] == "2"  # personal email should now rank first


# ---------- score_email / rank_emails ----------

def _make_email(uid="1", subject="hello", from_="a@example.com",
                 to=None, cc=None, date=None):
    return {
        "uid": uid,
        "subject": subject,
        "from": from_,
        "to": to or [],
        "cc": cc or [],
        "date": date or datetime.now(timezone.utc),
    }


def test_score_email_returns_expected_shape(sender_stats):
    email = _make_email()
    result = priority.score_email(email, "me@example.com", sender_stats)
    assert "score" in result
    assert "score_breakdown" in result
    assert "reasons" in result
    assert 0.0 <= result["score"] <= 100.0


def test_score_email_records_interaction_by_default(sender_stats):
    email = _make_email(from_="sender@example.com")
    priority.score_email(email, "me@example.com", sender_stats)
    assert sender_stats.get_count("sender@example.com") == 1


def test_score_email_can_skip_recording(sender_stats):
    email = _make_email(from_="sender@example.com")
    priority.score_email(email, "me@example.com", sender_stats, record_interaction=False)
    assert sender_stats.get_count("sender@example.com") == 0


def test_rank_emails_sorts_descending_by_score(sender_stats):
    now = datetime.now(timezone.utc)
    urgent_direct = _make_email(
        uid="1", subject="URGENT: action required",
        to=["me@example.com"], date=now,
    )
    quiet_old = _make_email(
        uid="2", subject="fyi newsletter", to=["someone-else@example.com"],
        date=now - timedelta(days=10),
    )

    ranked = priority.rank_emails([quiet_old, urgent_direct], "me@example.com", sender_stats)

    assert ranked[0]["uid"] == "1"
    assert ranked[0]["score"] >= ranked[1]["score"]
