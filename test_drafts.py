"""
Tests for drafts.py: prompt construction and sqlite-backed DraftStore.
Does NOT test actual LLM draft quality -- that requires a live Ollama run.
"""
import os
import tempfile

import pytest

import drafts


# ---------- build_draft_prompt ----------

def test_build_draft_prompt_includes_original_content():
    prompt = drafts.build_draft_prompt(
        subject="Meeting tomorrow",
        sender="boss@example.com",
        body="Can we move our meeting to 3pm?",
    )
    assert "Meeting tomorrow" in prompt
    assert "boss@example.com" in prompt
    assert "Can we move our meeting to 3pm?" in prompt


def test_build_draft_prompt_includes_grounding_rules():
    prompt = drafts.build_draft_prompt("Subject", "sender@example.com", "body")
    assert "invent" in prompt.lower()
    assert "reviewed by the user before sending" in prompt.lower()


def test_build_draft_prompt_forbids_placeholder_signoff():
    """Regression test: a live run produced '[User's Name]' as a literal
    placeholder because 'leave the name blank' was ambiguous. The prompt
    must now explicitly forbid placeholder brackets."""
    prompt = drafts.build_draft_prompt("Subject", "sender@example.com", "body")
    assert "no placeholder text" in prompt.lower()
    assert "[your name]" in prompt.lower() or "[user's name]" in prompt.lower()


# ---------- strip_placeholder_signoff ----------

def test_strip_placeholder_signoff_removes_real_world_case():
    """Exact string observed live, three times, despite the prompt
    explicitly forbidding it -- confirms this needs code, not prompt wording."""
    raw = "Thank you for the welcoming message. I am excited to join the Dauntless faction.\n\nBest,   \n[User's Name]"
    cleaned = drafts.strip_placeholder_signoff(raw)
    assert "[User's Name]" not in cleaned
    assert cleaned.endswith("Best,")
    assert "Dauntless faction" in cleaned  # real content untouched


@pytest.mark.parametrize("placeholder", [
    "[Your Name]", "[User's Name]", "[Name]", "[Sender Name]",
    "[Full Name]", "(Your Name)", "[YOUR NAME]", "[your name]",
])
def test_strip_placeholder_signoff_catches_common_variants(placeholder):
    raw = f"Thanks for reaching out.\n\nBest,\n{placeholder}"
    cleaned = drafts.strip_placeholder_signoff(raw)
    assert placeholder not in cleaned


def test_strip_placeholder_signoff_leaves_real_names_alone():
    raw = "Thanks for reaching out.\n\nBest,\nJohn Smith"
    cleaned = drafts.strip_placeholder_signoff(raw)
    assert "John Smith" in cleaned


def test_strip_placeholder_signoff_handles_empty_string():
    assert drafts.strip_placeholder_signoff("") == ""


def test_strip_placeholder_signoff_no_placeholder_present():
    raw = "Thanks for reaching out.\n\nBest,"
    assert drafts.strip_placeholder_signoff(raw) == raw


# ---------- DraftStore ----------

@pytest.fixture
def draft_store():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    store = drafts.DraftStore(db_path=path)
    yield store
    store.close()
    os.remove(path)


def test_save_returns_draft_id(draft_store):
    draft_id = draft_store.save("uid123", "Subject", "sender@example.com", "Draft text here")
    assert draft_id
    assert isinstance(draft_id, str)


def test_get_returns_saved_draft(draft_store):
    draft_id = draft_store.save("uid123", "Subject", "sender@example.com", "Draft text here")
    result = draft_store.get(draft_id)
    assert result["email_uid"] == "uid123"
    assert result["subject"] == "Subject"
    assert result["draft_text"] == "Draft text here"
    assert result["status"] == "pending_review"


def test_get_returns_none_for_unknown_id(draft_store):
    assert draft_store.get("does-not-exist") is None


def test_list_all_returns_all_drafts(draft_store):
    draft_store.save("uid1", "Subject 1", "a@example.com", "Draft 1")
    draft_store.save("uid2", "Subject 2", "b@example.com", "Draft 2")
    all_drafts = draft_store.list_all()
    assert len(all_drafts) == 2


def test_list_all_filters_by_status(draft_store):
    d1 = draft_store.save("uid1", "Subject 1", "a@example.com", "Draft 1")
    draft_store.save("uid2", "Subject 2", "b@example.com", "Draft 2")
    draft_store.update_status(d1, "sent")

    pending = draft_store.list_all(status="pending_review")
    sent = draft_store.list_all(status="sent")
    assert len(pending) == 1
    assert len(sent) == 1
    assert pending[0]["email_uid"] == "uid2"


def test_update_status_returns_true_on_success(draft_store):
    draft_id = draft_store.save("uid1", "Subject", "a@example.com", "Draft")
    assert draft_store.update_status(draft_id, "approved") is True
    assert draft_store.get(draft_id)["status"] == "approved"


def test_update_status_returns_false_for_unknown_id(draft_store):
    assert draft_store.update_status("does-not-exist", "approved") is False


def test_drafts_persist_across_reopen():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        s1 = drafts.DraftStore(db_path=path)
        draft_id = s1.save("uid1", "Subject", "a@example.com", "Draft")
        s1.close()

        s2 = drafts.DraftStore(db_path=path)
        result = s2.get(draft_id)
        assert result is not None
        assert result["draft_text"] == "Draft"
        s2.close()
    finally:
        os.remove(path)


def test_multiple_drafts_can_exist_for_same_email_uid(draft_store):
    """A user might ask for a redraft -- each call should get its own
    draft_id rather than overwriting the previous attempt."""
    id1 = draft_store.save("uid1", "Subject", "a@example.com", "First attempt")
    id2 = draft_store.save("uid1", "Subject", "a@example.com", "Second attempt")
    assert id1 != id2
    assert len(draft_store.list_all()) == 2
