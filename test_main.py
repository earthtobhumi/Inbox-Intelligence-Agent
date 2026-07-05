"""
Tests for the email agent's non-LLM, non-network logic: error handling,
body truncation, graph routing, and checkpointer persistence.

These use mocks for IMAP and the LLM. They do NOT verify:
  - actual draft/summary quality from qwen2.5:7b
  - actual IMAP server behavior
  - actual Ollama connectivity
Those require a live run against a real mailbox + running Ollama instance.
"""
import json
from unittest.mock import MagicMock, patch

import pytest
from imap_tools.errors import MailboxLoginError
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

import main


# ---------- _clean_body ----------

def test_clean_body_prefers_plaintext():
    result = main._clean_body("plain text body", "<p>html body</p>")
    assert result == "plain text body"


def test_clean_body_falls_back_to_html_strip():
    result = main._clean_body("", "<p>Hello <b>world</b></p>")
    assert "<" not in result
    assert "Hello" in result and "world" in result


def test_clean_body_truncates_long_content():
    long_text = "a" * 10000
    result = main._clean_body(long_text, "")
    assert len(result) <= main.MAX_EMAIL_CHARS + len("\n[...truncated...]")
    assert result.endswith("[...truncated...]")


def test_clean_body_empty_input():
    assert main._clean_body("", "") == ""


# ---------- list_unread_emails tool ----------

def test_list_unread_emails_returns_json_on_success():
    fake_mail = MagicMock()
    fake_mail.uid = "123"
    fake_mail.subject = "Test Subject"
    fake_mail.from_ = "sender@example.com"
    fake_mail.date.strftime.return_value = "2026-07-03 10:00:00"

    fake_mailbox = MagicMock()
    fake_mailbox.__enter__.return_value = fake_mailbox
    fake_mailbox.__exit__.return_value = False
    fake_mailbox.fetch.return_value = [fake_mail]

    with patch("main.connect", return_value=fake_mailbox):
        result = main.list_unread_emails.invoke({})

    parsed = json.loads(result)
    assert len(parsed) == 1
    assert parsed[0]["uid"] == "123"
    assert parsed[0]["subject"] == "Test Subject"


def test_list_unread_emails_empty_inbox_returns_empty_list():
    fake_mailbox = MagicMock()
    fake_mailbox.__enter__.return_value = fake_mailbox
    fake_mailbox.__exit__.return_value = False
    fake_mailbox.fetch.return_value = []

    with patch("main.connect", return_value=fake_mailbox):
        result = main.list_unread_emails.invoke({})

    assert json.loads(result) == []


def test_list_unread_emails_handles_login_failure_gracefully():
    with patch("main.connect", side_effect=MailboxLoginError("bad creds", b"")):
        result = main.list_unread_emails.invoke({})

    assert result.startswith("Error:")
    assert "login failed" in result.lower()


def test_list_unread_emails_handles_connection_error_gracefully():
    with patch("main.connect", side_effect=ConnectionError("host unreachable")):
        result = main.list_unread_emails.invoke({})

    assert result.startswith("Error:")


# ---------- summarize_email tool ----------

def test_summarize_email_uid_not_found():
    fake_mailbox = MagicMock()
    fake_mailbox.__enter__.return_value = fake_mailbox
    fake_mailbox.__exit__.return_value = False
    fake_mailbox.fetch.return_value = iter([])  # no matching email

    with patch("main.connect", return_value=fake_mailbox):
        result = main.summarize_email.invoke({"uid": "999"})  # valid-looking numeric uid

    assert result.startswith("Error:")
    assert "no email found" in result.lower()


def test_summarize_email_malformed_uid_does_not_crash():
    """Regression test: a hallucinated/non-numeric UID from the LLM must
    come back as an error string, not raise and crash the tool node."""
    fake_mailbox = MagicMock()
    fake_mailbox.__enter__.return_value = fake_mailbox
    fake_mailbox.__exit__.return_value = False

    with patch("main.connect", return_value=fake_mailbox):
        result = main.summarize_email.invoke({"uid": "does-not-exist"})

    assert result.startswith("Error:")
    assert "not a valid" in result.lower()


def test_summarize_email_handles_login_failure_gracefully():
    with patch("main.connect", side_effect=MailboxLoginError("bad creds", b"")):
        result = main.summarize_email.invoke({"uid": "1"})

    assert result.startswith("Error:")


def test_summarize_email_handles_llm_failure_gracefully():
    fake_mail = MagicMock()
    fake_mail.subject = "Subject"
    fake_mail.from_ = "sender@example.com"
    fake_mail.date = "2026-07-03"
    fake_mail.text = "body text"
    fake_mail.html = ""

    fake_mailbox = MagicMock()
    fake_mailbox.__enter__.return_value = fake_mailbox
    fake_mailbox.__exit__.return_value = False
    fake_mailbox.fetch.return_value = iter([fake_mail])

    broken_llm = MagicMock()
    broken_llm.invoke.side_effect = Exception("connection refused")

    with patch("main.connect", return_value=fake_mailbox), \
         patch("main.raw_llm", broken_llm):
        result = main.summarize_email.invoke({"uid": "1"})

    assert result.startswith("Error:")
    assert "ollama" in result.lower()


# ---------- score_priority_emails tool ----------

def test_score_priority_emails_returns_ranked_json(tmp_path):
    now_email = MagicMock()
    now_email.uid = "1"
    now_email.subject = "URGENT: need this today"
    now_email.from_ = "boss@example.com"
    now_email.to = ["me@example.com"]
    now_email.cc = []
    now_email.date = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)

    old_email = MagicMock()
    old_email.uid = "2"
    old_email.subject = "weekly digest"
    old_email.from_ = "newsletter@example.com"
    old_email.to = ["someone-else@example.com"]
    old_email.cc = []
    old_email.date = now_email.date - __import__("datetime").timedelta(days=10)

    fake_mailbox = MagicMock()
    fake_mailbox.__enter__.return_value = fake_mailbox
    fake_mailbox.__exit__.return_value = False
    fake_mailbox.fetch.return_value = [old_email, now_email]

    fake_stats = main.priority.SenderStats(db_path=str(tmp_path / "test_stats.sqlite"))

    with patch("main.connect", return_value=fake_mailbox), \
         patch("main.USER_EMAIL", "me@example.com"), \
         patch("main.get_sender_stats", return_value=fake_stats):
        result = main.score_priority_emails.invoke({})

    parsed = json.loads(result)
    assert len(parsed) == 2
    # the urgent, directly-addressed, recent email should outrank the old newsletter
    assert parsed[0]["uid"] == "1"
    assert parsed[0]["score"] > parsed[1]["score"]
    assert "score_breakdown" in parsed[0]
    assert "reasons" in parsed[0]

    fake_stats.close()


def test_score_priority_emails_empty_inbox():
    fake_mailbox = MagicMock()
    fake_mailbox.__enter__.return_value = fake_mailbox
    fake_mailbox.__exit__.return_value = False
    fake_mailbox.fetch.return_value = []

    with patch("main.connect", return_value=fake_mailbox):
        result = main.score_priority_emails.invoke({})

    assert json.loads(result) == []


def test_score_priority_emails_handles_login_failure_gracefully():
    with patch("main.connect", side_effect=MailboxLoginError("bad creds", b"")):
        result = main.score_priority_emails.invoke({})

    assert result.startswith("Error:")
    assert "login failed" in result.lower()


# ---------- draft_reply / list_drafts tools ----------

def test_draft_reply_strips_placeholder_signoff_from_llm_output(tmp_path):
    """Integration test for the deterministic guardrail: even if the LLM
    ignores the prompt instruction and emits a placeholder, draft_reply
    must strip it before saving/returning."""
    fake_mail = MagicMock()
    fake_mail.subject = "About your faction"
    fake_mail.from_ = "friend@example.com"
    fake_mail.text = "Welcome to the Dauntless faction!"
    fake_mail.html = ""

    fake_mailbox = MagicMock()
    fake_mailbox.__enter__.return_value = fake_mailbox
    fake_mailbox.__exit__.return_value = False
    fake_mailbox.fetch.return_value = iter([fake_mail])

    # Exact real-world output observed live, despite the prompt forbidding it
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(
        content="Thank you for the welcoming message. I am excited to join the Dauntless faction.\n\nBest,   \n[User's Name]"
    )

    fake_store = main.drafts.DraftStore(db_path=str(tmp_path / "test_drafts_placeholder.sqlite"))

    with patch("main.connect", return_value=fake_mailbox), \
         patch("main.raw_llm", fake_llm), \
         patch("main.get_draft_store", return_value=fake_store):
        result = main.draft_reply.invoke({"uid": "6"})

    parsed = json.loads(result)
    assert "[User's Name]" not in parsed["draft_text"]
    assert "Dauntless faction" in parsed["draft_text"]  # real content preserved

    fake_store.close()


def test_draft_reply_saves_and_returns_draft(tmp_path):
    fake_mail = MagicMock()
    fake_mail.subject = "Can we reschedule?"
    fake_mail.from_ = "colleague@example.com"
    fake_mail.text = "Hi, can we move our 2pm meeting to 4pm today?"
    fake_mail.html = ""

    fake_mailbox = MagicMock()
    fake_mailbox.__enter__.return_value = fake_mailbox
    fake_mailbox.__exit__.return_value = False
    fake_mailbox.fetch.return_value = iter([fake_mail])

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="Sure, 4pm works for me. See you then.")

    fake_store = main.drafts.DraftStore(db_path=str(tmp_path / "test_drafts.sqlite"))

    with patch("main.connect", return_value=fake_mailbox), \
         patch("main.raw_llm", fake_llm), \
         patch("main.get_draft_store", return_value=fake_store):
        result = main.draft_reply.invoke({"uid": "1"})

    parsed = json.loads(result)
    assert parsed["email_uid"] == "1"
    assert parsed["status"] == "pending_review"
    assert "4pm" in parsed["draft_text"]
    assert "not been sent" in parsed["note"].lower()

    # confirm it was actually persisted, not just returned
    saved = fake_store.get(parsed["draft_id"])
    assert saved is not None
    assert saved["draft_text"] == parsed["draft_text"]

    fake_store.close()


def test_draft_reply_handles_uid_not_found():
    fake_mailbox = MagicMock()
    fake_mailbox.__enter__.return_value = fake_mailbox
    fake_mailbox.__exit__.return_value = False
    fake_mailbox.fetch.return_value = iter([])

    with patch("main.connect", return_value=fake_mailbox):
        result = main.draft_reply.invoke({"uid": "999"})

    assert result.startswith("Error:")
    assert "no email found" in result.lower()


def test_draft_reply_handles_malformed_uid():
    fake_mailbox = MagicMock()
    fake_mailbox.__enter__.return_value = fake_mailbox
    fake_mailbox.__exit__.return_value = False

    with patch("main.connect", return_value=fake_mailbox):
        result = main.draft_reply.invoke({"uid": "not-numeric"})

    assert result.startswith("Error:")
    assert "not a valid" in result.lower()


def test_draft_reply_handles_llm_failure_gracefully():
    fake_mail = MagicMock()
    fake_mail.subject = "Subject"
    fake_mail.from_ = "sender@example.com"
    fake_mail.text = "body"
    fake_mail.html = ""

    fake_mailbox = MagicMock()
    fake_mailbox.__enter__.return_value = fake_mailbox
    fake_mailbox.__exit__.return_value = False
    fake_mailbox.fetch.return_value = iter([fake_mail])

    broken_llm = MagicMock()
    broken_llm.invoke.side_effect = Exception("connection refused")

    with patch("main.connect", return_value=fake_mailbox), \
         patch("main.raw_llm", broken_llm):
        result = main.draft_reply.invoke({"uid": "1"})

    assert result.startswith("Error:")
    assert "ollama" in result.lower()


def test_draft_reply_does_not_persist_on_llm_failure(tmp_path):
    """A failed LLM call shouldn't leave a garbage draft in storage."""
    fake_mail = MagicMock()
    fake_mail.subject = "Subject"
    fake_mail.from_ = "sender@example.com"
    fake_mail.text = "body"
    fake_mail.html = ""

    fake_mailbox = MagicMock()
    fake_mailbox.__enter__.return_value = fake_mailbox
    fake_mailbox.__exit__.return_value = False
    fake_mailbox.fetch.return_value = iter([fake_mail])

    broken_llm = MagicMock()
    broken_llm.invoke.side_effect = Exception("connection refused")

    fake_store = main.drafts.DraftStore(db_path=str(tmp_path / "test_drafts2.sqlite"))

    with patch("main.connect", return_value=fake_mailbox), \
         patch("main.raw_llm", broken_llm), \
         patch("main.get_draft_store", return_value=fake_store):
        main.draft_reply.invoke({"uid": "1"})

    assert fake_store.list_all() == []
    fake_store.close()


def test_list_drafts_returns_only_pending(tmp_path):
    fake_store = main.drafts.DraftStore(db_path=str(tmp_path / "test_drafts3.sqlite"))
    id1 = fake_store.save("uid1", "Subject 1", "a@example.com", "Draft 1")
    fake_store.save("uid2", "Subject 2", "b@example.com", "Draft 2")
    fake_store.update_status(id1, "sent")

    with patch("main.get_draft_store", return_value=fake_store):
        result = main.list_drafts.invoke({})

    parsed = json.loads(result)
    assert len(parsed) == 1
    assert parsed[0]["email_uid"] == "uid2"

    fake_store.close()


def test_list_drafts_empty_returns_empty_list(tmp_path):
    fake_store = main.drafts.DraftStore(db_path=str(tmp_path / "test_drafts4.sqlite"))

    with patch("main.get_draft_store", return_value=fake_store):
        result = main.list_drafts.invoke({})

    assert json.loads(result) == []
    fake_store.close()


# ---------- graph routing ----------

def test_router_routes_to_tools_when_tool_calls_present():
    fake_message = MagicMock()
    fake_message.tool_calls = [{"name": "list_unread_emails", "args": {}, "id": "1"}]
    state = {"messages": [fake_message]}
    assert main.router(state) == "tools"


def test_router_routes_to_end_when_no_tool_calls():
    fake_message = MagicMock()
    fake_message.tool_calls = None
    state = {"messages": [fake_message]}
    assert main.router(state) == "end"


def test_llm_node_prepends_capability_boundary_system_prompt():
    """Regression test: the model implied it could send email twice in a
    row across two separate live tests (first "would you like me to send
    this?", then after a prompt fix, "before sending it" + offering to
    save an already-saved draft). llm_node must prepend a system message
    that explicitly forbids both patterns on every call."""
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="ok")

    with patch("main.llm", fake_llm):
        main.llm_node({"messages": [{"role": "user", "content": "hi"}]})

    call_args = fake_llm.invoke.call_args[0][0]
    system_msg = call_args[0]
    assert isinstance(system_msg, SystemMessage)
    prompt_lower = system_msg.content.lower()
    assert "cannot send" in prompt_lower
    assert "already been saved" in prompt_lower or "already saved" in prompt_lower
    assert "should i send" in prompt_lower or "before sending" in prompt_lower


def test_graph_compiles_with_no_checkpointer():
    graph = main.build_graph()
    assert graph is not None


def test_graph_compiles_and_runs_with_checkpointer_mocked_llm():
    """End-to-end graph run with a mocked LLM (no tool call -> should
    go straight to END) and an in-memory checkpointer, verifying state
    is retrievable by thread_id afterward."""
    checkpointer = InMemorySaver()
    graph = main.build_graph(checkpointer=checkpointer)

    fake_response = AIMessage(content="Hello, how can I help with your inbox?")

    config = {"configurable": {"thread_id": "test-thread"}}

    mocked_llm = MagicMock()
    mocked_llm.invoke.return_value = fake_response

    with patch("main.llm", mocked_llm):
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "hi"}]},
            config=config,
        )

    assert result["messages"][-1].content == "Hello, how can I help with your inbox?"

    # Verify checkpointer actually persisted this thread's state
    snapshot = graph.get_state(config)
    assert snapshot is not None
    assert len(snapshot.values["messages"]) >= 2  # user + AI message
