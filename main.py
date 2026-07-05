import os
import json
import sqlite3
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from imap_tools import MailBox, AND
from imap_tools.errors import MailboxLoginError

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

import priority
import drafts

load_dotenv()

IMAP_HOST = os.getenv("IMAP_HOST")
IMAP_USER = os.getenv("IMAP_USER")
IMAP_PASS = os.getenv("IMAP_PASS")
INBOX = os.getenv("INBOX", "INBOX")
# Used for directness scoring (To: you vs Cc'd vs neither). Falls back to
# IMAP_USER since for most single-account setups they're the same address.
USER_EMAIL = os.getenv("USER_EMAIL", IMAP_USER)

# Model swapped from qwen2.5:3b -> qwen2.5:7b.
# 3b was too weak for reliable tool-call formatting + judgment-heavy tasks
# (priority scoring, drafting). 7b at Q4 fits comfortably in 6GB VRAM.
CHAT_MODEL = "qwen2.5:7b"

# Cap on raw email body characters sent to the LLM. Prevents blowing past
# the model's usable context on long threads / HTML-heavy marketing mail.
MAX_EMAIL_CHARS = 4000

# Explicit capability boundary. Added after a live run where the model,
# despite no send tool existing, closed a draft-reply response by asking
# "would you like me to send this?" -- implying a capability it doesn't
# have. That's worse than a visible error: it looks like it worked.
SYSTEM_PROMPT = (
    "You are an email assistant with exactly these tools: list_unread_emails, "
    "score_priority_emails, summarize_email, draft_reply, list_drafts. "
    "You CANNOT send, reply to, delete, or modify any email -- no such tool "
    "exists, and none is planned. draft_reply already saves the draft the "
    "moment the tool runs; there is no separate save step.\n\n"
    "When you report a draft_reply result back to the user:\n"
    "- State plainly that it has ALREADY been saved (it has, as soon as the "
    "tool returned) -- never offer to save it, that already happened.\n"
    "- Never use the words 'send', 'sending', or 'sent' in connection with "
    "anything you might do. You have no ability to send email in any form.\n"
    "- Never ask 'should I send this' or 'before sending it' or any "
    "variation implying you could carry out sending.\n"
    "- Instead, tell the user the draft is saved and they can review/edit/"
    "send it themselves from their own email client.\n"
    "- Never claim any action succeeded unless a tool result actually "
    "confirms it."
)

DB_PATH = "checkpoints.sqlite"


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def connect() -> MailBox:
    """Open an authenticated IMAP connection. Raises on failure — callers
    (tools) are responsible for catching and turning this into a string
    the LLM can react to, since tools must never raise."""
    if not all([IMAP_HOST, IMAP_USER, IMAP_PASS]):
        raise RuntimeError(
            "Missing IMAP credentials. Check IMAP_HOST, IMAP_USER, IMAP_PASS in .env"
        )
    mailbox = MailBox(IMAP_HOST)
    mailbox.login(IMAP_USER, IMAP_PASS, initial_folder=INBOX)
    return mailbox


def _clean_body(text: str, html: str) -> str:
    """Prefer plaintext body; fall back to a crude HTML strip. Always
    truncate — untruncated bodies are the most common way to silently
    blow a small local model's context window."""
    body = text or ""
    if not body and html:
        import re
        body = re.sub(r"<[^>]+>", " ", html)
        body = re.sub(r"\s+", " ", body).strip()
    if len(body) > MAX_EMAIL_CHARS:
        body = body[:MAX_EMAIL_CHARS] + "\n[...truncated...]"
    return body


raw_llm = init_chat_model(CHAT_MODEL, model_provider="ollama")

_sender_stats = None


def get_sender_stats() -> priority.SenderStats:
    """Lazy singleton so a fresh sqlite connection isn't opened on import
    (matters for tests, which patch this out entirely)."""
    global _sender_stats
    if _sender_stats is None:
        _sender_stats = priority.SenderStats()
    return _sender_stats


_draft_store = None


def get_draft_store() -> drafts.DraftStore:
    global _draft_store
    if _draft_store is None:
        _draft_store = drafts.DraftStore()
    return _draft_store


@tool
def list_unread_emails():
    """List all unread email messages from the user's inbox."""
    try:
        with connect() as mailbox:
            unread_emails = list(
                mailbox.fetch(criteria=AND(seen=False), headers_only=True, mark_seen=False)
            )
    except MailboxLoginError:
        return "Error: IMAP login failed. Check IMAP_USER/IMAP_PASS (use an app password, not your normal login password)."
    except (RuntimeError, ConnectionError, OSError) as e:
        return f"Error: could not connect to mail server ({e})."

    if not unread_emails:
        return json.dumps([])

    return json.dumps([
        {
            "uid": mail.uid,
            "date": mail.date.strftime("%Y-%m-%d %H:%M:%S"),
            "subject": mail.subject,
            "from": mail.from_,
        }
        for mail in unread_emails
    ])


@tool
def score_priority_emails():
    """Fetch unread emails and rank them by priority (urgency language,
    whether you were directly addressed, sender familiarity, recency).
    Returns a JSON list sorted highest-priority first, each with a 0-100
    score and short reasons."""
    try:
        with connect() as mailbox:
            unread_emails = list(
                mailbox.fetch(criteria=AND(seen=False), headers_only=True, mark_seen=False)
            )
    except MailboxLoginError:
        return "Error: IMAP login failed. Check IMAP_USER/IMAP_PASS (use an app password, not your normal login password)."
    except (RuntimeError, ConnectionError, OSError) as e:
        return f"Error: could not connect to mail server ({e})."

    if not unread_emails:
        return json.dumps([])

    email_dicts = [
        {
            "uid": mail.uid,
            "date": mail.date,
            "subject": mail.subject,
            "from": mail.from_,
            "to": list(mail.to or []),
            "cc": list(mail.cc or []),
        }
        for mail in unread_emails
    ]

    ranked = priority.rank_emails(email_dicts, USER_EMAIL, get_sender_stats())

    # dates aren't JSON-serializable as datetime objects; stringify for output
    for e in ranked:
        e["date"] = e["date"].strftime("%Y-%m-%d %H:%M:%S") if e["date"] else None

    return json.dumps(ranked)


@tool
def summarize_email(uid: str):
    """Summarize the content of a specific email using its unique UID identifier."""
    try:
        with connect() as mailbox:
            email_message = next(mailbox.fetch(AND(uid=[uid]), mark_seen=False), None)
    except MailboxLoginError:
        return "Error: IMAP login failed. Check IMAP_USER/IMAP_PASS."
    except (RuntimeError, ConnectionError, OSError) as e:
        return f"Error: could not connect to mail server ({e})."
    except TypeError:
        # imap_tools raises TypeError on a malformed UID (e.g. non-numeric).
        # A small local model can hallucinate a bad UID from prior tool
        # output -- this must come back as a string, not crash the tool node.
        return f"Error: '{uid}' is not a valid email UID."

    if email_message is None:
        return f"Error: no email found with UID {uid}."

    body = _clean_body(email_message.text, email_message.html)

    prompt = (
        "Summarize the following email content in a concise manner:\n\n"
        f"Subject: {email_message.subject}\n"
        f"From: {email_message.from_}\n"
        f"Date: {email_message.date}\n\n"
        f"Content: {body}"
    )

    try:
        return raw_llm.invoke(prompt).content
    except Exception as e:
        return f"Error: LLM call failed ({e}). Is Ollama running (`ollama serve`) with {CHAT_MODEL} pulled?"


@tool
def draft_reply(uid: str):
    """Draft a reply to a specific email using its UID. The draft is
    generated for review and saved locally -- it is NEVER sent
    automatically. Returns the draft text and a draft_id you can use to
    look it up later."""
    try:
        with connect() as mailbox:
            email_message = next(mailbox.fetch(AND(uid=[uid]), mark_seen=False), None)
    except MailboxLoginError:
        return "Error: IMAP login failed. Check IMAP_USER/IMAP_PASS."
    except (RuntimeError, ConnectionError, OSError) as e:
        return f"Error: could not connect to mail server ({e})."
    except TypeError:
        return f"Error: '{uid}' is not a valid email UID."

    if email_message is None:
        return f"Error: no email found with UID {uid}."

    body = _clean_body(email_message.text, email_message.html)
    prompt = drafts.build_draft_prompt(email_message.subject, email_message.from_, body)

    try:
        draft_text = raw_llm.invoke(prompt).content
    except Exception as e:
        return f"Error: LLM call failed ({e}). Is Ollama running (`ollama serve`) with {CHAT_MODEL} pulled?"

    # Deterministic cleanup -- the prompt instruction alone did not
    # reliably stop the model from leaving a bracketed name placeholder,
    # so this runs regardless of what the model produced.
    draft_text = drafts.strip_placeholder_signoff(draft_text)

    draft_id = get_draft_store().save(
        email_uid=uid, subject=email_message.subject, sender=email_message.from_, draft_text=draft_text
    )

    return json.dumps({
        "draft_id": draft_id,
        "email_uid": uid,
        "subject": email_message.subject,
        "draft_text": draft_text,
        "status": "pending_review",
        "note": "This draft has been saved for your review. It has NOT been sent.",
    })


@tool
def list_drafts():
    """List all saved drafts pending review (not yet sent or discarded)."""
    all_drafts = get_draft_store().list_all(status="pending_review")
    if not all_drafts:
        return json.dumps([])
    return json.dumps(all_drafts)


llm = init_chat_model(CHAT_MODEL, model_provider="ollama")
llm = llm.bind_tools([list_unread_emails, score_priority_emails, summarize_email, draft_reply, list_drafts])


def llm_node(state: ChatState):
    # Prepended fresh each call, not stored in state -- avoids duplicating
    # it every turn as the checkpointed history grows.
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [response]}


def router(state: ChatState):
    last_message = state["messages"][-1]
    return "tools" if getattr(last_message, "tool_calls", None) else "end"


tools_node = ToolNode([list_unread_emails, score_priority_emails, summarize_email, draft_reply, list_drafts])

builder = StateGraph(ChatState)
builder.add_node("llm", llm_node)
builder.add_node("tools", tools_node)
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", router, {"tools": "tools", "end": END})
builder.add_edge("tools", "llm")


def build_graph(checkpointer=None):
    """Factory so tests / the FastAPI layer can compile with or without
    a checkpointer (e.g. an in-memory saver for unit tests)."""
    return builder.compile(checkpointer=checkpointer)


def _make_sqlite_checkpointer():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()  # creates checkpoint tables on first run; no-op after
    return saver


if __name__ == "__main__":
    checkpointer = _make_sqlite_checkpointer()
    graph = build_graph(checkpointer=checkpointer)

    thread_config = {"configurable": {"thread_id": "cli-session-1"}}

    print(f"Using model: {CHAT_MODEL}")
    print("Type an instruction or 'quit' to exit.")

    while True:
        user_input = input("User: ")
        if user_input.lower() == "quit":
            break

        try:
            final_content = None
            for update in graph.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config=thread_config,
                stream_mode="updates",
            ):
                for node_name, node_output in update.items():
                    for msg in node_output.get("messages", []):
                        tool_calls = getattr(msg, "tool_calls", None)
                        if node_name == "llm" and tool_calls:
                            for tc in tool_calls:
                                print(f"  [tool call] {tc['name']}({tc.get('args', {})})")
                        elif node_name == "tools":
                            tool_name = getattr(msg, "name", "unknown_tool")
                            content = getattr(msg, "content", "")
                            preview = content if len(content) <= 300 else content[:300] + "...[truncated]"
                            print(f"  [tool result: {tool_name}] {preview}")
                        elif node_name == "llm" and not tool_calls:
                            final_content = getattr(msg, "content", None)
        except Exception as e:
            print(f"AI: [error running graph: {e}]")
            print("    Is Ollama running locally with the model pulled?")
            continue

        print("AI:", final_content if final_content else "(no response returned)")
