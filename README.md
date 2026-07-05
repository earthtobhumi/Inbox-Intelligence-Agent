# Email Agent — v2 hardening pass

Changes made to `main.py` from the original draft:

1. **Model swapped `qwen2.5:3b` → `qwen2.5:7b`.** 3b was unreliable for
   judgment-heavy tasks (priority scoring, drafting) and risked malformed
   tool-call output. 7b at Q4 quantization fits comfortably in a 6GB
   laptop GPU.
2. **Tools never raise.** `connect()`, `list_unread_emails`, and
   `summarize_email` now catch IMAP login failures, connection errors,
   and — this one was caught by the test suite, not anticipated up front —
   malformed UIDs (`imap_tools` raises a raw `TypeError` on a non-numeric
   UID, which a small model can plausibly hallucinate after misreading its
   own prior tool output). All failure paths return a string the LLM can
   react to, instead of crashing the tool node.
3. **Email bodies are capped at `MAX_EMAIL_CHARS` (4000 chars)** with a
   crude HTML-tag strip fallback when `.text` is empty. Prevents long
   threads / HTML marketing mail from blowing past the model's usable
   context.
4. **Added a `SqliteSaver` checkpointer**, gated behind `build_graph()`
   so tests can compile the graph with no checkpointer or an in-memory
   one. This is required before the FastAPI layer, since each HTTP
   request will otherwise start from a blank state.
5. **Added priority scoring (`priority.py` + `score_priority_emails` tool).**
   Ranks unread emails 0-100 using four signals: urgency keywords in the
   subject, directness (To: you vs Cc'd vs neither), sender familiarity,
   and recency. Deterministic and LLM-free — scoring doesn't need the
   model at all, only the drafting/summarizing steps do.

   **Known simplification, worth being upfront about:** with raw IMAP
   (not the Gmail API), there's no built-in sender-importance signal to
   query. "Sender familiarity" here is a locally-learned proxy — a count
   in a sqlite table (`sender_stats.sqlite`) that increments every time
   an email from that sender gets scored. It starts cold on a fresh
   install (every sender scores 0 on familiarity) and gets more accurate
   the more you use it. This is a defensible design tradeoff for a raw-IMAP
   setup, but say so plainly if asked in an interview — it is not real
   send/reply history the way Gmail API's People API would give you.

6. **Fixed a real ranking bug found during the first live run against a
   real inbox.** Google's automated security-alert sender was outranking
   an actual personal email, purely because that sender had emailed more
   times historically — familiarity was rewarding bot frequency, the
   opposite of what it's meant to signal. Added `is_automated_sender()`,
   a pattern match on common no-reply/notifications/security-alert
   address conventions, and gated familiarity to 0 for any sender it
   flags. Covered by a regression test (`test_rank_emails_personal_email_outranks_frequent_automated_sender`)
   that reproduces the exact scenario. This heuristic is best-effort, not
   exhaustive — an automated sender with an unusual address won't be caught.

7. **CLI now prints tool calls and raw tool results, not just the final
   answer.** Switched from `graph.invoke()` to `graph.stream(..., stream_mode="updates")`
   in the `__main__` loop. This was prompted by a real observed issue: the
   model's final prose echoed the wrong identifier back (a sender's email
   address instead of the actual UID it used), even though the underlying
   tool call had executed correctly. Printing each tool call and its raw
   result lets you audit what actually happened at each step instead of
   trusting the model's self-report of what it did.
8. **Added draft reply generation (`drafts.py` + `draft_reply`/`list_drafts`
   tools).** `draft_reply(uid)` fetches the real email, builds a grounded
   prompt instructing the model not to invent facts not present in the
   original, generates a draft, and saves it to a `drafts.sqlite` table
   with status `pending_review`. **Nothing is ever sent.** `list_drafts`
   returns everything still pending review. Each call to `draft_reply`
   creates a new `draft_id` rather than overwriting a previous attempt,
   so asking for a redraft doesn't destroy the first one.

   The prompt explicitly tells the model to say so instead of forcing a
   reply when the original email doesn't need one (e.g. a notification) --
   worth checking in your live test whether it actually respects that,
   since it's an instruction, not an enforced constraint.
9. **Fixed a real overclaiming bug found during the first live draft test —
   took two iterations to actually close.** After drafting a reply, the
   model first asked "would you like me to send this?", implying a send
   capability that doesn't exist anywhere in this codebase. Added a
   `SYSTEM_PROMPT` forbidding that. **Re-tested live and the model still
   said "before sending it" and offered to "save" a draft that was
   already saved** — same underlying problem, just softer wording. The
   prompt was rewritten to be more directive: explicit forbidden phrases,
   an explicit statement that saving already happened, and an instruction
   to redirect the user to their own email client for the send step.
   Prepended fresh on every `llm_node` call, not persisted into
   checkpointed state. **Important caveat:** a system prompt is a strong
   nudge, not an enforced guarantee, especially with a 7B model. The unit
   test here only confirms the *prompt itself* contains the right
   instructions -- it cannot confirm the model actually obeys them each
   time, since that requires a live call. Re-check this live again before
   trusting it; if it still slips, the fix would need to move from prompt
   wording to post-processing the model's output (e.g. rejecting/rewriting
   any response containing "send" in connection with the assistant).

   **Confirmed live: this fix worked.** Third live test showed the model
   correctly stating the draft "has been saved" (not offering to save it)
   and closing with "you can review this draft in your email client and
   send it as needed" -- correctly attributing the send action to the
   user, no overclaiming.
10. **Escalated the placeholder sign-off fix from prompt wording to a
    deterministic code guardrail, after the prompt-only fix failed a
    third time live.** `[User's Name]` appeared again, verbatim, in the
    raw `draft_text` from the LLM itself -- despite the prompt explicitly
    forbidding it. This confirms a 7B model will not reliably follow a
    "don't do X" instruction 100% of the time, so this could not stay
    prompt-only. Added `drafts.strip_placeholder_signoff()`, a regex-based
    post-processing step applied to every LLM draft output before it's
    saved or returned, regardless of what the model produced. Tested
    against the exact real-world string that kept recurring, plus common
    variants (`[Your Name]`, `[Name]`, `(Your Name)`, case-insensitive),
    and confirmed it does NOT strip real names like "John Smith" --
    only bracketed/parenthesized placeholder patterns.

## What was actually tested, and how

I do not have Ollama or a real IMAP mailbox available in the environment
this was built in — no local model runtime, and network access is
restricted to package registries. So there was no live end-to-end run
against your real inbox or your `qwen2.5:7b` model. Be aware of that
limit before you assume this is demo-ready.

What **was** verified, for real, against the actual installed libraries
(`langgraph`, `langchain`, `langchain-ollama`, `imap-tools` — real
versions, not assumed ones):

- `python -m py_compile main.py priority.py` — no syntax errors.
- `import main` — no import-time errors; `init_chat_model(..., provider="ollama")`
  does not eagerly connect, so this succeeds without Ollama running.
- **95 unit tests**, all passing (`test_main.py` + `test_priority.py` +
  `test_drafts.py`), using mocked IMAP mailboxes, real sqlite temp files,
  and mocked/real LangChain message objects:
  - body truncation and HTML-fallback logic
  - `list_unread_emails` success, empty-inbox, login-failure, and
    connection-error paths
  - `summarize_email` success-path wiring, UID-not-found, **malformed
    UID (regression test for a bug the test suite itself caught)**,
    login failure, and LLM invocation failure
  - **priority scoring**: each signal function in isolation (urgency
    keywords, directness incl. display-name parsing, recency incl.
    naive-vs-aware datetime handling, familiarity capping), automated-
    sender detection, **familiarity suppression for automated senders
    (regression test reproducing the real ranking bug found live)**,
    weight sum sanity check, `SenderStats` persistence across reopen,
    and `score_priority_emails` end-to-end with mocked IMAP
  - **draft generation**: prompt construction includes original content
    and grounding rules, `DraftStore` CRUD + status filtering + persistence
    across reopen, multiple drafts coexisting for the same email uid,
    `draft_reply` end-to-end (saves + returns), UID-not-found, malformed
    UID, LLM failure, and confirming a failed LLM call does NOT leave a
    garbage draft persisted, `list_drafts` filtering to pending-only
  - graph router (`tools` vs `end` edges)
  - graph compiles with no checkpointer and with an in-memory one
  - full graph run with a mocked LLM response, checkpointed by
    `thread_id`, and state retrievable afterward via `get_state()`
  - `SqliteSaver` — table setup (`saver.setup()`) and graph compilation
    against a real sqlite file, confirmed end to end
  - `graph.stream(..., stream_mode="updates")` output shape confirmed
    with a mocked multi-turn LLM (tool-call request → tool result →
    final answer), verifying the CLI's tool-call/tool-result printing
    logic correctly distinguishes an in-progress tool call from a
    final response with an empty `tool_calls` list

## Confirmed against a real inbox (not just mocks)

This has now actually been run against a real Gmail inbox with
`qwen2.5:7b` on Ollama. Multiple real issues surfaced through live
testing and were fixed as a result — see items 6, 7, and 9 above.
Confirmed working live:

- Tool-call/tool-result printing shows real IMAP data flowing through
  correctly (`list_unread_emails`, `score_priority_emails`).
- The Google-alert-vs-personal-email ranking fix works as intended:
  personal email scored 21.0 and ranked first; automated senders capped
  at 20.0 with the suppression reason shown.
- `draft_reply` produced a draft that stayed grounded in the real email
  content (referenced the actual faction/welcome context, no invented
  names or facts) and was correctly persisted with `status: pending_review`
  and an explicit "NOT been sent" note in the raw tool result.

**Confirmed live: the "don't imply sending" fix held up on the third
test** -- the model correctly said the draft "has been saved" and closed
with "you can review this draft ... before sending it," correctly
attributing the send action to the user rather than itself.

**Not yet confirmed live:** the placeholder guardrail is now code-level
(item 10) rather than prompt-only, and unit-tested against the exact
recurring string, but hasn't been re-run against a live Ollama call yet.
Draft another email and confirm no bracketed placeholder appears in the
final saved draft. This one should be a hard guarantee now regardless of
what the model outputs, unlike the earlier prompt-only attempts.

What was **not** tested and needs a live check on your machine:

- Actual draft/summary quality from `qwen2.5:7b` — read a few real
  outputs before trusting them in a demo.
- Actual tool-call reliability from `qwen2.5:7b` in the ReAct loop —
  watch for malformed tool calls on the first few live runs, especially
  now that there are 3 tools instead of 2 for the model to choose between.
- Whether the priority scoring *rubric* actually feels right against your
  real inbox — the weights (urgency 0.35, directness 0.20, familiarity
  0.20, recency 0.25) are a reasonable starting point, not a tuned result.
  Expect to adjust after seeing real output.
- Real IMAP login against Gmail (app password flow, SSL, folder names).
- Ollama actually running on GPU vs CPU (`ollama ps` after a query).

## Setup

```bash
pip install -r requirements.txt
ollama pull qwen2.5:7b
cp .env.example .env   # then fill in real IMAP credentials + USER_EMAIL
python main.py
```

## Running tests

```bash
pytest test_main.py test_priority.py test_drafts.py -v
```

## Known gap to close next

`connect()` opens a new IMAP connection per tool call. Fine for a demo,
but if you're calling `list_unread_emails` then `summarize_email` several
times in one turn, that's several serial logins. Worth revisiting once
the draft-generation node is added, since a full turn could now involve
scoring + summarizing + drafting in sequence.
