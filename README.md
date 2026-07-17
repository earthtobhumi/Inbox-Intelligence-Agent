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
11. **Added a FastAPI layer (`api.py`) with dedicated REST endpoints**,
    deliberately NOT a single `/chat` passthrough:
    - `GET /inbox/priority` -- ranked unread emails as JSON
    - `GET /drafts` (optional `?status=` filter, default `pending_review`,
      `status=all` for everything) -- list saved drafts
    - `GET /drafts/{draft_id}` -- fetch one draft
    - `PATCH /drafts/{draft_id}` -- update status (`approved`/`discarded`/
      `sent`/`pending_review`); this does NOT send email, it only updates
      the local record, e.g. after you've sent the reply yourself
    - `POST /drafts/generate` -- generate a new draft for a UID
    - `GET /health`

    **Design decision, worth explaining if asked:** `/inbox/priority` and
    the `GET /drafts` routes do not go through the LangGraph agent or an
    LLM call at all. They call the same deterministic Python
    (`priority.rank_emails`, `DraftStore`) the LangGraph tools already
    use, directly. Routing a pure data-fetch through an LLM tool-router
    would add latency and a new failure point (malformed tool calls,
    model unavailability) for zero benefit -- the LLM is reserved for
    where generation actually happens (`POST /drafts/generate`).

    **Implementation note:** `api.py` intentionally does NOT reuse the
    tool functions in `main.py` (`draft_reply`, `score_priority_emails`)
    directly -- those are tested by 95 existing tests with an exact,
    relied-upon error-string contract (`"Error: ..."`), and refactoring
    them to also serve HTTP responses risked destabilizing that suite for
    marginal benefit. Instead `api.py` reuses the shared *primitives*
    (`connect()`, `priority.rank_emails`, `DraftStore`, `raw_llm`,
    `strip_placeholder_signoff`) and re-implements the thin
    request/response glue itself, translating exceptions to proper HTTP
    status codes (401 login failure, 503 connection failure, 404 not
    found, 422 invalid input, 502 LLM failure) instead of error strings.
    This does mean the draft-generation logic exists in two places
    (`main.draft_reply` and `api.generate_draft`) -- a real, acknowledged
    duplication tradeoff, not an oversight.

    Verified with 18 new tests using FastAPI's `TestClient` (mocked IMAP/
    LLM, same pattern as the rest of the suite) AND a real `uvicorn`
    server boot hitting actual HTTP endpoints (`/health`, `/openapi.json`)
    over a real socket, not just the in-process test client.
12. **Added JWT auth (`auth.py`) with genuine multi-user isolation, not
    just a login screen in front of one shared mailbox.** Each account
    has its own IMAP credentials (encrypted at rest with Fernet, never
    stored in plaintext) and its own `SenderStats`/`DraftStore` sqlite
    files under `./data/`, keyed by username. Two users hitting the same
    endpoints get completely independent inboxes, priority scores, and
    draft histories -- verified directly by two regression tests
    (`test_two_users_see_independent_drafts`,
    `test_two_users_have_independent_sender_familiarity`) rather than
    just asserting the token check works.

    - `POST /auth/register` -- creates an account, returns a JWT
    - `POST /auth/login` -- authenticates, returns a JWT
    - Every other route now requires `Authorization: Bearer <token>`

    **Real bug caught by the test suite during this build:**
    `PasswordTooLongError` is a subclass of `ValueError` (bcrypt has a
    hard 72-byte input limit), so an `except ValueError` clause placed
    before `except PasswordTooLongError` silently swallowed it, always
    returning 409 instead of 422 for an overly long password. Fixed by
    reordering to catch the more specific exception first -- a good
    example of why exception-order bugs need a test that actually
    triggers the specific case, not just the general one.

    **Known simplifications, worth being upfront about:**
    - Registration does NOT verify IMAP credentials against a live
      server. A wrong password only surfaces later as a 401 from a route
      that actually connects. Better UX would check live at registration;
      this trades that for a simpler, faster account-creation step.
    - `JWT_SECRET_KEY` falls back to an insecure hardcoded default if
      unset, to keep local dev friction low -- this is NOT safe for
      anything beyond local testing, and the code says so via a comment,
      but there's no runtime check forcing you to set a real one.
    - Password hashing uses `bcrypt` directly rather than `passlib`,
      after `passlib`'s bcrypt backend threw an `AttributeError` due to a
      known version incompatibility with modern `bcrypt` releases
      (`passlib` expects `bcrypt.__about__.__version__`, which newer
      `bcrypt` removed). Worth knowing if you ever see this exact error
      elsewhere -- it's a library compatibility issue, not something
      wrong with your setup.

    Verified with 28 new tests (`test_api.py`, rewritten to go through a
    real register→login→authenticated-request flow rather than mocking
    auth away) and 19 new tests for `auth.py` in isolation (password
    hashing, JWT roundtrip/expiry/tampering, Fernet encryption roundtrip,
    `UserStore` CRUD). Also confirmed over a real `uvicorn` server boot:
    registration returns a real signed JWT, and an unauthenticated
    request to a protected route is correctly rejected with 401.
13. **Added scheduled digest via Celery + Redis + Celery Beat**
    (`celery_app.py`, `tasks.py`, `digest.py`), plus `GET`/`PATCH /digest/config`.

    **Design decisions:**
    - Digest content is generated **deterministically** from already-scored
      email data (reuses `priority.score_email`), NOT via an LLM call --
      consistent with the rest of this project's philosophy. A scheduled
      background job is exactly where you don't want a new LLM-shaped
      failure point (model down, malformed output, slow response holding
      up a worker).
    - This is the first feature that actually **sends** email. Deliberately
      different from `draft_reply`, which never sends: a digest is the
      assistant notifying the user about their own inbox, not replying to
      someone on the user's behalf. That distinction is why it needed its
      own explicit sign-off rather than reusing draft infrastructure.
    - Digest scoring does NOT record sender interactions in `SenderStats` --
      viewing a digest summary of an email shouldn't count as the user
      having engaged with that sender the way opening their live inbox does.
    - SMTP host is best-effort derived from the IMAP host (`imap.X` ->
      `smtp.X`, works for Gmail/Outlook/Yahoo) unless given explicitly at
      registration. Each user's own IMAP app password is reused for SMTP
      auth (true for Gmail; most providers use the same credential for both).
    - `check_and_send_due_digests` runs hourly via Celery Beat and checks
      every digest-enabled user's individual `frequency` +
      `last_digest_sent_at` via `digest.is_digest_due()` -- the hourly beat
      cadence is just the check frequency, not the send frequency. Each
      user's actual digest is dispatched as its own separate task, so one
      user's IMAP/SMTP failure can't block or crash another's.
    - `send_digest_for_user` catches and logs failures rather than letting
      them crash the worker; SMTP failures specifically get retried
      (transient network issues), IMAP login/connection failures and
      missing users are logged and given up on for that run.

    **Real bug found via genuine end-to-end testing during this build:**
    `auth.py` read `FERNET_KEY`/`JWT_SECRET_KEY` from `os.environ` at
    import time but never called `load_dotenv()` itself -- it was
    silently relying on `main.py` (imported elsewhere in the chain) to
    have already loaded `.env` first. This worked by accident whenever
    `main.py` happened to import before `auth.py`, but `tasks.py` imports
    `auth` *before* `main`, which would have made encryption/JWT signing
    silently fall back to insecure defaults in a real Celery worker
    process. Fixed by making `auth.py` call `load_dotenv()` itself,
    removing the import-order dependency entirely. This was only caught
    because verification here used two genuinely separate OS processes
    (a real worker + a real dispatch script) instead of testing everything
    in one process where the bug couldn't have surfaced.

    **What was verified for real, and what wasn't -- read this carefully,
    this feature has a real testing gap:**
    - A real Redis server was installed and run in the build environment,
      confirmed reachable (`redis-cli ping` -> `PONG`).
    - A real Celery worker was started against that real Redis, connected
      successfully, and correctly registered both tasks.
    - `celery beat` was started for real and confirmed it loads the
      schedule config and connects to the broker without error.
    - A task was dispatched through the **real broker** and received a
      real broker-issued task ID, confirming publish-to-queue works.
    - Redis queue **durability** was incidentally confirmed: a task
      dispatched to an earlier (killed) worker was still in the queue and
      got picked up correctly by a newly started worker.
    - Full task **business logic** (`send_digest_for_user`) was verified
      end-to-end through Celery's real task machinery (registration,
      `bind=True`, retry decorator) using `task_always_eager=True` with
      mocked IMAP/SMTP in the same process -- confirmed it fetches,
      scores, composes, sends (mocked), and calls `mark_digest_sent`
      correctly, and confirmed the mocked SMTP call received the right
      recipient/subject.

    **What was NOT successfully verified: a fully separate worker
    process picking up a task from a fully separate dispatching process
    and completing it asynchronously end-to-end.** Two attempts were made;
    both failed for environment reasons, not code reasons the first time
    (the worker process was a separate OS process that hadn't been
    started with the same mocks in place, so it tried a real network call
    and hung) -- and the second, after fixing that, failed because the
    background worker process didn't survive between separate tool
    invocations in the sandboxed build environment. This is a genuine,
    acknowledged gap: **the real async worker-picks-up-task-from-separate-process
    path has not been confirmed working, only inferred from its
    individual pieces all working correctly in isolation.** Run this
    yourself before trusting it:
    ```bash
    # terminal 1
    redis-server
    # terminal 2
    celery -A celery_app worker --loglevel=info
    # terminal 3
    celery -A celery_app beat --loglevel=info
    # terminal 4 -- register a real user via the API, then:
    python3 -c "from tasks import send_digest_for_user; send_digest_for_user.delay('yourusername')"
    ```
    Watch terminal 2 for the task being received and its result.

    Covered by 14 new tests (`test_digest.py`) for pure due-date logic,
    content formatting, and mocked SMTP sending, plus 11 new tests for
    the digest-config fields/methods added to `auth.py`'s `UserStore`,
    plus 9 new tests for the `/digest/config` API endpoints (including a
    per-user isolation test confirming Alice changing her frequency
    doesn't affect Bob's).

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
- **175 unit tests**, all passing (`test_main.py` + `test_priority.py` +
  `test_drafts.py` + `test_api.py` + `test_auth.py` + `test_digest.py`), using mocked IMAP mailboxes, real sqlite temp files,
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

**Also not yet confirmed live:** the FastAPI layer (item 11). Confirmed
working via mocked tests and a real `uvicorn` boot with mocked IMAP/LLM
underneath, but not yet hit against your real Gmail inbox or real
Ollama instance. Run `uvicorn api:app --reload` and try `GET /inbox/priority`,
`GET /drafts`, and `POST /drafts/generate` for real before trusting them.

**Also not yet confirmed live: JWT auth (item 12).** Register/login
confirmed working over a real HTTP server (real signed JWT returned,
unauthenticated requests correctly rejected with 401), but the
IMAP-dependent routes haven't been exercised end-to-end with a real
account's real credentials yet -- only with mocked IMAP underneath.
Register a real account via `/auth/register` with your actual IMAP
credentials and confirm `/inbox/priority` and `/drafts/generate` work
through the full authenticated path against your live inbox.

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
sudo apt install redis-server   # or point REDIS_URL at an existing instance
redis-server --daemonize yes
cp .env.example .env
# Fill in real IMAP credentials + USER_EMAIL for the CLI agent (main.py).
# Also generate and set FERNET_KEY and JWT_SECRET_KEY for the API (api.py) --
# see .env.example for the exact commands to generate them.
python main.py          # CLI chat agent (single account, from .env)
```

## Running the API

```bash
uvicorn api:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive API docs. The API is
multi-user and requires auth -- register an account first:

```bash
curl -X POST http://127.0.0.1:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "yourname",
    "password": "your-password",
    "user_email": "you@gmail.com",
    "imap_host": "imap.gmail.com",
    "imap_user": "you@gmail.com",
    "imap_pass": "your-gmail-app-password",
    "digest_frequency": "daily"
  }'
```

This returns an `access_token`. Use it on every other request:

```bash
TOKEN="paste-the-access_token-here"

curl http://127.0.0.1:8000/inbox/priority -H "Authorization: Bearer $TOKEN"
curl http://127.0.0.1:8000/drafts -H "Authorization: Bearer $TOKEN"
curl -X POST http://127.0.0.1:8000/drafts/generate \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"uid": "6"}'
curl http://127.0.0.1:8000/digest/config -H "Authorization: Bearer $TOKEN"
curl -X PATCH http://127.0.0.1:8000/digest/config \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"frequency": "weekly"}'
```

Each registered account gets its own IMAP credentials, priority-scoring
history, draft history, and digest schedule -- a second account
registered the same way is completely isolated from the first (verified
by tests, see items 12-13 in the changelog below).

## Running the scheduled digest (Celery + Redis + Beat)

Three separate processes, each in its own terminal:

```bash
redis-server                                    # if not already running
celery -A celery_app worker --loglevel=info     # executes digest tasks
celery -A celery_app beat --loglevel=info       # triggers the hourly check
```

Beat checks every digest-enabled user hourly and dispatches a digest for
anyone actually due, based on their individual `frequency` +
`last_digest_sent_at`. To trigger one manually without waiting:

```bash
python3 -c "from tasks import send_digest_for_user; send_digest_for_user.delay('yourusername')"
```

Watch the worker terminal for the task being received and its result.
**This exact flow has not been confirmed working end-to-end in the build
environment** -- see item 13 in the changelog for exactly what was and
wasn't verified, and why. Run it for real before trusting it.

## Running tests

```bash
pytest test_main.py test_priority.py test_drafts.py test_api.py test_auth.py test_digest.py -v
```

## Known gap to close next

`connect()` opens a new IMAP connection per tool call. Fine for a demo,
but if you're calling `list_unread_emails` then `summarize_email` several
times in one turn, that's several serial logins. Worth revisiting once
the draft-generation node is added, since a full turn could now involve
scoring + summarizing + drafting in sequence.

## Scoring backlog (non-blocking, tracked for later tuning)

**Automated senders can still score high via directness + recency alone,
even with familiarity suppressed.** Found live: a routine Google
"Security alert" scored 45 (directness 1.0 + recency 1.0), which can
approach or beat a real personal email that isn't equally recent.
Familiarity suppression (item 6) solved one of three ways a bot sender
could inflate its score, not all three. Possible fix: for senders where
`is_automated_sender()` is true, also suppress directness -- automated
mail is always "directly addressed to you" by nature, so that signal
carries no real information there. Only urgency language should be able
to push an automated email's score up (so a genuine fraud alert with
"immediate action required" still ranks appropriately, but a routine
"new sign-in" notification doesn't). Not fixed yet -- deliberately
deferred since it's a rubric refinement, not a correctness bug, and the
weights will likely need more than one more tuning pass as real usage
continues.
