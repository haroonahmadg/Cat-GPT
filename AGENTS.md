# AGENTS.md — Cat GPT Project Brief

> **Code wins over comments.** This document was produced by reading the
> actual source files top-to-bottom. Every claim is grounded in code.
> The final section records every place the owner's brief differed from
> reality.

---

## 1. Project Purpose & Identity Rule

Cat GPT is a public portfolio demo of a ChatGPT-style AI assistant built on
Groq LLMs, LangGraph orchestration, FAISS-backed RAG, and a Next.js frontend.
The assistant **must always identify itself as "Cat GPT"** — this is
hard-coded in the system prompt built by `build_system_prompt()` in
`agent/graph.py`:

```
You are Cat GPT, a playful cat-themed AI assistant. Your name is Cat GPT.
Never identify yourself as ChatGPT, GPT, Claude, Gemini, or any other
assistant. If anyone asks your name, who you are, or who built you,
always answer that you are Cat GPT.
```

This rule cannot be overridden by user messages because the `SystemMessage`
is prepended on every call in `assistant_node()`.

---

## 2. Architecture Map — File to Responsibility

```
ai-assistant/                    <- monorepo root
|
+-- api/
|   +-- main.py                  FastAPI app (~1350 lines). All routes.
|   |                            Auth middleware; SSE streaming; file upload;
|   |                            voice; image library; thread CRUD; health.
|   +-- config.py                Env-driven settings (see section 5 for full list).
|   +-- auth.py                  Token creation/verification (HMAC-SHA256).
|   |                            Google Identity Services + email/password.
|   |                            argon2-cffi password hashing.
|   +-- store.py                 Per-user thread bookkeeping (SQLite ui.sqlite):
|   |                            pinned_threads, deleted_threads,
|   |                            renamed_threads, archived_threads, users.
|   +-- streaming.py             SSE helpers: format_sse(), extract helpers.
|   +-- rate_limit.py            In-memory sliding-window AttemptLimiter
|   |                            (LOGIN_EMAIL_LIMITER, LOGIN_IP_LIMITER,
|   |                             REGISTER_IP_LIMITER).
|   +-- schemas.py               Pydantic request/response models.
|
+-- agent/
|   +-- graph.py                 LangGraph StateGraph (~1868 lines).
|   |                            AgentState, build_system_prompt(),
|   |                            assistant_node (with model fallback loop),
|   |                            _explain_groq_error(), build_graph(),
|   |                            create_checkpointer() -> SqliteSaver,
|   |                            stream_agent(), get_all_saved_threads(),
|   |                            get_thread_history(), remove_last_turn(),
|   |                            parse_input_messages(), prepare_image_for_groq().
|   +-- tools.py                 All @tool functions (~2339 lines).
|   |                            emit_tool_event(), active_user_id ContextVar,
|   |                            active_source_filter ContextVar. See section 6.
|   +-- voice.py                 TTS (Groq playai-tts -> gTTS fallback) +
|   |                            STT (Groq Whisper whisper-large-v3-turbo).
|   +-- image.py                 Thin helper module.
|   +-- llm_recovery.py          recover_tool_call() -- safety-net for models
|   |                            that emit a tool call as text instead of
|   |                            structured format.
|   +-- quiz.py                  Quiz helper.
|
+-- database/
|   +-- db.py                    Long-term user memories + calendar events.
|                                SQLite at data/assistant.db.
|                                Tables: memories, events, chat_history.
|
+-- rag/
|   +-- ingest.py                FAISS + HuggingFace embeddings ingestion.
|   |                            Per-user isolation via user_id metadata in
|   |                            each chunk. progress_context ContextVar for
|   |                            progress callbacks.
|   +-- search.py                search_pdf() (similarity) +
|                                get_document_preview() (order-based fallback).
|                                Per-user filter applied in similarity search.
|                                FAISS global index at data/faiss/.
|
+-- utils/
|   +-- security.py              sanitize_path(), validate_file_security(),
|                                sanitize_user_input().
|
+-- forntend/                    <- INTENTIONAL MISSPELLING. NEVER rename.
|   +-- app/
|   |   +-- page.tsx             Main chat page (~42 KB). All UI state.
|   |   +-- layout.tsx           Root layout.
|   |   +-- threads/             Thread-specific route(s).
|   +-- components/
|   |   +-- Sidebar.tsx          Thread list, auth panel.
|   |   +-- ChatInput.tsx        Message composer, file/image attach, voice.
|   |   +-- CatLogo.tsx          SVG logo.
|   |   +-- CatMood.tsx          Animated cat mood indicator.
|   |   +-- HomeHero.tsx         Landing hero.
|   |   +-- QuickActions.tsx     Shortcut buttons.
|   +-- lib/
|   |   +-- api.ts               Backend client. Parses SSE frames.
|   |   +-- types.ts             DTO + frontend type definitions.
|   |   +-- useAuth.ts           Auth state hook (Google + email/password).
|   |   +-- useChat.ts           Chat state hook (~26 KB).
|   +-- tailwind.config.ts       Tailwind CSS (frontend uses Tailwind,
|   |                            NOT plain CSS -- see section 10 discrepancies).
|   +-- next.config.mjs          Next.js config.
|
+-- tests/                       pytest suite (no live network).
|   +-- test_agent.py
|   +-- test_auth.py
|   +-- test_calculator.py
|   +-- test_db.py
|   +-- test_export.py
|   +-- test_extra_tools.py
|   +-- test_image_library.py
|   +-- test_model_fallback.py
|   +-- test_rag.py
|   +-- test_rag_citations.py
|   +-- test_security.py
|
+-- app.py                       Original Streamlit frontend. Intentionally
|                                untouched. Runs against the same on-disk data
|                                as the FastAPI backend.
+-- Dockerfile                   python:3.14-slim. CMD uses sh -c so ${PORT}
|                                expands. gTTS installed twice with --no-deps
|                                (duplicate line, harmless but present).
+-- requirements.txt             Pinned deps including langchain, langgraph,
|                                groq, faiss-cpu, sentence-transformers, etc.
+-- render.yaml                  Render deployment config.
+-- .env.example                 Documented env vars.
```

---

## 3. Streaming Event Contract (actual shapes from code)

Events are SSE frames: `event: <type>\ndata: <json>\n\n`

Source: `api/main.py` `event_stream()` + `api/streaming.py` + `forntend/lib/api.ts` `parseSseFrame()`.

| Type | Data shape | When sent |
|------|-----------|-----------|
| `text` | `{"text": "..."}` | Each `updates` chunk that has assistant text |
| `image` | `{"url": "/generated-images/<file>"}` | Tool returned `[GENERATED_IMAGE]<filename>` |
| `activity` | `{"message": "..."}` | Each `custom` stream chunk with a `message` key (tool events via `emit_tool_event()`) |
| `citations` | `{"items": [{"file": str, "page": int or null, "preview": str}, ...]}` | After stream completes, if any RAG citations were accumulated |
| `done` | `{}` | Turn finished normally |
| `error` | `{"message": "..."}` | Any unhandled exception in `event_stream()` |

**Contract rules:**

- Clients **must ignore unknown event types** (`parseSseFrame()` returns `null` for unrecognised names — safe to add new types).
- Existing shapes are **stable** — do not change field names or remove fields.
- New event types **may be added** freely (clients already handle this gracefully).
- `activity` events originate from `emit_tool_event()` in `agent/tools.py` and `rag/ingest.py`. The custom chunk shape is `{"message": str, "event": str, "tool": str, ...metadata}`. Only the `message` field is surfaced.
- `citation` custom events are **intercepted before** becoming `activity` events; they accumulate into the final `citations` SSE frame.

---

## 4. Auth & Isolation Model

### Token format

```
<user_id>.<expires_at_unix>.<hmac_sha256_signature>
```

Stateless — no server-side session storage. Verified by `verify_session_token()` in `api/auth.py`.

### Auth methods

1. **Google Identity Services** — `POST /auth/google` — verifies Google ID token JWT, creates or links account.
2. **Email + password** — `POST /auth/register`, `POST /auth/login` — argon2-cffi hashing.
3. **Set/change password** — `POST /auth/set-password` (authenticated; must supply current password if one exists).

### Public paths (no token required)

`/health`, `/auth/google`, `/auth/register`, `/auth/login`, `/docs`, `/openapi.json`, `/redoc`

All other routes require `Authorization: Bearer <token>`.

### Rate limits (in-memory, per-process)

- Login: 5 attempts/email, 20 attempts/IP in 15-minute windows.
- Register: 10 attempts/IP in 1-hour window.
- Returns HTTP 429 with plain-language message.

### Per-user isolation

| Resource | How isolated |
|----------|-------------|
| Threads / checkpoints | `user_id` passed to `get_all_saved_threads()`; ownership verified before delete/export |
| Thread metadata (pins, renames, archives, deletes) | `(thread_id, user_id)` composite key in `api/store.py` tables |
| Uploads | `UPLOAD_DIR/<sha256(user_id)>/<thread_id>/` — ownership validated in `_is_user_upload_path()` |
| Generated images | `data/generated_images/<sha256(user_id)>/` — `get_generated_image_dir()` in `agent/tools.py` |
| RAG index | Global FAISS index with `user_id` metadata per chunk; `search_pdf()` filters by `user_id` |
| Long-term memories | `save_memory()` / `get_memories()` scoped by `user_id` in `database/db.py` |
| Email send log | `email_send_log` table keyed by `user_id` + date |

The `active_user_id` ContextVar in `agent/tools.py` carries the authenticated user's ID into tools. It is set by `stream_agent()` in `graph.py` before invoking the graph.

---

## 5. Configuration (api/config.py + agent/graph.py)

### Required env vars

| Name | Used by | Notes |
|------|---------|-------|
| `GROQ_API_KEY` | graph.py, voice.py | LLM + TTS + STT |
| `SESSION_SECRET` | auth.py | HMAC signing |
| `GOOGLE_CLIENT_ID` | auth.py | Google OAuth |
| `GROQ_MODELS` | graph.py | Comma-separated fallback list (up to 6). Falls back to `GROQ_MODEL`. |
| `GROQ_VISION_MODEL` | graph.py | Separate model for image/vision requests |

### Optional env vars (defaults in parentheses)

`GROQ_TTS_VOICE` (`Fritz-PlayAI`), `POLLINATIONS_IMAGE_MODEL` (`flux`),
`SESSION_TTL_SECONDS` (604800), `FRONTEND_ORIGIN` (`http://localhost:3000`),
`MAX_FILE_MB` (25), `MAX_IMAGE_MB` (10), `MAX_UPLOAD_FILES` (10),
`MAX_IMAGES_PER_REQUEST` (3), `MAX_PROMPT_CHARS` (8000),
`GROQ_MAX_TOKENS` (1024), `GROQ_VISION_MAX_TOKENS` (512),
`LLM_TEMPERATURE` (0.3), `RATE_LIMIT_MAX_RETRIES` (2),
`RATE_LIMIT_MAX_WAIT_SECONDS` (60),
`UI_DB_PATH` (`data/ui.sqlite`), `UPLOAD_DIR` (`data/upload`),
`CHECKPOINT_DB_PATH` (`data/checkpoints.sqlite`),
`DATA_DIR` (`<repo-root>/data`),
`HF_API_TOKEN` (empty — unused for image gen), `HF_IMAGE_MODEL`.

### Optional API-key tools

`CRICAPI_KEY`, `OMDB_API_KEY`, `RESEND_API_KEY`, `EMAIL_DAILY_LIMIT` (default: 5).

Tools return a plain-language "not configured" message when their key is absent — see section 6.

---

## 6. Tool Inventory

All tools live in `agent/tools.py` and are registered in the `TOOLS` list in `agent/graph.py`.

### Media-marker convention (generate_image, generate_qr_code)

When an image tool succeeds it returns:

```
[GENERATED_IMAGE]<hex16>.png
```

`api/streaming.py` `extract_generated_image_filename()` detects this prefix and
converts the `updates` chunk into a dedicated `image` SSE event with
`{"url": "/generated-images/<filename>"}`. The filename is NOT sent as text.

### Not-configured convention (tools requiring external API keys)

When an env var is missing the tool returns a plain English string and never raises:

| Tool | Missing key | Exact return string |
|------|------------|---------------------|
| `get_cricket_score` | `CRICAPI_KEY` | `"Cricket scores are not configured on this server."` |
| `get_movie_info` | `OMDB_API_KEY` | `"Movie information is not configured on this server."` |
| `send_email` | `RESEND_API_KEY` | `"Email sending is not configured on this server."` |

### Full tool list

| Tool | Description | External service | API key? |
|------|-------------|-----------------|----------|
| `calculator` | Safe AST-based math evaluator | None | No |
| `web_search` | Tavily web search | api.tavily.com | `TAVILY_API_KEY` (LangChain reads from env) |
| `rag_search` | FAISS semantic search over user's documents | Local | No |
| `create_event` | Add calendar event to SQLite | Local | No |
| `list_events` | List all calendar events | Local | No |
| `delete_event` | Delete a calendar event by ID | Local | No |
| `save_user_memory` | Store key/value memory per user | Local | No |
| `get_user_memories` | Retrieve user memories | Local | No |
| `generate_image` | Image via Pollinations Flux | image.pollinations.ai | No |
| `generate_qr_code` | QR code via api.qrserver.com | api.qrserver.com | No |
| `get_weather` | Current weather + 3-day forecast via open-meteo.com | open-meteo.com + geocoding | No |
| `get_prayer_times` | Prayer times via aladhan.com | api.aladhan.com | No |
| `convert_currency` | FX conversion via open.er-api.com | open.er-api.com | No |
| `get_crypto_price` | Crypto price via CoinGecko (BTC/ETH/SOL/etc.) | api.coingecko.com | No |
| `get_cricket_score` | Live cricket scores via CricAPI | api.cricapi.com | `CRICAPI_KEY` |
| `get_movie_info` | Movie info via OMDb | omdbapi.com | `OMDB_API_KEY` |
| `get_datetime` | Date/time + timezone conversion | IANA zoneinfo | No |
| `generate_password` | Cryptographically random password (8-64 chars) | None | No |
| `define_word` | English dictionary via dictionaryapi.dev | dictionaryapi.dev | No |
| `get_cat_fact` | Random cat fact via catfact.ninja | catfact.ninja | No |
| `send_email` | Email to user's verified address via Resend | api.resend.com | `RESEND_API_KEY` |

### emit_tool_event() signature

```python
emit_tool_event(message: str, *, event: str, tool: str, **metadata) -> None
```

Uses `langgraph.config.get_stream_writer()`. Safe to call outside LangGraph context (silently no-ops). Standard `event` values seen in the codebase: `tool_started`, `tool_completed`, `tool_error`, `citation`, `model_fallback`, `rate_limit_retry`, `tool_call_recovered`.

---

## 7. Commands

### Run backend locally

```bash
# From repo root (ai-assistant/)
uvicorn api.main:app --reload --port 8000
```

Requires `.env` with at minimum: `GROQ_API_KEY`, `SESSION_SECRET`, `GOOGLE_CLIENT_ID`, `GROQ_MODELS`, `GROQ_VISION_MODEL`.

### Run frontend locally

```bash
# From forntend/ directory
cd forntend
npm install      # first time only
npm run dev      # starts Next.js dev server on port 3000
```

Requires `forntend/.env.local` containing:

```
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

### Run pytest

```bash
# From repo root (ai-assistant/)
pytest tests/
```

No live network required — all external HTTP is mocked inside each test file.

### Legacy Streamlit frontend (unchanged)

```bash
streamlit run app.py
```

---

## 8. Conventions

### Naming

- Python: `snake_case` throughout. Pydantic models: `PascalCase`.
- TypeScript: `camelCase` for frontend-facing shapes; `snake_case` in DTOs that mirror `api/schemas.py`.
- Tool functions: `snake_case` matching the name the LLM sees.
- API route paths: hyphen-separated (e.g. `/auth/set-password`).

### Error style

- **Backend errors surfaced to the LLM** are plain English (see `_explain_groq_error()`, tool return strings).
- **HTTP errors** use FastAPI's `HTTPException(status_code=..., detail="Plain English message")`.
- **Raw stack traces never reach the client.** The SSE `error` event carries `{type(exc).__name__}: {exc}` — informative but not a raw traceback.
- **Tool failures return a string** (never raise), so LangGraph surfaces the reason to the model.

### Test style

- pytest with `TestClient` (FastAPI) or direct function imports.
- External HTTP always mocked (`unittest.mock.patch`, `monkeypatch`).
- Fixtures use `tempfile.mkdtemp` for ephemeral SQLite databases.
- No live Groq/Tavily/etc. calls.

### Commit style

No explicit convention found in the repo. The owner's brief requested
`"docs: add AGENTS.md project brief"` for this commit. Follow
conventional-commits style (`<type>: <description>`) for future work.

---

## 9. Hard Rules & Pitfalls

### `forntend/` spelling

The frontend directory is **intentionally misspelled**. Vercel Root Directory is set to `forntend`. **Never rename it.** Doing so breaks the production deployment.

### Streaming event contract is additive-only

- Do **not** change field names or remove fields from existing event types.
- **Do** add new event types freely — `parseSseFrame()` returns `null` for unknown names, so clients skip them silently.

### LangGraph owns orchestration

All multi-turn conversation state, tool calling, and checkpointing flows through LangGraph's `StateGraph`. **Never replace this with hand-rolled loops in the API layer.** `stream_agent()` in `graph.py` is the only entry point for chat.

### Checkpoint DB compatibility

`SqliteSaver` checkpoints are stored in `data/checkpoints.sqlite`. The schema is managed by LangGraph. Do not alter this file manually or change the checkpoint path without migrating existing data.

### Dockerfile constraints

- Base image: `python:3.14-slim`
- `gTTS` is installed with `--no-deps` due to a `click` version conflict with other dependencies.
- CMD uses `sh -c "uvicorn ... --port ${PORT:-8000}"` so the Render-injected `$PORT` variable expands at runtime. Using exec-form `["uvicorn", ...]` would break PORT expansion.
- The `gTTS` install line appears **twice** in the Dockerfile. This is harmless but redundant.

### app.py must not be imported

`app.py` (the Streamlit frontend) executes Streamlit-only calls at import time. Importing it from any non-Streamlit context will crash. Logic shared between both frontends is intentionally duplicated.

### Per-user isolation is mandatory

Every read/write of threads, memories, RAG results, uploads, and generated images must be scoped to the authenticated `user_id`. The `active_user_id` ContextVar carries this into tools; `request.state.user_id` carries it into API routes.

### FAISS index is global

All users share one FAISS index at `data/faiss/`. Per-user isolation is enforced at query time by filtering on the `user_id` metadata field in each chunk. **Never remove this filter.**

### Rate limiter is in-memory / single-process

`api/rate_limit.py` stores attempt counts in process memory. With multiple workers or servers, move the state to a shared store (database or Redis).

---

## 10. Discrepancies vs. Owner's Brief

The following points differ from the brief. **Code wins.**

| # | Brief claim | Actual code |
|---|-------------|------------|
| 1 | "Signed session tokens; ONLY auth and health routes are public." | Correct, but public paths also include `/docs`, `/openapi.json`, `/redoc` (FastAPI auto-routes). |
| 2 | "api/config.py — env-driven settings: GROQ_API_KEY, GROQ_MODEL, GROQ_MODELS, GROQ_VISION_MODEL, SESSION_SECRET, FRONTEND_ORIGIN…" | `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_MODELS`, `GROQ_VISION_MODEL` are **not in `api/config.py`**. They are read directly by `agent/graph.py`. `api/config.py` handles only: upload limits, image-gen settings (HF), CORS, session params, auth input limits, and rate-limit params. |
| 3 | "agent/tools.py — a contextvar (active_user_id or similar)…" | There are two ContextVars: `active_user_id` and `active_source_filter`. |
| 4 | "generate_image (Pollinations; image results flow through a media marker contract…)" | Correct. Marker is `"[GENERATED_IMAGE]"`. **Also: `generate_qr_code` uses the same marker contract.** The brief only mentions `generate_image`. |
| 5 | "API-key tools return plain-language 'not configured' when their key is missing." | Exact wording: Cricket: `"Cricket scores are not configured on this server."` Movies: `"Movie information is not configured on this server."` Email: `"Email sending is not configured on this server."` |
| 6 | "agent/voice.py — Groq text-to-speech with a gTTS fallback." | Correct for TTS. There is also **speech-to-text** (`transcribe_audio`) using Groq Whisper (`whisper-large-v3-turbo`). The brief does not mention STT. |
| 7 | "rag/ingest.py, rag/search.py — FAISS + HuggingFace embeddings, per-user indexes." | There is **one global FAISS index** (not per-user indexes). Per-user isolation is applied at query time by filtering chunk metadata on `user_id`. |
| 8 | "utils/security.py — upload validation (extension allowlist, size cap, path sanitization)." | Correct. But note: `utils/security.py` allowlist (for Streamlit path) includes `.py` and extensionless files; `api/config.py` allowlist (FastAPI path) does **not**. Two intentionally different allowlists. |
| 9 | Implied "vanilla CSS" from general instructions. | The frontend **uses Tailwind CSS**, not vanilla CSS (`tailwind.config.ts`, `postcss.config.mjs`, `tailwindcss` devDependency). |
| 10 | "database/db.py — separate SQLite store for agent chat/long-term memory." | `database/db.py` stores long-term memories, calendar events, and chat history at `data/assistant.db`. This is separate from `data/checkpoints.sqlite` (LangGraph) and `data/ui.sqlite` (thread metadata). |
| 11 | "SqliteSaver checkpoints at data/checkpoints.sqlite." | Correct — default path, configurable via `CHECKPOINT_DB_PATH`. |
| 12 | Brief implies `GROQ_MODEL` is primary and `GROQ_MODELS` is the fallback list. | Reversed: `GROQ_MODELS` (comma-separated) is the **preferred** setting; `GROQ_MODEL` is the **legacy** single-model fallback. |
| 13 | "tests/ — pytest; all external HTTP is mocked; no live network in tests." | Correct. |
| 14 | Brief mentions "Backend on Railway." | `render.yaml` is also present (Render deployment alternative). |
| 15 | Brief says gTTS installed "with --no-deps due to a click conflict." | Correct. Additionally, **the gTTS install line is duplicated** in the Dockerfile (two identical lines — harmless). |
| 16 | Brief does not mention `api/streaming.py`, `api/rate_limit.py`, `api/schemas.py`, `agent/llm_recovery.py`, `agent/quiz.py`, or `agent/image.py`. | All of these files exist. `api/streaming.py` is especially important — it contains `GENERATED_IMAGE_MARKER` and `extract_generated_image_filename()`. |
| 17 | Brief does not mention the **thread archive/unarchive** feature. | `archived_threads` table in `api/store.py`; `/threads/{id}/archive` and `/threads/{id}/unarchive` routes in `api/main.py`. |
| 18 | Brief does not mention **thread rename** or **thread export** features. | Both implemented: `PUT /threads/{id}/rename` and `GET /threads/{id}/export` (also `.export.md` alias). |
| 19 | Brief does not mention the **message edit** feature. | `POST /threads/{id}/edit` calls `remove_last_turn()` to strip the last user turn from the checkpoint before re-streaming. |
| 20 | Brief does not mention the **image library** or **media library** endpoints. | `GET /images`, `GET /library/media`, `GET /library/media/{folder}/{filename}`, `POST /images/generate` (direct image generation without chat). |

---

*Generated by code exploration — not inferred from comments or the brief.*
