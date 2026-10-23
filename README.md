# Cat GPT

Cat GPT is a full-stack AI assistant. The primary web UI is a Next.js app,
served alongside a FastAPI backend. A Streamlit interface remains available
through the root `app.py`; it shares some backend and local data code with the
FastAPI application.

## Architecture

```text
Browser
  ├── Next.js UI (forntend/)
  └── FastAPI API (api/)
        ├── LangGraph agent and tools (agent/)
        ├── Retrieval-augmented generation / FAISS (rag/)
        ├── Account and UI persistence (api/store.py)
        └── Chat, thread, and memory persistence (database/)
```

The FastAPI API owns authentication, user-scoped thread operations, streamed
responses, uploads, image/media endpoints, and voice endpoints. The agent
orchestrates model and tool calls. RAG ingestion and search use a local
FAISS index. SQLite and generated files live under `data/` by default.

The root `app.py` is a separate Streamlit interface, not the Next.js entry
point. It is retained for compatibility and shares local storage paths with
the backend. Avoid running migration or cleanup operations against live data
without making a backup first.

## Local development

### Requirements

- Python version compatible with `requirements.txt` and the installed
  dependencies.
- Node.js and npm.
- API credentials configured locally; see `.env.example` and
  `cat-gpt-deploy/frontend.env.local.example`.

### Start the FastAPI backend

From the repository root, create and activate a virtual environment, install
Python dependencies, and copy the example environment file:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set the required values in `.env` before starting the API. At minimum, the
example identifies `GROQ_API_KEY`, `SESSION_SECRET`, and `GOOGLE_CLIENT_ID`.
Optional integrations and limits are documented there as well. Never commit
`.env` or put server credentials in a `NEXT_PUBLIC_` variable.

```powershell
python -m uvicorn api.main:app --reload
```

The API is available at `http://localhost:8000`; its health endpoint is
`http://localhost:8000/health` and the interactive API docs are at
`http://localhost:8000/docs`.

### Start the Next.js frontend

In a second terminal:

```powershell
Set-Location .\forntend
npm ci
Copy-Item ..\cat-gpt-deploy\frontend.env.local.example .env.local
npm run dev
```

The default frontend API URL is `http://localhost:8000`. Override it with
`NEXT_PUBLIC_API_URL` in `forntend/.env.local` when the backend runs elsewhere.
Open `http://localhost:3000`.

The directory is intentionally named `forntend` in the repository and
deployment configuration; retain that spelling when configuring paths.

### Optional Streamlit interface

The separate legacy/alternate UI is started from the repository root:

```powershell
streamlit run app.py
```

It uses the same configured services and some of the same local databases.

## Tests and checks

Run the Python test suite from the repository root:

```powershell
python -m pytest
```

Build and type-check the frontend:

```powershell
Set-Location .\forntend
npm run build
```

Tests should use temporary databases, local fixtures, or mocked upstream
services; do not run tests against production accounts or data directories.

## Important runtime contracts

- Keep authenticated data isolated by user across threads, uploads, memories,
  generated media, and retrieval.
- Chat streaming uses server-sent events. Existing event types and payloads
  are consumed by both clients; preserve their shapes when changing streaming.
- The LangGraph agent contains research-mode routing, but the current Next.js
  UI does not expose or send a research-mode toggle.
- Local SQLite databases, FAISS indexes, uploads, and generated media are
  runtime data, not source files. Back them up before migrations or manual
  cleanup.

## Repository map

| Path | Purpose |
| --- | --- |
| [`api/`](api/README.md) | FastAPI app, authentication, request schemas, and API services |
| [`agent/`](agent/README.md) | LangGraph orchestration, model recovery, voice, and tools |
| [`rag/`](rag/README.md) | Document ingestion, FAISS indexing, retrieval, and deletion |
| [`database/`](database/README.md) | SQLite thread, message, and memory persistence helpers |
| [`forntend/`](forntend/README.md) | Next.js App Router UI and typed API client |
| [`tests/`](tests/README.md) | Pytest suite and test fixtures |
| [`scripts/`](scripts/README.md) | Explicitly invoked maintenance and migration scripts |
| [`utils/`](utils/README.md) | Shared file and input security helpers |
| [`cat-gpt-deploy/`](cat-gpt-deploy/README.md) | Deployment examples and provider-specific guidance |
| [`cat-gpt-deploy/DEPLOYMENT.md`](cat-gpt-deploy/DEPLOYMENT.md) | Render/Vercel deployment guide |
| `app.py` | Separate Streamlit interface |
| `data/` | Runtime databases, uploads, FAISS index, and generated media |

## Deployment

The root `render.yaml` describes the FastAPI service. The frontend is deployed
separately from `forntend/`; see
[`cat-gpt-deploy/DEPLOYMENT.md`](cat-gpt-deploy/DEPLOYMENT.md) for the full
Render and Vercel setup and the limitations of ephemeral deployment storage.
