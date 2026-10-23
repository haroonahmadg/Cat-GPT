# AGENTS.md — Cat GPT project brief

> Code wins over comments. This repo is the live implementation for Cat GPT.

## Project purpose

Cat GPT is a portfolio chat assistant built on FastAPI, LangGraph, FAISS-backed RAG, and a Next.js frontend. The assistant must self-identify as "Cat GPT" in the system prompt, and the app uses per-user isolation for threads, uploads, RAG indexes, and memory.

## Current architecture

- Backend: `api/main.py` owns the FastAPI routes, auth, streaming, uploads, generated images, and voice endpoints.
- Agent: `agent/graph.py` is the LangGraph orchestrator. It keeps `research_mode` in `AgentState` and routes to research plan/synthesis when enabled.
- Tools: `agent/tools.py` is the tool inventory; the app emits custom activity/citation events with `emit_tool_event()`.
- RAG: `rag/ingest.py` and `rag/search.py` maintain user-scoped FAISS metadata and retrieval previews.
- Frontend: `forntend/` is the Next.js App Router shell. It renders streaming chat, citations, markdown, export, and the cat mood indicator.

## Runtime contract

SSE event types currently emitted by the app are:
- `text` — `{"text": "..."}`
- `image` — `{"url": "/generated-images/<file>"}`
- `activity` — `{"message": "..."}`
- `citations` — `{"items": [{"file": str, "page": int | null, "preview": str}]}`
- `done` — `{}`
- `error` — `{"message": "..."}`

Clients must ignore unknown event types. Existing event shapes stay stable; new event types are additive only.

## Feature status

Implemented and in the codebase:
- F1 markdown + syntax highlighting
- F2 cat mood indicator
- F3 export / print-to-PDF
- F4 citations + sources panel
- F5 voice input / Whisper transcription
- F6 research mode subgraph with `research_mode` state and plan/synthesis routing

## Repo rules

- Keep per-user isolation everywhere.
- Keep the event contract additive, not breaking.
- Keep LangGraph orchestration in the agent layer; do not hand-roll routing in the API.
- Preserve backward compatibility for old checkpoints when adding state fields.
- Focus on surgical fixes and existing project patterns.

## F6 audit

| Check | Status | Evidence |
|---|---|---|
| `AgentState.research_mode` | Present | `agent/graph.py` defines `research_mode: bool` and passes it through graph input. |
| Research routing | Present | `build_graph()` routes `START` and tool completion to research nodes when enabled. |
| Research plan event | Present | `research_plan_node()` emits a `plan` custom event; stream remains additive. |
| Synthesis | Present | `research_synthesize_node()` creates the final answer grounded in research results. |
| Backward compatibility | Present | `state.get("research_mode", False)` keeps older checkpoint values valid. |
