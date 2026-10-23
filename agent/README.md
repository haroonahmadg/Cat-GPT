# Agent

This package contains the LangGraph assistant and its model-facing tools.

- `graph.py` defines the agent state, model setup, tool registration, graph
  routing, streaming/run helpers, and checkpoint integration.
- `tools.py` implements callable assistant tools and their activity events.
- `llm_recovery.py` handles supported model/tool-call recovery paths.
- `voice.py` contains speech-related agent integrations.
- `quiz.py` contains quiz functionality.

The API passes authenticated user and thread context into the graph. Preserve
that namespacing and user isolation when changing agent state or checkpoint
configuration. Streaming tool activity and response events are part of the API
contract; coordinate changes with `api/streaming.py` and the frontend client.

Research-mode routing remains implemented in the graph and defaults off when
the request does not enable it. The current Next.js UI does not expose that
toggle.

See the [project guide](../README.md) for setup and tests.
