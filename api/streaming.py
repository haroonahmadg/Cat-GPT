"""
Helpers for turning agent.graph.stream_agent()'s (mode, chunk) tuples into
Server-Sent Events.

The text/activity extraction logic here mirrors app.py's
extract_streamed_ai_text() / extract_activity_label(). It is duplicated
(not imported) for the same reason as api/store.py's duplication of
app.py's UI-database helpers: app.py cannot be imported outside a
Streamlit runtime. The duplicated logic here is intentionally kept small
and dependency-free (plain dict/string handling only).
"""

from __future__ import annotations

import json
from typing import Any


def format_sse(event: str, data: dict[str, Any]) -> str:
    """Format one Server-Sent Event."""

    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def extract_updates_text(node_update: Any) -> str:
    """
    Extract assistant text from an "updates" stream chunk, e.g.
    {"assistant": {"messages": [...]}} or {"tools": {"messages": [...]}}.
    """

    if not isinstance(node_update, dict):
        return ""

    assistant_update = node_update.get("assistant")

    if not isinstance(assistant_update, dict):
        return ""

    messages = assistant_update.get("messages", [])

    if not messages:
        return ""

    last_message = messages[-1]
    content = getattr(last_message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    text_parts.append(str(text))

        return "\n".join(text_parts)

    return ""


def extract_activity_message(custom_event: Any) -> str:
    """
    Extract a human-readable label from a "custom" stream chunk — the
    events emitted by agent.tools.emit_tool_event() and
    rag.ingest.emit_ingest_event() via langgraph.config.get_stream_writer().
    """

    if isinstance(custom_event, dict):
        message = custom_event.get("message")

        if isinstance(message, str) and message.strip():
            return message.strip()

    return ""



GENERATED_IMAGE_MARKER = "[GENERATED_IMAGE]"


def extract_generated_image_filename(node_update: Any) -> str | None:
    """
    Checks a "updates" stream chunk's tool output for the image marker
    generate_image() (agent/tools.py) writes on success, and returns
    just the filename if found, else None.

    Mirrors extract_updates_text()'s shape-handling exactly, but reads
    from the "tools" node's output rather than "assistant", since a
    tool's result (not the assistant's own message) is where this
    marker appears.
    """

    if not isinstance(node_update, dict):
        return None

    tools_update = node_update.get("tools")

    if not isinstance(tools_update, dict):
        return None

    messages = tools_update.get("messages", [])

    if not messages:
        return None

    last_message = messages[-1]
    content = getattr(last_message, "content", "")

    if isinstance(content, str) and content.startswith(GENERATED_IMAGE_MARKER):
        return content[len(GENERATED_IMAGE_MARKER):].strip()

    return None