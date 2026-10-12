"""
Recovery for Groq's `tool_use_failed` (HTTP 400) errors.

Some models write a tool call as text (e.g.
`<function=rag_search><parameter=query>...</parameter></function>`)
instead of the structured format Groq requires, and Groq rejects the
request outright. The error response still carries the model's intended
call in its `failed_generation` field, so this module extracts it and
builds a LangGraph-compatible tool call, letting the graph's normal
tool-routing (LangGraph's tools_condition / ToolNode in agent/graph.py) execute it
as if the model had called it correctly.

This is a safety net, not a primary code path: models that produce
valid structured tool calls do not exercise this module. It exists for
resilience when a provider returns malformed tool-call output.

Deliberately has no project imports, so it stays safe to use from
agent/graph.py without import-order concerns.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

_FUNCTION_RE = re.compile(r"<function=([A-Za-z0-9_\-]+)>", re.IGNORECASE)
_PARAM_RE = re.compile(
    r"<parameter=([A-Za-z0-9_\-]+)>\s*(.*?)\s*</parameter>",
    re.IGNORECASE | re.DOTALL,
)


def _error_body(exc: Exception) -> dict[str, Any]:
    """Best-effort extraction of the Groq error's JSON body.

    The groq SDK attaches the parsed response body to raised exceptions
    as `.body`; this degrades gracefully to an empty dict if that shape
    ever changes, rather than raising a second error while recovering
    from the first.
    """

    body = getattr(exc, "body", None)

    if isinstance(body, dict):
        inner = body.get("error")
        return inner if isinstance(inner, dict) else body

    return {}


def is_tool_use_failed(exc: Exception) -> bool:
    """True if this exception is Groq's malformed-tool-call rejection."""

    if _error_body(exc).get("code") == "tool_use_failed":
        return True

    return "tool_use_failed" in str(exc)


def extract_failed_generation(exc: Exception) -> str:
    """Return the model's raw malformed output, or the error text as a
    fallback when the structured body isn't available."""

    value = _error_body(exc).get("failed_generation")

    if isinstance(value, str) and value.strip():
        return value

    # Fallback: str(exc) renders newlines as a literal backslash-n.
    return str(exc).replace("\\n", "\n")


def recover_tool_call(
    exc: Exception,
    allowed_tools: set[str],
) -> dict[str, Any] | None:
    """
    Parse the intended tool call out of a tool_use_failed error.

    Returns a dict shaped for AIMessage(tool_calls=[...]):
        {"name": ..., "args": {...}, "id": ..., "type": "tool_call"}

    Returns None (never recovers) when:
      - this isn't a tool_use_failed error at all
      - no function name could be parsed from the malformed output
      - the parsed function name isn't in allowed_tools (never execute
        a tool the caller didn't explicitly approve, even if the
        parsed name looks plausible)
      - no usable arguments could be parsed

    Callers should always pass the exact set of currently-bound tool
    names as allowed_tools — this is the only safety check standing
    between a garbled model output and code execution.
    """

    if not is_tool_use_failed(exc):
        return None

    raw = extract_failed_generation(exc)

    function_match = _FUNCTION_RE.search(raw)

    if not function_match:
        return None

    name = function_match.group(1)

    if name not in allowed_tools:
        return None

    args = {
        key: value.strip()
        for key, value in _PARAM_RE.findall(raw)
        if value.strip()
    }

    if not args:
        return None

    return {
        "name": name,
        "args": args,
        "id": f"recovered_{uuid.uuid4().hex[:12]}",
        "type": "tool_call",
    }