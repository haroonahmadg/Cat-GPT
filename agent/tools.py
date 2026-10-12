from __future__ import annotations

import ast
import math
import contextvars
import operator as op
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.config import get_stream_writer

from rag.search import search_pdf, get_document_preview

from database.db import (
    get_connection,
    get_memories,
    save_memory,
)


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# STREAMING EVENTS
# =========================================================

def emit_tool_event(
    message: str,
    *,
    event: str,
    tool: str,
    **metadata: Any,
) -> None:
    """
    Emit a real backend event to the LangGraph stream.

    This function does not select tools and does not perform
    any routing.

    It only reports what the current tool is doing.
    """

    try:
        writer = get_stream_writer()

        writer(
            {
                "message": message,
                "event": event,
                "tool": tool,
                **metadata,
            }
        )

    except Exception:
        # Tool execution must still work when it is called
        # outside a LangGraph streaming context.
        pass


# =========================================================
# ACTIVE SOURCE FILTER (thread/message attachment scoping)
# =========================================================

active_source_filter: "contextvars.ContextVar[list[str] | None]" = (
    contextvars.ContextVar("active_source_filter", default=None)
)
active_user_id: "contextvars.ContextVar[str | None]" = (
    contextvars.ContextVar("active_user_id", default=None)
)

# =========================================================
# IMAGE GENERATION CONFIGURATION
# =========================================================

import os
import logging
import hashlib

logger = logging.getLogger(__name__)

# Pollinations' hosted Flux endpoint does not require an inference-credit
# balance or API token. Availability and free access are controlled by the
# provider, so failures are reported rather than silently billed elsewhere.
POLLINATIONS_IMAGE_URL = "https://image.pollinations.ai/prompt/"
POLLINATIONS_IMAGE_MODEL = os.getenv("POLLINATIONS_IMAGE_MODEL", "flux").strip() or "flux"
IMAGE_TIMEOUT_SECONDS = 120

GENERATED_IMAGE_DIR = Path(
    os.getenv("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))
) / "generated_images"
GENERATED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGE_PROMPT_LENGTH = 500

# The backend's event_stream() in api/main.py and extract_generated_image_filename()
# in api/streaming.py look for this exact prefix to detect that a tool's
# output is an image rather than plain text, and swap it into a
# dedicated "image" SSE event instead of displaying the marker as text.
GENERATED_IMAGE_MARKER = "[GENERATED_IMAGE]"


def get_generated_image_dir(user_id: str, *, create: bool = False) -> Path:
    """Return an opaque, account-scoped image directory."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("A valid user_id is required.")
    account_dir = GENERATED_IMAGE_DIR / hashlib.sha256(
        user_id.strip().encode("utf-8")
    ).hexdigest()
    if create:
        account_dir.mkdir(parents=True, exist_ok=True)
    return account_dir


# =========================================================
# CALCULATOR CONFIGURATION
# =========================================================

MAX_EXPRESSION_LENGTH = 500
MAX_POWER = 1000


# =========================================================
# ALLOWED OPERATORS
# =========================================================

BINARY_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
}


UNARY_OPERATORS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


# =========================================================
# ALLOWED MATHEMATICAL FUNCTIONS
# =========================================================

ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
}


# =========================================================
# ALLOWED CONSTANTS
# =========================================================

ALLOWED_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


# =========================================================
# SAFE POWER
# =========================================================

def _safe_power(
    base,
    exponent,
):
    """
    Safely calculate exponentiation.
    """

    if abs(exponent) > MAX_POWER:

        raise ValueError(
            f"Exponent is too large. "
            f"Maximum allowed absolute exponent is "
            f"{MAX_POWER}."
        )

    return op.pow(
        base,
        exponent,
    )


# =========================================================
# AST EVALUATOR
# =========================================================

def _evaluate_node(node):
    """
    Recursively evaluate a mathematical AST node.

    Only explicitly allowed AST node types are evaluated.
    """

    # -----------------------------------------------------
    # Numeric constants
    # -----------------------------------------------------

    if isinstance(
        node,
        ast.Constant,
    ):

        if isinstance(
            node.value,
            bool,
        ):

            raise ValueError(
                "Boolean values are not valid numbers."
            )

        if isinstance(
            node.value,
            (int, float),
        ):

            return node.value

        raise ValueError(
            "Only numeric values are allowed."
        )

    # -----------------------------------------------------
    # Binary operations
    # -----------------------------------------------------

    if isinstance(
        node,
        ast.BinOp,
    ):

        operator_type = type(
            node.op
        )

        if operator_type is ast.Pow:

            left = _evaluate_node(
                node.left
            )

            right = _evaluate_node(
                node.right
            )

            return _safe_power(
                left,
                right,
            )

        if (
            operator_type
            not in BINARY_OPERATORS
        ):

            raise ValueError(
                f"Unsupported operator: "
                f"{operator_type.__name__}"
            )

        left = _evaluate_node(
            node.left
        )

        right = _evaluate_node(
            node.right
        )

        operation = BINARY_OPERATORS[
            operator_type
        ]

        return operation(
            left,
            right,
        )

    # -----------------------------------------------------
    # Unary operations
    # -----------------------------------------------------

    if isinstance(
        node,
        ast.UnaryOp,
    ):

        operator_type = type(
            node.op
        )

        if (
            operator_type
            not in UNARY_OPERATORS
        ):

            raise ValueError(
                f"Unsupported unary operator: "
                f"{operator_type.__name__}"
            )

        operand = _evaluate_node(
            node.operand
        )

        operation = UNARY_OPERATORS[
            operator_type
        ]

        return operation(
            operand
        )

    # -----------------------------------------------------
    # Mathematical functions
    # -----------------------------------------------------

    if isinstance(
        node,
        ast.Call,
    ):

        if not isinstance(
            node.func,
            ast.Name,
        ):

            raise ValueError(
                "Only approved mathematical "
                "functions are allowed."
            )

        function_name = node.func.id

        if (
            function_name
            not in ALLOWED_FUNCTIONS
        ):

            raise ValueError(
                f"Unsupported function: "
                f"{function_name}"
            )

        if node.keywords:

            raise ValueError(
                "Keyword arguments are not supported."
            )

        function = ALLOWED_FUNCTIONS[
            function_name
        ]

        arguments = [
            _evaluate_node(
                argument
            )
            for argument in node.args
        ]

        try:

            return function(
                *arguments
            )

        except TypeError as exc:

            raise ValueError(
                f"Invalid arguments for function "
                f"'{function_name}'."
            ) from exc

    # -----------------------------------------------------
    # Mathematical constants
    # -----------------------------------------------------

    if isinstance(
        node,
        ast.Name,
    ):

        if (
            node.id
            in ALLOWED_CONSTANTS
        ):

            return ALLOWED_CONSTANTS[
                node.id
            ]

        raise ValueError(
            f"Unknown constant or variable: "
            f"{node.id}"
        )

    # -----------------------------------------------------
    # Everything else
    # -----------------------------------------------------

    raise ValueError(
        f"Unsupported expression element: "
        f"{type(node).__name__}"
    )

# =========================================================
# GENERIC-QUERY DETECTION (for rag_search's document-preview fallback)
# =========================================================

_GENERIC_QUERY_PHRASES = {
    "tell me about",
    "describe",
    "summarize",
    "summarise",
    "what is this",
    "what's this",
    "explain this",
    "this file",
    "this document",
    "the attached",
    "the file",
    "the document",
    "attached file",
    "attached document",
}


def _is_generic_query(query: str) -> bool:
    """
    True when a query is too vague to score well against any specific
    chunk via similarity search (e.g. "tell me about the attached
    file"), as opposed to a real content question (e.g. "what was my
    internship role"). Used only to decide whether rag_search should
    fall back to returning a document's opening chunks in order,
    instead of ranked-by-relevance results.

    Deliberately conservative: a short query OR one matching a known
    generic phrase triggers the fallback; anything else is treated as
    a real, specific question and left to normal similarity search.
    """

    lowered = query.strip().lower()

    if not lowered:
        return False

    word_count = len(lowered.split())

    if word_count <= 6:
        return True

    return any(phrase in lowered for phrase in _GENERIC_QUERY_PHRASES)
# =========================================================
# CALCULATION
# =========================================================

def _calculate(
    expression,
):
    """
    Parse and safely evaluate a mathematical expression.
    """

    if not isinstance(
        expression,
        str,
    ):

        raise ValueError(
            "Expression must be provided as text."
        )

    expression = expression.strip()

    if not expression:

        raise ValueError(
            "Please provide a mathematical expression."
        )

    if len(expression) > MAX_EXPRESSION_LENGTH:

        raise ValueError(
            f"Expression is too long. "
            f"Maximum length is "
            f"{MAX_EXPRESSION_LENGTH} characters."
        )

    try:

        tree = ast.parse(
            expression,
            mode="eval",
        )

    except SyntaxError as exc:

        raise ValueError(
            "Invalid mathematical expression."
        ) from exc
    except (ValueError, TypeError, RecursionError) as exc:

        raise ValueError(
            "Invalid mathematical expression."
        ) from exc

    result = _evaluate_node(
        tree.body
    )

    if isinstance(
        result,
        float,
    ):

        if not math.isfinite(
            result
        ):

            raise ValueError(
                "The calculation produced "
                "a non-finite result."
            )

    return result


# =========================================================
# RESULT FORMATTING
# =========================================================

def _format_result(
    result,
):
    """
    Convert the calculation result into a clean string.
    """

    if isinstance(
        result,
        float,
    ):

        if result.is_integer():

            return str(
                int(result)
            )

        return f"{result:.12g}"

    return str(result)


# =========================================================
# CALCULATOR TOOL
# =========================================================

@tool
def calculator(
    expression: str,
) -> str:
    """
    Calculate a mathematical expression safely.

    Supported operators:
    +, -, *, /, //, %, **

    Supported functions:
    sqrt, abs, round, ceil, floor

    Supported constants:
    pi, e

    Use this tool when the user needs arithmetic
    or mathematical calculations.
    """

    emit_tool_event(
        "Calculating expression",
        event="tool_started",
        tool="calculator",
    )

    try:

        result = _calculate(
            expression
        )

        formatted_result = (
            _format_result(
                result
            )
        )

        emit_tool_event(
            "Calculation completed",
            event="tool_completed",
            tool="calculator",
        )

        return formatted_result

    except ZeroDivisionError:

        emit_tool_event(
            "Calculation failed",
            event="tool_error",
            tool="calculator",
        )

        return (
            "Calculator error: "
            "division by zero is not allowed."
        )

    except OverflowError:

        emit_tool_event(
            "Calculation failed",
            event="tool_error",
            tool="calculator",
        )

        return (
            "Calculator error: "
            "the result is too large to calculate."
        )

    except ValueError as exc:

        emit_tool_event(
            "Calculation failed",
            event="tool_error",
            tool="calculator",
        )

        return (
            f"Calculator error: {exc}"
        )

    except RecursionError:

        emit_tool_event(
            "Calculation failed",
            event="tool_error",
            tool="calculator",
        )

        return (
            "Calculator error: "
            "the expression is too deeply nested."
        )

    except Exception:

        emit_tool_event(
            "Calculation failed",
            event="tool_error",
            tool="calculator",
        )

        return (
            "Calculator error: "
            "unable to evaluate this expression."
        )


# =========================================================
# WEB SEARCH TOOL
# =========================================================

@tool
def web_search(
    query: str,
) -> str:
    """
    Search the web for current or factual information.

    Use this tool when the user asks for:

    - current information
    - recent news
    - live or changing information
    - information that requires web research
    - a specific topic that may not be reliably known
      from the model

    Do not use this tool for simple arithmetic.
    """

    if not isinstance(
        query,
        str,
    ):

        return (
            "Web search error: "
            "query must be text."
        )

    query = query.strip()

    if not query:

        return (
            "Web search error: "
            "search query cannot be empty."
        )

    if len(query) > 500:

        return (
            "Web search error: "
            "search query is too long."
        )

    emit_tool_event(
        "Searching the web",
        event="tool_started",
        tool="web_search",
    )

    try:

        search = TavilySearch(
            max_results=5,
            topic="general",
        )

        result = search.invoke(
            {
                "query": query,
            }
        )

        if not result:

            emit_tool_event(
                "Web search completed",
                event="tool_completed",
                tool="web_search",
            )

            return (
                "No search results were found."
            )

        emit_tool_event(
            "Web search completed",
            event="tool_completed",
            tool="web_search",
        )

        return str(
            result
        )

    except Exception as exc:

        emit_tool_event(
            "Web search failed",
            event="tool_error",
            tool="web_search",
        )

        return (
            f"Web search error: {exc}"
        )


# =========================================================
# RAG SEARCH TOOL
# =========================================================


@tool
def rag_search(
    query: str,
) -> str:
    """
    Search the user's indexed PDF documents for relevant
    information.

    Use this tool when the user asks a question that should
    be answered from information contained in uploaded or
    indexed documents.
    """

    if not isinstance(
        query,
        str,
    ):

        return (
            "RAG search error: "
            "query must be text."
        )

    query = query.strip()

    if not query:

        return (
            "RAG search error: "
            "query cannot be empty."
        )

    if len(query) > 500:

        return (
            "RAG search error: "
            "query is too long."
        )

    emit_tool_event(
        "Searching indexed documents",
        event="tool_started",
        tool="rag_search",
    )

    try:

        source_filter = active_source_filter.get()
        user_id = active_user_id.get()
        results = search_pdf(
            query,
            k=4,
            source_filter=source_filter,
            user_id=user_id
        )

        # A generic request ("tell me about this file") or a real
        # question that happened to score nothing against any chunk —
        # when we know exactly which file(s) are attached, fall back
        # to just returning that document's opening content instead of
        # reporting nothing was found.
        used_preview = False

        if source_filter and (not results or _is_generic_query(query)):
            try:
                preview_results = get_document_preview(
                    source_filter,
                    k=4,user_id=user_id
                )
                if preview_results:
                    results = preview_results
                    used_preview = True
            except (FileNotFoundError, ValueError):
                # No metadata available for a preview — keep whatever
                # similarity search already returned (possibly empty).
                pass

        if not results:

            emit_tool_event(
                "Document search completed",
                event="tool_completed",
                tool="rag_search",
            )

            if source_filter:
                return (
                    "No relevant information was found in the "
                    "currently attached document(s)."
                )

            return (
                "No relevant information was found "
                "in the uploaded documents."
            )

        formatted_results = []

        for index, result in enumerate(
            results,
            start=1,
        ):

            if not isinstance(result, dict):
                continue

            content = (
                result.get(
                    "content",
                    "",
                ) or ""
            ).strip()

            metadata = result.get(
                "metadata",
                {},
            ) or {}

            if not content:
                continue

            page = metadata.get(
                "page"
            )

            if isinstance(page, int):

                page_number = (
                    page + 1
                )

                source_info = (
                    f"Page {page_number}"
                )

            else:

                source_info = (
                    "Unknown page"
                )

            formatted_results.append(
                f"Result {index} "
                f"({source_info}):\n"
                f"{content}"
            )

        if not formatted_results:

            emit_tool_event(
                "Document search completed",
                event="tool_completed",
                tool="rag_search",
            )

            return (
                "No readable information was found "
                "in the uploaded documents."
            )

        emit_tool_event(
            "Document search completed",
            event="tool_completed",
            tool="rag_search",
            result_count=len(
                formatted_results
            ),
            preview_fallback=used_preview,
        )

        return "\n\n".join(
            formatted_results
        )

    except FileNotFoundError:

        emit_tool_event(
            "Document index was not found",
            event="tool_error",
            tool="rag_search",
        )

        return (
            "RAG search error: "
            "no FAISS index was found. "
            "Please upload and ingest a PDF first."
        )

    except Exception as exc:

        emit_tool_event(
            "Document search failed",
            event="tool_error",
            tool="rag_search",
        )

        return (
            f"RAG search error: {exc}"
        )


# =========================================================
# IMAGE GENERATION ERROR DESCRIPTION
# =========================================================

def _describe_image_error(exc: Exception) -> str:
    """
    Turn an image-provider failure into a specific, user-facing reason.

    The model sees this text as the tool result, so it must name the
    real cause instead of something generic like "service unreachable".
    """

    status = getattr(getattr(exc, "response", None), "status_code", None)

    if status == 429:
        return "the free image service is rate-limited. Please wait and try again."

    if status:
        return f"the image service returned status {status}."

    if isinstance(exc, requests.exceptions.ConnectionError):
        return (
            "could not connect to the free image service. Check your "
            "internet connection."
        )

    return f"{type(exc).__name__}: {exc}"


# =========================================================
# IMAGE GENERATION TOOL
# =========================================================

@tool
def generate_image(
    prompt: str,
) -> str:
    """
    Generate an image from a text description.

    Use this tool when the user asks you to create, draw, generate,
    or make a picture, image, illustration, or artwork of something.

    Do not use this tool for editing or analyzing an existing uploaded
    image — only for creating a new image from a text description.
    """

    if not isinstance(prompt, str):
        return "Image generation error: prompt must be text."

    prompt = prompt.strip()

    if not prompt:
        return "Image generation error: prompt cannot be empty."

    if len(prompt) > MAX_IMAGE_PROMPT_LENGTH:
        return (
            "Image generation error: prompt is too long "
            f"(max {MAX_IMAGE_PROMPT_LENGTH} characters)."
        )

    emit_tool_event(
        "Generating image",
        event="tool_started",
        tool="generate_image",
    )

    saved_path = None

    try:
        response = requests.get(
            f"{POLLINATIONS_IMAGE_URL}{requests.utils.quote(prompt, safe='')}",
            params={
                "width": 1024,
                "height": 1024,
                "nologo": "true",
                "model": POLLINATIONS_IMAGE_MODEL,
            },
            timeout=IMAGE_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        if not response.headers.get("content-type", "").lower().startswith("image/"):
            raise ValueError("the service returned data that was not an image.")
        if not response.content:
            raise ValueError("the service returned an empty image.")
        user_id = active_user_id.get()
        if not user_id:
            raise RuntimeError("An authenticated user is required to save generated images.")
        filename = f"{uuid.uuid4().hex[:16]}.png"
        saved_path = get_generated_image_dir(user_id, create=True) / filename

        from PIL import Image
        from io import BytesIO

        with Image.open(BytesIO(response.content)) as image:
            image.convert("RGB").save(saved_path, format="PNG")

        if saved_path.stat().st_size == 0:
            raise ValueError("received an empty image.")

        emit_tool_event(
            "Image generation completed",
            event="tool_completed",
            tool="generate_image",
        )

        # The marker + filename is parsed by api/streaming.py to produce
        # a dedicated "image" SSE event. filename only (not the full
        # path) is sent to the client — api/main.py resolves it only
        # within the authenticated user's image directory.
        return f"{GENERATED_IMAGE_MARKER}{filename}"

    except requests.exceptions.Timeout:
        error_message = "the request timed out. Please try again in a moment."

    except Exception as exc:
        logger.exception("Image generation failed")
        error_message = _describe_image_error(exc)

    # Shared failure path: remove any half-written file, report once.
    if saved_path is not None:
        try:
            saved_path.unlink(missing_ok=True)
        except Exception:
            pass

    emit_tool_event(
        "Image generation failed",
        event="tool_error",
        tool="generate_image",
    )

    return f"Image generation error: {error_message}"


# =========================================================
# CREATE EVENT TOOL
# =========================================================

@tool
def create_event(
    title: str,
    event_date: str,
    event_time: str = "",
    description: str = "",
) -> str:
    """
    Create a calendar event.

    event_date must use YYYY-MM-DD format.
    event_time should use HH:MM format when provided.
    """

    if (
        not isinstance(
            title,
            str,
        )
        or not title.strip()
    ):

        return (
            "Calendar error: "
            "event title cannot be empty."
        )

    if len(title.strip()) > 200:

        return (
            "Calendar error: "
            "event title is too long (max 200 characters)."
        )

    if (
        not isinstance(
            event_date,
            str,
        )
        or not event_date.strip()
    ):

        return (
            "Calendar error: "
            "event date cannot be empty."
        )

    if not isinstance(
        event_time,
        str,
    ):

        return (
            "Calendar error: "
            "event time must be text."
        )

    if not isinstance(
        description,
        str,
    ):

        return (
            "Calendar error: "
            "description must be text."
        )

    if len(description.strip()) > 2000:

        return (
            "Calendar error: "
            "description is too long (max 2000 characters)."
        )

    try:

        datetime.strptime(
            event_date.strip(),
            "%Y-%m-%d",
        )

    except ValueError:

        return (
            "Calendar error: "
            "date must use YYYY-MM-DD format."
        )

    if event_time.strip():

        try:

            datetime.strptime(
                event_time.strip(),
                "%H:%M",
            )

        except ValueError:

            return (
                "Calendar error: "
                "time must use HH:MM format."
            )

    emit_tool_event(
        "Creating calendar event",
        event="tool_started",
        tool="create_event",
    )

    connection = None

    try:

        connection = get_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO events
            (title, event_date, event_time, description)
            VALUES (?, ?, ?, ?)
            """,
            (
                title.strip(),
                event_date.strip(),
                event_time.strip(),
                description.strip(),
            ),
        )

        event_id = cursor.lastrowid

        connection.commit()

        emit_tool_event(
            "Calendar event created",
            event="tool_completed",
            tool="create_event",
        )

        return (
            f"Event created successfully.\n"
            f"Event ID: {event_id}\n"
            f"Title: {title.strip()}\n"
            f"Date: {event_date.strip()}\n"
            f"Time: "
            f"{event_time.strip() or 'Not specified'}"
        )

    except Exception as exc:

        emit_tool_event(
            "Calendar event creation failed",
            event="tool_error",
            tool="create_event",
        )

        return (
            f"Calendar error: {exc}"
        )

    finally:

        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


# =========================================================
# LIST EVENTS TOOL
# =========================================================

@tool
def list_events() -> str:
    """
    List all calendar events.
    """

    emit_tool_event(
        "Loading calendar events",
        event="tool_started",
        tool="list_events",
    )

    connection = None

    try:

        connection = get_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                title,
                event_date,
                event_time,
                description
            FROM events
            ORDER BY event_date, event_time
            """
        )

        events = cursor.fetchall()

        if not events:

            emit_tool_event(
                "Calendar lookup completed",
                event="tool_completed",
                tool="list_events",
            )

            return (
                "No calendar events found."
            )

        results = []

        for event in events:

            (
                event_id,
                title,
                date,
                time,
                description,
            ) = event

            results.append(
                f"Event ID: {event_id}\n"
                f"Title: {title}\n"
                f"Date: {date}\n"
                f"Time: "
                f"{time or 'Not specified'}\n"
                f"Description: "
                f"{description or 'None'}"
            )

        emit_tool_event(
            "Calendar lookup completed",
            event="tool_completed",
            tool="list_events",
        )

        return "\n\n".join(
            results
        )

    except Exception as exc:

        emit_tool_event(
            "Calendar lookup failed",
            event="tool_error",
            tool="list_events",
        )

        return (
            f"Calendar error: {exc}"
        )

    finally:

        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


# =========================================================
# DELETE EVENT TOOL
# =========================================================

@tool
def delete_event(
    event_id: int,
) -> str:
    """
    Delete a calendar event using its event ID.
    """

    if (
        not isinstance(
            event_id,
            int,
        )
        or isinstance(
            event_id,
            bool,
        )
    ):

        return (
            "Calendar error: "
            "event ID must be an integer."
        )

    emit_tool_event(
        "Deleting calendar event",
        event="tool_started",
        tool="delete_event",
    )

    connection = None

    try:

        connection = get_connection()

        cursor = connection.cursor()

        cursor.execute(
            "DELETE FROM events WHERE id = ?",
            (event_id,),
        )

        if cursor.rowcount == 0:

            emit_tool_event(
                "Calendar event was not found",
                event="tool_completed",
                tool="delete_event",
            )

            return (
                f"No event found with ID "
                f"{event_id}."
            )

        connection.commit()

        emit_tool_event(
            "Calendar event deleted",
            event="tool_completed",
            tool="delete_event",
        )

        return (
            f"Event {event_id} "
            f"deleted successfully."
        )

    except Exception as exc:

        emit_tool_event(
            "Calendar event deletion failed",
            event="tool_error",
            tool="delete_event",
        )

        return (
            f"Calendar error: {exc}"
        )

    finally:

        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


# =========================================================
# SAVE MEMORY TOOL
# =========================================================

@tool
def save_user_memory(
    memory_key: str,
    memory_value: str,
) -> str:
    """
    Save an important piece of information the user wants remembered
    for future conversations — a preference, fact, or detail.

    Use this tool when the user explicitly asks you to remember
    something, or shares a stable preference/fact they'll likely want
    recalled later.
    """

    emit_tool_event(
        "Saving user memory",
        event="tool_started",
        tool="save_user_memory",
    )

    try:
        user_id = active_user_id.get()

        if not user_id:
            emit_tool_event(
                "Saving user memory failed",
                event="tool_error",
                tool="save_user_memory",
            )
            return "Memory error: no active user session was found."

        save_memory(
            memory_key.strip(),
            memory_value.strip(),
            user_id,
        )

        emit_tool_event(
            "User memory saved",
            event="tool_completed",
            tool="save_user_memory",
        )

        return (
            f"Memory saved successfully: "
            f"{memory_key.strip()} = "
            f"{memory_value.strip()}"
        )

    except Exception as exc:
        emit_tool_event(
            "Saving user memory failed",
            event="tool_error",
            tool="save_user_memory",
        )
        return f"Memory error: {exc}"

# =========================================================
# GET MEMORIES TOOL
# =========================================================

@tool
def get_user_memories() -> str:

    """
    Retrieve previously saved facts, preferences, or details about the
    user from long-term memory.
    """
    emit_tool_event(
        "Loading user memories",
        event="tool_started",
        tool="get_user_memories",
    )

    try:
        user_id = active_user_id.get()

        if not user_id:
            emit_tool_event(
                "Memory lookup failed",
                event="tool_error",
                tool="get_user_memories",
            )
            return "Memory error: no active user session was found."

        memories = get_memories(user_id)

        if not memories:
            emit_tool_event(
                "Memory lookup completed",
                event="tool_completed",
                tool="get_user_memories",
            )
            return "No saved memories found."

        result = "\n".join(f"- {key}: {value}" for key, value in memories)

        emit_tool_event(
            "Memory lookup completed",
            event="tool_completed",
            tool="get_user_memories",
        )

        return result

    except Exception as exc:
        emit_tool_event(
            "Memory lookup failed",
            event="tool_error",
            tool="get_user_memories",
        )
        return f"Memory error: {exc}"