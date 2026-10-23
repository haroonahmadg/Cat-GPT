from __future__ import annotations

import ast
import math
import contextvars
import operator as op
import re
import secrets
import string
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
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

HF_IMAGE_URL = "https://router.huggingface.co/v1/images/generations"
HF_IMAGE_MODELS = (
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1-schnell",
)
HF_IMAGE_STYLE_PREFIX = (
    "photorealistic photograph, natural lighting, fine detail, "
    "50mm lens — "
)
POLLINATIONS_IMAGE_URL = "https://image.pollinations.ai/prompt/"
IMAGE_TIMEOUT_SECONDS = 20

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
            source = metadata.get("source")
            page_number = page + 1 if isinstance(page, int) else None

            if page_number is not None:
                source_info = (
                    f"Page {page_number}"
                )

            else:

                source_info = (
                    "Unknown page"
                )

            source_marker = ""
            if isinstance(source, str) and source.strip():
                source_file = Path(source).name
                citation = {
                    "file": source_file,
                    "page": page_number,
                    "preview": str(metadata.get("preview") or content)[:240],
                }
                emit_tool_event(
                    "Document source found",
                    event="citation",
                    tool="rag_search",
                    citation=citation,
                )
                page_suffix = f" p.{page_number}" if page_number is not None else ""
                source_marker = f"\n[source: {source_file}{page_suffix}]"

            formatted_results.append(
                f"Result {index} "
                f"({source_info}):\n"
                f"{content}{source_marker}"
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
    """Turn provider errors into plain, specific messages for the model."""

    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 429:
        return "image service is busy or not configured right now, please try again"
    if status:
        return f"image service returned status {status}."
    if isinstance(exc, (requests.exceptions.Timeout, httpx.TimeoutException)):
        return "image service timed out, please try again"
    if isinstance(exc, (requests.exceptions.ConnectionError, httpx.RequestError)):
        return "image service could not be reached, please try again"
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
    api_key = (
        os.getenv("HF_API_KEY", "").strip()
        or os.getenv("HF_API_TOKEN", "").strip()
    )
    try:
        enhanced_prompt = f"{HF_IMAGE_STYLE_PREFIX}{prompt.strip()}"
        last_error = None
        for model_name in (HF_IMAGE_MODELS if api_key else ()):
            try:
                response = httpx.post(
                    HF_IMAGE_URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": model_name, "prompt": enhanced_prompt},
                    timeout=IMAGE_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                if response.headers.get("content-type", "").lower().startswith("image/"):
                    image_bytes = response.content
                else:
                    payload = response.json()
                    data = payload.get("data")
                    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
                        raise ValueError("the service response did not include image data.")
                    image_result = data[0]
                    encoded_image = image_result.get("b64_json")
                    image_url = image_result.get("url")
                    if isinstance(encoded_image, str) and encoded_image:
                        import base64

                        image_bytes = base64.b64decode(encoded_image, validate=True)
                    elif isinstance(image_url, str) and image_url:
                        emit_tool_event(
                            "Retrieving generated image",
                            event="tool_started",
                            tool="generate_image",
                        )
                        image_response = httpx.get(
                            image_url,
                            timeout=IMAGE_TIMEOUT_SECONDS,
                        )
                        image_response.raise_for_status()
                        image_bytes = image_response.content
                    else:
                        raise ValueError("the service response did not include image data.")
                if not image_bytes:
                    raise ValueError("the service returned an empty image.")
                user_id = active_user_id.get()
                if not user_id:
                    raise RuntimeError("An authenticated user is required to save generated images.")
                filename = f"{uuid.uuid4().hex[:16]}.png"
                saved_path = get_generated_image_dir(user_id, create=True) / filename

                from PIL import Image
                from io import BytesIO

                with Image.open(BytesIO(image_bytes)) as image:
                    image.convert("RGB").save(saved_path, format="PNG")

                if saved_path.stat().st_size == 0:
                    raise ValueError("received an empty image.")

                emit_tool_event(
                    "Image generation completed",
                    event="tool_completed",
                    tool="generate_image",
                )
                return f"{GENERATED_IMAGE_MARKER}{filename}"
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Hugging Face image generation failed with %s for %s",
                    type(exc).__name__,
                    model_name,
                )
                if model_name == HF_IMAGE_MODELS[-1]:
                    break
                logger.info(
                    "Retrying Hugging Face image generation with fallback model %s",
                    HF_IMAGE_MODELS[1],
                )
                continue

        if not api_key:
            logger.info("HF_API_KEY is not configured; trying Pollinations fallback")
        else:
            logger.warning(
                "Hugging Face image generation failed with both configured models (%s); "
                "trying Pollinations fallback",
                type(last_error).__name__ if last_error else "unknown error",
            )
        emit_tool_event(
            "Trying alternate image provider",
            event="tool_started",
            tool="generate_image",
        )
        fallback_response = httpx.get(
            f"{POLLINATIONS_IMAGE_URL}{quote(prompt, safe='')}",
            params={"width": 1024, "height": 1024, "nologo": "true"},
            timeout=IMAGE_TIMEOUT_SECONDS,
        )
        fallback_response.raise_for_status()
        if not fallback_response.headers.get("content-type", "").lower().startswith("image/"):
            raise ValueError("the alternate image provider returned a non-image response.")
        fallback_bytes = fallback_response.content
        if not fallback_bytes:
            raise ValueError("the alternate image provider returned an empty image.")
        user_id = active_user_id.get()
        if not user_id:
            raise RuntimeError("An authenticated user is required to save generated images.")
        filename = f"{uuid.uuid4().hex[:16]}.png"
        saved_path = get_generated_image_dir(user_id, create=True) / filename

        from PIL import Image
        from io import BytesIO

        with Image.open(BytesIO(fallback_bytes)) as image:
            image.convert("RGB").save(saved_path, format="PNG")

        if saved_path.stat().st_size == 0:
            raise ValueError("received an empty image.")

        emit_tool_event(
            "Image generation completed",
            event="tool_completed",
            tool="generate_image",
        )
        return f"{GENERATED_IMAGE_MARKER}{filename}"

    except (requests.exceptions.Timeout, httpx.TimeoutException):
        error_message = "image service timed out, please try again"
    except Exception as exc:
        logger.warning("Image generation failed with %s", type(exc).__name__)
        error_message = _describe_image_error(exc)

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


# =========================================================
# EXTRA ASSISTANT TOOLS
# =========================================================

def _tool_env(name: str, default: str | None = None) -> str | None:
    """Read tool configuration through the agent's shared environment helper."""
    from agent.graph import get_env

    return get_env(name, default)


def _get_json(
    feature: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
) -> tuple[Any | None, str | None]:
    """Fetch and decode a JSON API response with specific failure messages."""
    try:
        response = httpx.get(url, params=params, timeout=10)
    except httpx.TimeoutException:
        return None, f"The {feature} request timed out. Please try again."
    except httpx.RequestError as exc:
        return None, f"Could not connect to {feature} ({type(exc).__name__})."

    if response.status_code != 200:
        detail = ""
        if feature == "cricket scores":
            try:
                error_payload = response.json()
            except (ValueError, TypeError):
                error_payload = None
            if isinstance(error_payload, dict):
                detail = str(error_payload.get("message", ""))[:160]
        suffix = f": {detail}" if detail else ""
        return None, f"The {feature} service returned HTTP {response.status_code}{suffix}."
    try:
        return response.json(), None
    except (ValueError, TypeError):
        return None, f"The {feature} service returned invalid JSON."


def _weather_description(code: Any) -> str:
    if not isinstance(code, int):
        return "unknown conditions"
    descriptions = {
        0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
        45: "fog", 48: "rime fog", 51: "light drizzle", 53: "drizzle",
        55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
        71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
        81: "heavy rain showers", 82: "violent rain showers",
        95: "thunderstorm", 96: "thunderstorm with hail",
        99: "heavy thunderstorm with hail",
    }
    return descriptions.get(code, "unknown conditions")


@tool
def get_weather(city: str) -> str:
    """Use this tool for current conditions and a 3-day forecast; prefer it over web_search for live weather."""
    city = city.strip() if isinstance(city, str) else ""
    if not city:
        return "Weather lookup needs a city name."
    if len(city) > 100:
        return "City name is too long for weather lookup (maximum 100 characters)."

    emit_tool_event("Looking up city", event="tool_started", tool="get_weather")
    locations, error = _get_json(
        "city lookup",
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
    )
    if error:
        return f"Weather lookup failed: {error}"
    if not isinstance(locations, dict) or not isinstance(locations.get("results"), list):
        return "Weather lookup failed: the city service returned unexpected data."
    results = locations["results"]
    if not results:
        return f"I couldn't find a city named {city[:100]}."
    location = results[0]
    if not isinstance(location, dict):
        return "Weather lookup failed: the city service returned an invalid location."
    latitude, longitude = location.get("latitude"), location.get("longitude")
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        return "Weather lookup failed: the city service omitted its coordinates."

    emit_tool_event("Getting weather forecast", event="tool_started", tool="get_weather")
    forecast, error = _get_json(
        "weather forecast",
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,weather_code",
            "timezone": "auto",
            "forecast_days": 3,
        },
    )
    if error:
        return f"Weather lookup failed: {error}"
    if not isinstance(forecast, dict):
        return "Weather lookup failed: the forecast service returned unexpected data."
    current, daily = forecast.get("current"), forecast.get("daily")
    if not isinstance(current, dict) or not isinstance(daily, dict):
        return "Weather lookup failed: the forecast omitted current or daily conditions."
    try:
        temperature = float(current["temperature_2m"])
        humidity = int(current["relative_humidity_2m"])
        wind = float(current["wind_speed_10m"])
        current_code = current["weather_code"]
        current_desc = _weather_description(current_code)
        dates = daily["time"]
        highs, lows, codes = (
            daily["temperature_2m_max"],
            daily["temperature_2m_min"],
            daily["weather_code"],
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return "Weather lookup failed: the forecast omitted required weather values."
    if (
        not math.isfinite(temperature)
        or not math.isfinite(wind)
        or not 0 <= humidity <= 100
        or not isinstance(current_code, int)
    ):
        return "Weather lookup failed: the forecast returned invalid current conditions."
    if not all(isinstance(values, list) for values in (dates, highs, lows, codes)):
        return "Weather lookup failed: the daily forecast has an unexpected format."
    if min(len(dates), len(highs), len(lows), len(codes)) < 3:
        return "Weather lookup failed: the service returned fewer than three forecast days."
    forecast_lines = []
    for day, high, low, code in zip(dates[:3], highs[:3], lows[:3], codes[:3]):
        if (
            not isinstance(day, str)
            or not isinstance(high, (int, float))
            or not isinstance(low, (int, float))
            or not math.isfinite(high)
            or not math.isfinite(low)
            or not isinstance(code, int)
        ):
            return "Weather lookup failed: the daily forecast contains invalid values."
        forecast_lines.append(
            f"{day}: {_weather_description(code)}, {float(low):.0f}-{float(high):.0f}°C"
        )
    city_name = location.get("name", city)
    return (
        f"{city_name}: {temperature:.1f}°C, {current_desc}, humidity {humidity}%, "
        f"wind {wind:.1f} km/h. Next 3 days: {'; '.join(forecast_lines)}"
    )[:600]


@tool
def get_prayer_times(city: str, country: str = "Pakistan") -> str:
    """Use this tool for today's prayer schedule; prefer it over web_search for prayer times."""
    if not isinstance(city, str) or not city.strip():
        return "Prayer-time lookup needs a city name."
    if not isinstance(country, str) or not country.strip():
        return "Prayer-time lookup needs a country name."
    if len(city) > 100 or len(country) > 100:
        return "City and country names must be 100 characters or fewer."

    emit_tool_event("Looking up prayer times", event="tool_started", tool="get_prayer_times")
    payload, error = _get_json(
        "prayer-time lookup",
        "https://api.aladhan.com/v1/timingsByCity",
        params={"city": city.strip(), "country": country.strip(), "method": 1},
    )
    if error:
        if "HTTP 400" in error or "HTTP 404" in error:
            return f"Could not find prayer times for {city.strip()}, {country.strip()}; check the spelling."
        return f"Prayer-time lookup failed: {error}"
    data = payload.get("data") if isinstance(payload, dict) else None
    timings = data.get("timings") if isinstance(data, dict) else None
    if not isinstance(timings, dict):
        return (
            f"Could not find prayer times for {city.strip()}, {country.strip()}; "
            "check the spelling or try another city."
        )
    names = ("Fajr", "Dhuhr", "Asr", "Maghrib", "Isha")
    if any(
        not isinstance(timings.get(name), str) or len(timings[name]) > 20
        for name in names
    ):
        return "Prayer-time lookup failed: the service returned an incomplete prayer schedule."
    return ", ".join(f"{name} {timings[name]}" for name in names)[:600]


@tool
def convert_currency(amount: float, from_code: str, to_code: str) -> str:
    """Use this tool for an exchange-rate conversion; prefer it over web_search for current rates."""
    if not isinstance(amount, (int, float)) or isinstance(amount, bool) or not math.isfinite(amount) or amount < 0:
        return "Currency conversion needs a finite, non-negative amount."
    source = from_code.strip().upper() if isinstance(from_code, str) else ""
    target = to_code.strip().upper() if isinstance(to_code, str) else ""
    if not re.fullmatch(r"[A-Z]{3}", source) or not re.fullmatch(r"[A-Z]{3}", target):
        return "Use three-letter currency codes, such as USD, EUR, or PKR."

    emit_tool_event("Getting exchange rate", event="tool_started", tool="convert_currency")
    payload, error = _get_json(
        "currency conversion",
        f"https://open.er-api.com/v6/latest/{source}",
    )
    if error:
        return f"Currency conversion failed: {error}"
    if not isinstance(payload, dict):
        return "Currency conversion failed: the exchange-rate service returned unexpected data."
    rates = payload.get("rates")
    if payload.get("result") != "success" or not isinstance(rates, dict):
        return f"The exchange-rate service doesn't recognize currency code {source}."
    rate = rates.get(target)
    if not isinstance(rate, (int, float)) or not math.isfinite(rate):
        return f"The exchange-rate service doesn't recognize currency code {target}."
    rate_date = payload.get("time_last_update_utc") or payload.get("time_last_update_unix")
    if not isinstance(rate_date, (str, int, float)):
        return "Currency conversion failed: the service omitted the rate date."
    converted = amount * float(rate)
    if not math.isfinite(converted):
        return "Currency conversion failed: the converted amount is outside the supported range."
    return (
        f"{amount:g} {source} = {converted:,.2f} {target} "
        f"(rate {rate:g}; updated {rate_date})."
    )[:600]


_COINGECKO_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "DOGE": "dogecoin",
    "ADA": "cardano", "XRP": "ripple", "BNB": "binancecoin", "LTC": "litecoin",
    "DOT": "polkadot", "AVAX": "avalanche-2", "MATIC": "matic-network",
    "POL": "polygon-ecosystem-token", "LINK": "chainlink", "SHIB": "shiba-inu",
}


@tool
def get_crypto_price(symbol: str = "BTC") -> str:
    """Use this tool for a live cryptocurrency quote; prefer it over web_search for crypto prices."""
    ticker = symbol.strip().upper() if isinstance(symbol, str) else ""
    coin_id = _COINGECKO_IDS.get(ticker)
    if not coin_id:
        return "Unsupported symbol. Examples: BTC, ETH, SOL, DOGE, ADA, XRP."

    emit_tool_event("Getting cryptocurrency prices", event="tool_started", tool="get_crypto_price")
    payload, error = _get_json(
        "cryptocurrency price lookup",
        "https://api.coingecko.com/api/v3/simple/price",
        params={
            "ids": coin_id,
            "vs_currencies": "usd,pkr",
            "include_24hr_change": "true",
        },
    )
    if error:
        return f"Crypto-price lookup failed: {error}"
    quote_data = payload.get(coin_id) if isinstance(payload, dict) else None
    if not isinstance(quote_data, dict):
        return f"The price service returned no quote for {ticker}."
    try:
        usd, pkr = quote_data["usd"], quote_data["pkr"]
        usd_change = quote_data["usd_24h_change"]
        pkr_change = quote_data["pkr_24h_change"]
    except KeyError:
        return "Crypto-price lookup failed: the service omitted a price or 24-hour change."
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in (usd, pkr, usd_change, pkr_change)
    ):
        return "Crypto-price lookup failed: the service returned invalid price data."
    return (
        f"{ticker}: ${usd:,.2f} ({usd_change:+.2f}% 24h); "
        f"PKR {pkr:,.2f} ({pkr_change:+.2f}% 24h)."
    )[:600]


@tool
def get_cricket_score(team: str = "") -> str:
    """Use this tool for live cricket scores; prefer it over web_search for current matches."""
    api_key = _tool_env("CRICAPI_KEY")
    if not api_key:
        return "Cricket scores are not configured on this server."

    emit_tool_event("Checking current cricket matches", event="tool_started", tool="get_cricket_score")
    payload, error = _get_json(
        "cricket scores",
        "https://api.cricapi.com/v1/currentMatches",
        params={"apikey": api_key, "offset": 0},
    )
    if error:
        if "HTTP 429" in error or (
            "HTTP 403" in error
            and any(word in error.lower() for word in ("daily", "limit", "quota"))
        ):
            return "Cricket score daily limit reached, try tomorrow."
        return f"Cricket score lookup failed: {error}"
    if not isinstance(payload, dict):
        return "Cricket score lookup failed: the service returned unexpected data."
    status_message = f"{payload.get('status', '')} {payload.get('message', '')}".lower()
    if any(term in status_message for term in ("daily limit", "quota", "limit exceeded", "per day")):
        return "Cricket score daily limit reached, try tomorrow."
    if payload.get("status") not in ("success", "ok"):
        message = str(payload.get("message", "the service rejected the request"))[:180]
        return f"Cricket score lookup failed: {message}."
    matches = payload.get("data")
    if not isinstance(matches, list):
        return "Cricket score lookup failed: the service omitted its match list."

    output = []
    normalized_team = team.strip().casefold() if isinstance(team, str) else ""
    for match in matches:
        if not isinstance(match, dict):
            continue
        teams = match.get("teams")
        if not isinstance(teams, list):
            info = match.get("teamInfo")
            teams = [
                item.get("name") for item in info
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            ] if isinstance(info, list) else []
        names = [name for name in teams if isinstance(name, str)]
        if normalized_team and not any(normalized_team in name.casefold() for name in names):
            continue
        if not names:
            continue
        scores = match.get("score")
        score_text = ""
        if isinstance(scores, list):
            score_text = " | ".join(
                f"{item.get('inning', 'Innings')}: {item.get('r', '?')}/{item.get('w', '?')} "
                f"({item.get('o', '?')} ov)"
                for item in scores[:2] if isinstance(item, dict)
            )
        status = match.get("status")
        if not isinstance(status, str) or not status:
            return "Cricket score lookup failed: a match is missing its status."
        output.append(f"{' vs '.join(names)}: {score_text or 'score pending'} — {status}")
        if len(output) == 3:
            break
    if not output:
        suffix = f" for {team.strip()}" if normalized_team else ""
        return f"No current cricket matches found{suffix}."
    return "\n".join(output)[:600]


@tool
def get_movie_info(title: str) -> str:
    """Use this tool for movie facts and a short plot; prefer it over web_search for film metadata."""
    query = title.strip() if isinstance(title, str) else ""
    if not query:
        return "Movie lookup needs a title."
    if len(query) > 200:
        return "Movie title is too long (maximum 200 characters)."
    api_key = _tool_env("OMDB_API_KEY")
    if not api_key:
        return "Movie information is not configured on this server."

    emit_tool_event("Looking up movie", event="tool_started", tool="get_movie_info")
    payload, error = _get_json(
        "movie lookup",
        "https://www.omdbapi.com/",
        params={"apikey": api_key, "t": query},
    )
    if error:
        return f"Movie lookup failed: {error}"
    if not isinstance(payload, dict):
        return "Movie lookup failed: the service returned unexpected data."
    if payload.get("Response") == "False":
        if payload.get("Error") == "Movie not found!":
            return f"Movie not found: {query[:120]}."
        return f"Movie lookup failed: {str(payload.get('Error', 'unknown API error'))[:180]}"
    if payload.get("Response") != "True":
        return "Movie lookup failed: the service omitted its response status."
    fields = ("Title", "Year", "Runtime", "Genre", "imdbRating", "Plot")
    if any(not isinstance(payload.get(field), str) or not payload[field].strip() for field in fields):
        return "Movie lookup failed: the service omitted title, year, runtime, genre, rating, or plot."
    plot = payload["Plot"].strip()
    if len(plot) > 220:
        plot = plot[:217].rstrip() + "..."
    return (
        f"{payload['Title']} ({payload['Year']}) — {payload['Runtime']}, {payload['Genre']}; "
        f"IMDb {payload['imdbRating']}. {plot}"
    )[:600]


@tool
def get_datetime(
    timezone_name: str = "Asia/Karachi",
    convert_from: str | None = None,
) -> str:
    """Use this tool for date/time lookup; prefer it over web_search for timezone conversion and current time."""
    if not isinstance(timezone_name, str) or len(timezone_name) > 100:
        return "Timezone name must be a valid IANA timezone."
    try:
        target_zone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, TypeError):
        return f"Unknown timezone: {timezone_name}."
    instant = datetime.now(UTC)
    target_time = instant.astimezone(target_zone)
    if convert_from is None:
        return f"{target_time.strftime('%Y-%m-%d %H:%M:%S %Z')} ({timezone_name})."
    if not isinstance(convert_from, str) or len(convert_from) > 100:
        return "Source timezone name must be a valid IANA timezone."
    try:
        source_zone = ZoneInfo(convert_from)
    except (ZoneInfoNotFoundError, TypeError):
        return f"Unknown timezone: {convert_from}."
    source_time = instant.astimezone(source_zone)
    return (
        f"{source_time.strftime('%Y-%m-%d %H:%M:%S %Z')} ({convert_from}) = "
        f"{target_time.strftime('%Y-%m-%d %H:%M:%S %Z')} ({timezone_name})."
    )[:600]


@tool
def generate_qr_code(text: str) -> str:
    """Use this tool to create a QR code image for text, a URL, or contact details."""
    content = text.strip() if isinstance(text, str) else ""
    if not content:
        return "QR-code generation needs non-empty text."
    if len(content) > 2000:
        return "QR-code text is too long (maximum 2000 characters)."
    user_id = active_user_id.get()
    if not user_id:
        return "Sign in before saving a generated QR code."

    emit_tool_event("Generating QR code", event="tool_started", tool="generate_qr_code")
    saved_path: Path | None = None
    try:
        response = httpx.get(
            "https://api.qrserver.com/v1/create-qr-code/",
            params={"size": "300x300", "data": content},
            timeout=10,
        )
        if response.status_code != 200:
            raise ValueError(f"QR-code service returned HTTP {response.status_code}.")
        if not response.headers.get("content-type", "").lower().startswith("image/png"):
            raise ValueError("QR-code service returned data that was not a PNG image.")
        if not response.content:
            raise ValueError("QR-code service returned an empty image.")

        filename = f"{uuid.uuid4().hex[:16]}.png"
        saved_path = get_generated_image_dir(user_id, create=True) / filename
        from io import BytesIO
        from PIL import Image

        with Image.open(BytesIO(response.content)) as image:
            if image.format != "PNG":
                raise ValueError("the returned image is not a valid PNG.")
            image.convert("RGB").save(saved_path, format="PNG")
        if saved_path.stat().st_size == 0:
            raise ValueError("the saved QR image is empty.")
        emit_tool_event("QR code generated", event="tool_completed", tool="generate_qr_code")
        return f"{GENERATED_IMAGE_MARKER}{filename}"
    except httpx.TimeoutException:
        error_message = "The QR-code request timed out. Please try again."
    except httpx.RequestError as exc:
        error_message = f"The QR-code request failed ({type(exc).__name__})."
    except ValueError as exc:
        error_message = str(exc)
    except Exception as exc:
        logger.exception("QR-code generation failed")
        error_message = f"QR-code image could not be validated or saved ({type(exc).__name__})."

    if saved_path is not None:
        try:
            saved_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove incomplete QR image %s", saved_path)
    emit_tool_event("QR code generation failed", event="tool_error", tool="generate_qr_code")
    return error_message


@tool
def generate_password(length: int = 16) -> str:
    """Use this tool to generate a strong random password for the user."""
    if not isinstance(length, int) or isinstance(length, bool):
        return "Password length must be a whole number from 8 to 64."
    if not 8 <= length <= 64:
        return "Password length must be between 8 and 64 characters."
    groups = (
        string.ascii_uppercase,
        string.ascii_lowercase,
        string.digits,
        "!@#$%^&*()-_=+[]{}?",
    )
    password = [secrets.choice(group) for group in groups]
    alphabet = "".join(groups)
    password.extend(secrets.choice(alphabet) for _ in range(length - len(password)))
    secrets.SystemRandom().shuffle(password)
    return "".join(password)


@tool
def define_word(word: str) -> str:
    """Use this tool for dictionary definitions and synonyms; prefer it over web_search for English meanings."""
    query = word.strip() if isinstance(word, str) else ""
    if not query:
        return "Dictionary lookup needs a word."
    if len(query) > 100:
        return "Word is too long for dictionary lookup (maximum 100 characters)."

    emit_tool_event("Looking up word", event="tool_started", tool="define_word")
    payload, error = _get_json(
        "dictionary lookup",
        f"https://api.dictionaryapi.dev/api/v2/entries/en/{quote(query, safe='')}",
    )
    if error:
        if "HTTP 404" in error:
            return f"No dictionary definition found for '{query}'."
        return f"Dictionary lookup failed: {error}"
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        return f"No dictionary definition found for '{query}'."
    meanings = payload[0].get("meanings")
    if not isinstance(meanings, list) or not meanings or not isinstance(meanings[0], dict):
        return "Dictionary lookup failed: the service returned no usable meanings."
    part = meanings[0].get("partOfSpeech")
    definitions = meanings[0].get("definitions")
    if not isinstance(part, str) or not isinstance(definitions, list) or not definitions:
        return "Dictionary lookup failed: the service omitted a part of speech or definition."
    definition_texts = [
        item.get("definition")
        for item in definitions[:2]
        if isinstance(item, dict) and isinstance(item.get("definition"), str)
    ]
    if not definition_texts:
        return "Dictionary lookup failed: the service returned no valid definitions."
    synonyms = []
    for meaning in meanings:
        if not isinstance(meaning, dict) or not isinstance(meaning.get("synonyms"), list):
            continue
        for synonym in meaning["synonyms"]:
            if isinstance(synonym, str) and synonym not in synonyms:
                synonyms.append(synonym)
            if len(synonyms) == 3:
                break
        if len(synonyms) == 3:
            break
    result = f"{query} ({part}): {'; '.join(definition_texts)}"
    if synonyms:
        result += f" Synonyms: {', '.join(synonyms)}."
    return result[:600]


@tool
def get_cat_fact() -> str:
    """Use this tool for cat facts; prefer it over web_search for a quick, verified fact."""
    emit_tool_event("Finding a cat fact", event="tool_started", tool="get_cat_fact")
    payload, error = _get_json("cat-fact lookup", "https://catfact.ninja/fact")
    if error:
        return f"Cat fact lookup failed: {error}"
    fact = payload.get("fact") if isinstance(payload, dict) else None
    if not isinstance(fact, str) or not fact.strip():
        return "Cat fact lookup failed: the service returned no fact."
    return fact.strip()[:600]


def _reserve_email_send(user_id: str, daily_limit: int) -> tuple[str | None, str | None]:
    """Atomically reserve a daily email slot so concurrent requests respect the cap."""
    from api.store import get_ui_connection

    reservation_id = uuid.uuid4().hex
    day = datetime.now(UTC).date().isoformat()
    connection = get_ui_connection()
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS email_send_log (
                reservation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                sent_date TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_email_send_user_day "
            "ON email_send_log(user_id, sent_date)"
        )
        connection.execute("BEGIN IMMEDIATE")
        count = connection.execute(
            "SELECT COUNT(*) FROM email_send_log WHERE user_id = ? AND sent_date = ?",
            (user_id, day),
        ).fetchone()[0]
        if count >= daily_limit:
            connection.rollback()
            return None, "daily email limit reached."
        connection.execute(
            "INSERT INTO email_send_log(reservation_id, user_id, sent_date) VALUES (?, ?, ?)",
            (reservation_id, user_id, day),
        )
        connection.commit()
        return reservation_id, None
    except Exception:
        connection.rollback()
        logger.exception("Could not reserve daily email allowance")
        return None, "Email sending is temporarily unavailable because the daily limit could not be checked."
    finally:
        connection.close()


def _release_email_reservation(reservation_id: str) -> None:
    from api.store import get_ui_connection

    connection = get_ui_connection()
    try:
        connection.execute("DELETE FROM email_send_log WHERE reservation_id = ?", (reservation_id,))
        connection.commit()
    except Exception:
        connection.rollback()
        logger.exception("Could not release failed email reservation")
    finally:
        connection.close()


@tool
def send_email(subject: str, body: str) -> str:
    """Use this tool to email a concise note to the authenticated user's own verified address."""
    clean_subject = subject.strip() if isinstance(subject, str) else ""
    if not clean_subject:
        return "Email subject cannot be empty."
    if len(clean_subject) > 200:
        return "Email subject is too long (maximum 200 characters)."
    clean_body = body[:2000] if isinstance(body, str) else ""
    if not clean_body.strip():
        return "Email body cannot be empty."
    api_key = _tool_env("RESEND_API_KEY")
    if not api_key:
        return "Email sending is not configured on this server."

    user_id = active_user_id.get()
    if not user_id:
        return "Sign in with a verified Google account before sending email."
    from api.store import get_ui_connection

    connection = get_ui_connection()
    try:
        user = connection.execute(
            "SELECT email, google_id FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    except Exception:
        logger.exception("Could not look up authenticated email recipient")
        return "Email sending is temporarily unavailable because your account could not be checked."
    finally:
        connection.close()
    if not user or not user[0] or not user[1]:
        return "Email sending requires a verified Google email address on your account."
    recipient = str(user[0]).strip()
    if not recipient:
        return "Email sending requires a verified Google email address on your account."

    raw_limit = _tool_env("EMAIL_DAILY_LIMIT", "5") or "5"
    try:
        daily_limit = int(raw_limit)
        if daily_limit < 1:
            raise ValueError
    except ValueError:
        return "Email sending is not configured: EMAIL_DAILY_LIMIT must be a positive whole number."
    reservation_id, error = _reserve_email_send(user_id, daily_limit)
    if error:
        return error
    assert reservation_id is not None

    emit_tool_event("Sending email to your verified address", event="tool_started", tool="send_email")
    try:
        response = httpx.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "from": "Cat GPT <onboarding@resend.dev>",
                "to": [recipient],
                "subject": clean_subject,
                "text": clean_body,
            },
            timeout=10,
        )
    except httpx.TimeoutException:
        _release_email_reservation(reservation_id)
        return "Email request timed out; please try again."
    except httpx.RequestError as exc:
        _release_email_reservation(reservation_id)
        return f"Email request failed ({type(exc).__name__}); please try again."
    if response.status_code == 429:
        _release_email_reservation(reservation_id)
        return "The email provider rate limit was reached; please try again later."
    if not 200 <= response.status_code < 300:
        _release_email_reservation(reservation_id)
        return f"Email provider rejected the request with HTTP {response.status_code}."
    try:
        result = response.json()
    except (ValueError, TypeError):
        return "Email provider accepted the request but returned invalid confirmation data."
    if not isinstance(result, dict) or not isinstance(result.get("id"), str):
        return "Email provider accepted the request but omitted its message ID."
    emit_tool_event("Email sent", event="tool_completed", tool="send_email")
    return f"Email sent to your verified address (message {result['id'][:80]})."