"""
FastAPI backend for the AI Assistant — Phase 1 (backend only).

This module is a thin transport layer. It does not reimplement any agent,
RAG, or voice logic: everything under agent/ and rag/ is imported and used
exactly as it already exists and is exercised by app.py (the Streamlit
frontend), which is left completely untouched and can keep running
independently against the same on-disk data (SQLite checkpoints, FAISS
index, ui.sqlite, uploaded files).

Known limitation (explicit, not accidental): file-ingestion progress
(the per-chunk "embedded 40/180" events from rag/ingest.py) is collected
into a response log rather than streamed live over SSE in this phase.
Live ingestion streaming is a reasonable Phase 2 addition.

Run with:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import math
import base64
import binascii
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from filelock import Timeout as FileLockTimeout
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware

from agent.graph import (
    get_all_saved_threads,
    get_thread_history,
    remove_last_turn,
    stream_agent,
)
from agent.tools import (
    GENERATED_IMAGE_MARKER,
    MAX_IMAGE_PROMPT_LENGTH,
    active_user_id,
    generate_image,
    get_generated_image_dir,
)
from agent.voice import text_to_speech, transcribe_audio
from rag.ingest import ingest_file, progress_context

from api.config import (
    ALLOWED_FILE_EXTENSIONS,
    ALLOWED_IMAGE_EXTENSIONS,
    EMAIL_MAX_LENGTH,
    FRONTEND_ORIGINS,
    MAX_FILE_MB,
    MAX_IMAGE_MB,
    MAX_IMAGES_PER_REQUEST,
    MAX_PROMPT_CHARS,
    MAX_UPLOAD_FILES,
    UPLOAD_DIR,
)
from api.schemas import (
    ArchiveResponse,
    CreateThreadResponse,
    DeleteThreadResponse,
    EditResponse,
    GenerateImageRequest,
    GeneratedImageOut,
    LibraryMediaOut,
    GoogleLoginRequest,
    LoginResponse,
    MeResponse,
    MessageOut,
    PasswordLoginRequest,
    PinResponse,
    RegisterRequest,
    RenameThreadRequest,
    SendMessageRequest,
    SetPasswordRequest,
    SetPasswordResponse,
    SpeakRequest,
    ThreadSummary,
    TranscribeResponse,
    UploadedFileResult,
)
from api.auth import (
    InvalidCredentialsError,
    UserNotFoundError,
    create_session_token,
    login_with_password,
    register_with_password,
    set_user_password,
    verify_google_credential,
    verify_session_token,
)
from api.rate_limit import (
    LOGIN_EMAIL_LIMITER,
    LOGIN_IP_LIMITER,
    REGISTER_IP_LIMITER,
    AttemptLimiter,
)
from api.store import (
    AccountExistsError,
    archive_thread as store_archive_thread,
    delete_thread as store_delete_thread,
    get_archived_threads,
    get_deleted_threads,
    get_pinned_threads,
    get_renamed_titles,
    get_user_by_id,
    initialize_ui_database,
    pin_thread as store_pin_thread,
    rename_thread as store_rename_thread,
    unarchive_thread as store_unarchive_thread,
    unpin_thread as store_unpin_thread,
)
from api.streaming import (
    extract_activity_message,
    extract_generated_image_filename,
    extract_updates_text,
    format_sse,
)


# ============================================================
# APP SETUP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Mirrors app.py's main(): safe to call repeatedly, migrates an
    # older ui.sqlite if needed.
    try:
        initialize_ui_database()
    except Exception as exc:
        raise RuntimeError(
            f"The app database could not be initialized: {exc}"
        ) from exc

    yield


app = FastAPI(title="AI Assistant API", lifespan=lifespan)


# ============================================================
# AUTH MIDDLEWARE
#
# Every route requires a valid session token, via an "Authorization:
# Bearer <token>" header, except the small set of paths explicitly
# exempted below. This is a middleware (not a per-route Depends())
# so every existing endpoint stays protected without touching each
# route's own signature or body.
#
# Generated images are authenticated and resolved inside the current
# user's image directory; the frontend loads them with bearer auth.
# ============================================================

_PUBLIC_PATHS = {
    "/health",
    "/auth/google",
    "/auth/register",
    "/auth/login",
    "/docs",
    "/openapi.json",
    "/redoc",
}

_PUBLIC_PREFIXES: tuple[str, ...] = ()


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):

        # CORS preflight requests carry no Authorization header and
        # must be allowed through, or the browser's actual request
        # never gets sent at all.
        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path

        if path in _PUBLIC_PATHS or path.startswith(_PUBLIC_PREFIXES):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                {"detail": "Not authenticated."},
                status_code=401,
            )

        token = auth_header[len("Bearer "):].strip()
        user_id = verify_session_token(token)

        if not user_id:
            return JSONResponse(
                {"detail": "Session expired or invalid. Please log in again."},
                status_code=401,
            )

        request.state.user_id = user_id
        return await call_next(request)


# ORDER MATTERS: Starlette treats the LAST middleware added as the
# OUTERMOST one. CORS must be outermost so that responses produced by
# AuthMiddleware (401s) also get CORS headers. If auth were outermost,
# the browser would block every 401 as a CORS error and the frontend
# would report "backend not working" instead of "please log in".
app.add_middleware(AuthMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HELPERS
# ============================================================

def _safe_path_part(value: str) -> str:
    """Make a user-supplied id safe to use as ONE folder name."""

    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", value or "").strip("_")[:100]

    return cleaned or "unknown"


def _user_upload_dir(user_id: str) -> Path:
    return UPLOAD_DIR / _safe_path_part(user_id)


def _is_user_upload_path(path_text: str, user_id: str) -> bool:
    """
    True only when path_text is an existing file inside THIS user's
    upload folder. Stops a client from pointing the agent/RAG at an
    arbitrary file on the server (or at another user's upload).
    """

    try:
        resolved = Path(path_text).resolve()
        base = _user_upload_dir(user_id).resolve()

        return resolved.is_file() and resolved.is_relative_to(base)

    except (OSError, ValueError, TypeError):
        return False



def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _raise_if_limited(*checks: tuple[AttemptLimiter, str]) -> None:
    """Raise HTTP 429 when any (limiter, key) pair is over its limit."""

    retry_after = max(
        (limiter.retry_after(key) for limiter, key in checks),
        default=0,
    )

    if retry_after > 0:
        minutes = max(1, math.ceil(retry_after / 60))

        raise HTTPException(
            status_code=429,
            detail=(
                "Too many attempts. Please try again in about "
                f"{minutes} minute{'s' if minutes != 1 else ''}."
            ),
            headers={"Retry-After": str(retry_after)},
        )


# ============================================================
# AUTH
# ============================================================

@app.get("/auth/me", response_model=MeResponse)
def current_user(request: Request) -> MeResponse:
    user = get_user_by_id(request.state.user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")

    return MeResponse(
        email=user["email"],
        name=user["name"],
        has_password=bool(user.get("password_hash")),
    )


@app.post("/auth/google", response_model=LoginResponse)
def google_login(
    request_body: GoogleLoginRequest,
    request: Request,
) -> LoginResponse:

    ip_key = _client_ip(request)
    _raise_if_limited((LOGIN_IP_LIMITER, ip_key))

    try:
        user = verify_google_credential(request_body.credential)
    except ValueError as exc:
        LOGIN_IP_LIMITER.record(ip_key)
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    LOGIN_IP_LIMITER.reset(ip_key)

    token = create_session_token(user["user_id"])

    return LoginResponse(
        token=token,
        email=user["email"],
        name=user["name"],
        has_password=bool(user.get("password_hash")),
    )


@app.post("/auth/register", response_model=LoginResponse)
def password_register(
    request_body: RegisterRequest,
    request: Request,
) -> LoginResponse:
    """Create an email + password account and sign it in."""

    ip_key = _client_ip(request)

    _raise_if_limited((REGISTER_IP_LIMITER, ip_key))

    try:
        user = register_with_password(
            request_body.email,
            request_body.password,
            request_body.name,
        )
    except AccountExistsError as exc:
        # Counts toward the limit so this cannot be used to probe
        # which emails are registered.
        REGISTER_IP_LIMITER.record(ip_key)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    REGISTER_IP_LIMITER.record(ip_key)

    token = create_session_token(user["user_id"])

    return LoginResponse(
        token=token,
        email=user["email"],
        name=user["name"],
        has_password=True,
    )


@app.post("/auth/login", response_model=LoginResponse)
def password_login(
    request_body: PasswordLoginRequest,
    request: Request,
) -> LoginResponse:
    """Sign in with email + password."""

    ip_key = _client_ip(request)
    email_key = f"email:{(request_body.email or '').strip().lower()[:EMAIL_MAX_LENGTH]}"

    _raise_if_limited(
        (LOGIN_EMAIL_LIMITER, email_key),
        (LOGIN_IP_LIMITER, ip_key),
    )

    try:
        user = login_with_password(request_body.email, request_body.password)
    except InvalidCredentialsError as exc:
        LOGIN_EMAIL_LIMITER.record(email_key)
        LOGIN_IP_LIMITER.record(ip_key)
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    LOGIN_EMAIL_LIMITER.reset(email_key)

    token = create_session_token(user["user_id"])

    return LoginResponse(
        token=token,
        email=user["email"],
        name=user["name"],
        has_password=bool(user.get("password_hash")),
    )


@app.post("/auth/set-password", response_model=SetPasswordResponse)
def set_password(
    request_body: SetPasswordRequest,
    request: Request,
) -> SetPasswordResponse:
    """
    Add (or change) a password for the signed-in account. Requires a
    valid session token like every other route; an account that
    already has a password must also send current_password.
    """

    user_id = request.state.user_id
    limit_key = f"user:{user_id}"

    _raise_if_limited((LOGIN_EMAIL_LIMITER, limit_key))

    try:
        set_user_password(
            user_id,
            request_body.new_password,
            request_body.current_password,
        )
    except UserNotFoundError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except InvalidCredentialsError as exc:
        # 403, not 401: the session is fine, only the current password
        # is wrong, so the frontend must not log the person out.
        LOGIN_EMAIL_LIMITER.record(limit_key)
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    LOGIN_EMAIL_LIMITER.reset(limit_key)

    return SetPasswordResponse(password_set=True)


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ============================================================
# THREADS
# ============================================================

@app.get("/generated-images/{filename}")
def get_generated_image(filename: str, request: Request) -> FileResponse:
    """
    Serves an image generate_image() (agent/tools.py) saved to
    Only serves an image from the authenticated user's directory, and
    validates the filename to prevent path traversal.
    """

    safe_name = Path(filename).name

    if safe_name != filename or not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if not re.fullmatch(r"[a-f0-9]{16}\.png", safe_name):
        raise HTTPException(status_code=400, detail="Invalid image filename.")

    file_path = get_generated_image_dir(request.state.user_id) / safe_name

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")

    return FileResponse(file_path)


@app.get("/images", response_model=list[GeneratedImageOut])
def list_generated_images(request: Request) -> list[GeneratedImageOut]:
    """List only generated images belonging to the signed-in account."""
    image_dir = get_generated_image_dir(request.state.user_id)
    if not image_dir.exists():
        return []

    try:
        files = sorted(
            (
                path for path in image_dir.iterdir()
                if path.is_file() and re.fullmatch(r"[a-f0-9]{16}\.png", path.name)
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return [
            GeneratedImageOut(
                filename=path.name,
                url=f"/generated-images/{path.name}",
                created_at=path.stat().st_mtime,
            )
            for path in files
        ]
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not load your image library: {exc}"
        ) from exc


@app.get("/library/media", response_model=list[LibraryMediaOut])
def list_library_media(request: Request) -> list[LibraryMediaOut]:
    """List generated images and uploaded media grouped by chat folder."""
    user_id = request.state.user_id
    media: list[LibraryMediaOut] = []
    generated_dir = get_generated_image_dir(user_id)
    upload_dir = _user_upload_dir(user_id)

    try:
        if generated_dir.exists():
            for path in generated_dir.iterdir():
                if path.is_file() and re.fullmatch(r"[a-f0-9]{16}\.png", path.name):
                    media.append(LibraryMediaOut(
                        filename=path.name,
                        url=f"/generated-images/{path.name}",
                        created_at=path.stat().st_mtime,
                        kind="image",
                        folder="Generated",
                    ))
        if upload_dir.exists():
            for folder in upload_dir.iterdir():
                if not folder.is_dir():
                    continue
                for path in folder.iterdir():
                    if path.is_file():
                        suffix = path.suffix.lower()
                        if suffix not in ALLOWED_FILE_EXTENSIONS | ALLOWED_IMAGE_EXTENSIONS:
                            continue
                        media.append(LibraryMediaOut(
                            filename=path.name,
                            url=f"/library/media/{folder.name}/{path.name}",
                            created_at=path.stat().st_mtime,
                            kind="image" if suffix in ALLOWED_IMAGE_EXTENSIONS else "file",
                            folder=folder.name,
                        ))
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not load your library: {exc}"
        ) from exc

    return sorted(media, key=lambda item: item.created_at, reverse=True)


@app.get("/library/media/{folder}/{filename}")
def get_library_media(folder: str, filename: str, request: Request) -> FileResponse:
    if _safe_path_part(folder) != folder or Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid media path.")
    path = _user_upload_dir(request.state.user_id) / folder / filename
    if not path.is_file() or path.suffix.lower() not in (
        ALLOWED_FILE_EXTENSIONS | ALLOWED_IMAGE_EXTENSIONS
    ):
        raise HTTPException(status_code=404, detail="Media not found.")
    return FileResponse(path)


@app.post("/images/generate", response_model=GeneratedImageOut)
def generate_account_image(
    request_body: GenerateImageRequest,
    request: Request,
) -> GeneratedImageOut:
    prompt = request_body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Describe the image you want.")
    if len(prompt) > MAX_IMAGE_PROMPT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt must be at most {MAX_IMAGE_PROMPT_LENGTH} characters.",
        )

    user_token = active_user_id.set(request.state.user_id)
    try:
        result = generate_image.invoke({"prompt": prompt})
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Image generation failed ({type(exc).__name__}): {exc}",
        ) from exc
    finally:
        active_user_id.reset(user_token)

    if not isinstance(result, str) or not result.startswith(GENERATED_IMAGE_MARKER):
        detail = result.removeprefix("Image generation error: ").strip() if isinstance(result, str) else ""
        raise HTTPException(
            status_code=502,
            detail=detail or "Image generation failed. Please try again.",
        )

    filename = result[len(GENERATED_IMAGE_MARKER):].strip()
    if not re.fullmatch(r"[a-f0-9]{16}\.png", filename):
        raise HTTPException(status_code=502, detail="Image generation returned an invalid result.")
    path = get_generated_image_dir(request.state.user_id) / filename
    try:
        created_at = path.stat().st_mtime
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Generated image could not be saved.") from exc

    return GeneratedImageOut(
        filename=filename,
        url=f"/generated-images/{filename}",
        created_at=created_at,
    )


@app.post("/threads", response_model=CreateThreadResponse)
def create_thread() -> CreateThreadResponse:
    """
    Create a new (client-side) thread id.

    A thread only becomes visible in GET /threads once it has at least
    one checkpointed message, exactly like app.py's behavior today —
    there is no separate "thread creation" record on the backend.
    """

    return CreateThreadResponse(thread_id=str(uuid.uuid4()))


@app.get("/threads", response_model=list[ThreadSummary])
def list_threads(request: Request) -> list[ThreadSummary]:
    user_id = request.state.user_id

    try:
        threads = get_all_saved_threads(user_id)
    except Exception:
        threads = {}

    try:
        deleted = get_deleted_threads(user_id)
    except Exception:
        deleted = set()

    try:
        pinned = get_pinned_threads(user_id)
    except Exception:
        pinned = set()

    try:
        renamed = get_renamed_titles(user_id)
    except Exception:
        renamed = {}

    try:
        archived = get_archived_threads(user_id)
    except Exception:
        archived = set()

    return [
        ThreadSummary(
            thread_id=thread_id,
            title=renamed.get(thread_id, title),
            pinned=thread_id in pinned,
            archived=thread_id in archived,
        )
        for thread_id, title in threads.items()
        if thread_id not in deleted
    ]


@app.get("/threads/{thread_id}/messages", response_model=list[MessageOut])
def get_messages(thread_id: str, request: Request) -> list[MessageOut]:
    user_id = request.state.user_id

    if not thread_id.strip():
        raise HTTPException(status_code=400, detail="A valid thread_id is required.")

    try:
        history = get_thread_history(thread_id=thread_id, user_id=user_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load thread history: {exc}") from exc

    return [MessageOut(**message) for message in history]


@app.delete("/threads/{thread_id}", response_model=DeleteThreadResponse)
def delete_thread_endpoint(thread_id: str, request: Request) -> DeleteThreadResponse:
    user_id = request.state.user_id

    if not thread_id.strip():
        raise HTTPException(status_code=400, detail="A valid thread_id is required.")

    # Ownership guard: deleting also removes the stored checkpoint, so
    # only act on threads that belong to this user. Anything else is
    # treated as already gone (idempotent, and reveals nothing about
    # other users' threads).
    try:
        owned_threads = get_all_saved_threads(user_id)
    except Exception:
        owned_threads = {}

    if thread_id not in owned_threads:
        return DeleteThreadResponse(deleted=True, cleanup_warning=None)

    deleted_ok, cleanup_error = store_delete_thread(thread_id, user_id)

    if not deleted_ok:
        raise HTTPException(status_code=500, detail=cleanup_error or "The chat could not be deleted.")

    return DeleteThreadResponse(deleted=True, cleanup_warning=cleanup_error)


@app.post("/threads/{thread_id}/pin", response_model=PinResponse)
def pin_thread_endpoint(thread_id: str, request: Request) -> PinResponse:

    try:
        store_pin_thread(thread_id, request.state.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PinResponse(pinned=True)


@app.post("/threads/{thread_id}/unpin", response_model=PinResponse)
def unpin_thread_endpoint(thread_id: str, request: Request) -> PinResponse:

    try:
        store_unpin_thread(thread_id, request.state.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PinResponse(pinned=False)


@app.post("/threads/{thread_id}/archive", response_model=ArchiveResponse)
def archive_thread_endpoint(thread_id: str, request: Request) -> ArchiveResponse:
    """Archive a thread — hides it from the main list without deleting
    it, same lifecycle as pin/unpin."""

    try:
        store_archive_thread(thread_id, request.state.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ArchiveResponse(archived=True)


@app.post("/threads/{thread_id}/unarchive", response_model=ArchiveResponse)
def unarchive_thread_endpoint(thread_id: str, request: Request) -> ArchiveResponse:

    try:
        store_unarchive_thread(thread_id, request.state.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ArchiveResponse(archived=False)


@app.put("/threads/{thread_id}/rename", response_model=ThreadSummary)
def rename_thread_endpoint(
    thread_id: str,
    request_body: RenameThreadRequest,
    request: Request,
) -> ThreadSummary:
    """
    Set a custom display title for a thread, overriding the
    auto-generated one. Stored in ui.sqlite (renamed_threads), the same
    way pins and deletions are — the LangGraph checkpoint itself is
    never touched.
    """

    user_id = request.state.user_id

    if not thread_id.strip():
        raise HTTPException(status_code=400, detail="A valid thread_id is required.")

    try:
        store_rename_thread(thread_id, user_id, request_body.title)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        pinned = get_pinned_threads(user_id)
    except Exception:
        pinned = set()

    try:
        archived = get_archived_threads(user_id)
    except Exception:
        archived = set()

    return ThreadSummary(
        thread_id=thread_id,
        title=request_body.title.strip(),
        pinned=thread_id in pinned,
        archived=thread_id in archived,
    )


@app.post("/threads/{thread_id}/edit", response_model=EditResponse)
def edit_last_message(thread_id: str, request: Request) -> EditResponse:
    """
    Remove the most recent user message (and everything after it) from
    the checkpoint, so the frontend can resend edited text as a fresh
    message via POST /threads/{thread_id}/messages/stream.

    This wraps agent.graph.remove_last_turn() directly — same tail-only
    edit semantics as app.py's edit feature (see graph.py's docstring on
    remove_last_turn for why only the last turn is ever touched).
    """

    if not thread_id.strip():
        raise HTTPException(status_code=400, detail="A valid thread_id is required.")

    removed = remove_last_turn(
        thread_id=thread_id,
        user_id=request.state.user_id,
    )

    return EditResponse(removed=removed)


@app.get("/threads/{thread_id}/export")
def export_thread(thread_id: str, request: Request) -> Response:

    if not thread_id.strip():
        raise HTTPException(status_code=400, detail="A valid thread_id is required.")

    try:
        history = get_thread_history(
            thread_id=thread_id,
            user_id=request.state.user_id,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not load thread history: {exc}"
        ) from exc

    lines = ["# AI Assistant conversation", ""]

    if not history:
        lines.append("_This conversation has no messages yet._")

    for message in history:
        content = (message.get("content") or "").strip()

        if not content:
            continue

        speaker = "You" if message.get("role") == "user" else "Assistant"
        lines.extend([f"**{speaker}:**", "", content, ""])

    transcript = "\n".join(lines).strip() + "\n"

    return Response(
        content=transcript,
        media_type="text/markdown",
        headers={
            "Content-Disposition": (
                f'attachment; filename="chat_{_safe_path_part(thread_id)[:8]}.md"'
            )
        },
    )


# ============================================================
# CHAT STREAMING (SSE)
# ============================================================

@app.post("/threads/{thread_id}/messages/stream")
def send_message_stream(
    thread_id: str,
    request_body: SendMessageRequest,
    request: Request,
) -> StreamingResponse:
    """
    Send a message and stream the reply as Server-Sent Events.

    Event types:
      "text"     — {"text": "..."} incremental/final assistant text,
                   from the "updates" stream_mode (same payload shape
                   app.py's extract_streamed_ai_text already parses).
      "image"    — {"url": "/generated-images/<file>"} an image made
                   by the generate_image tool.
      "activity" — {"message": "..."} live tool/ingestion activity,
                   from the "custom" stream_mode (agent.tools'/
                   rag.ingest's emit_*_event calls).
      "done"     — {} the turn finished normally.
      "error"    — {"message": "..."} something failed; no more events
                   follow.
    """

    user_id = request.state.user_id

    if not thread_id.strip():
        raise HTTPException(status_code=400, detail="A valid thread_id is required.")

    content = (request_body.content or "").strip()

    if len(content) > MAX_PROMPT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Message is too long ({len(content)} characters). "
                f"Please keep it under {MAX_PROMPT_CHARS} characters."
            ),
        )

    if len(request_body.images) > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=(
                f"You can send at most {MAX_IMAGES_PER_REQUEST} images per "
                "request."
            ),
        )

    if not content and not request_body.images and not request_body.files:
        raise HTTPException(
            status_code=400,
            detail="Provide a message, an image, or a file.",
        )

    # Re-validate that any referenced uploaded files exist AND belong to
    # this user's upload folder. This blocks a client from passing an
    # arbitrary server path (for example a system file) as an
    # "uploaded" file, and lets us return a clean 400 instead of an SSE
    # "error" event mid-stream.
    for file_payload in request_body.files:
        if not _is_user_upload_path(file_payload.file_path, user_id):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Uploaded file '{file_payload.filename}' could not be "
                    "found. Please re-upload it."
                ),
            )

    agent_message: dict[str, Any] = {
        "role": "user",
        "content": content,
    }

    if request_body.images:
        image_upload_dir = _user_upload_dir(user_id) / _safe_path_part(thread_id)
        try:
            image_upload_dir.mkdir(parents=True, exist_ok=True)
            for image in request_body.images:
                max_encoded_length = 4 * math.ceil(
                    MAX_IMAGE_MB * 1024 * 1024 / 3
                )
                if len(image.image_bytes) > max_encoded_length:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Image '{image.filename}' exceeds the {MAX_IMAGE_MB:g} MB limit.",
                    )
                try:
                    raw_image = base64.b64decode(image.image_bytes, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Image '{image.filename}' has invalid base64 data.",
                    ) from exc
                if not raw_image:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Image '{image.filename}' is empty.",
                    )
                if len(raw_image) > MAX_IMAGE_MB * 1024 * 1024:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Image '{image.filename}' exceeds the {MAX_IMAGE_MB:g} MB limit.",
                    )
                suffix = Path(image.filename).suffix.lower()
                if suffix not in ALLOWED_IMAGE_EXTENSIONS:
                    suffix = {
                        "image/jpeg": ".jpg",
                        "image/png": ".png",
                        "image/webp": ".webp",
                        "image/gif": ".gif",
                    }.get(image.image_type.lower(), ".png")
                (image_upload_dir / f"{uuid.uuid4().hex[:16]}{suffix}").write_bytes(raw_image)
        except HTTPException:
            raise
        except OSError as exc:
            raise HTTPException(
                status_code=500, detail=f"Could not save uploaded images: {exc}"
            ) from exc

        agent_message["images"] = [
            {
                "image_bytes": image.image_bytes,
                "image_type": image.image_type,
                "filename": image.filename,
            }
            for image in request_body.images
        ]

    if request_body.files:
        agent_message["files"] = [
            {
                "filename": file_payload.filename,
                "file_path": file_payload.file_path,
            }
            for file_payload in request_body.files
        ]

    def event_stream():
        """
        A plain (non-async) generator: agent.graph.stream_agent() makes
        blocking calls (SQLite, Groq, requests), so this intentionally
        avoids `async def` here — Starlette runs a sync generator passed
        to StreamingResponse in a thread pool automatically, keeping the
        event loop free instead of blocking it.
        """

        try:
            for stream_mode, chunk in stream_agent(
                user_id=user_id,
                messages=[agent_message],
                thread_id=thread_id,
            ):

                if stream_mode == "custom":
                    label = extract_activity_message(chunk)
                    if label:
                        yield format_sse("activity", {"message": label})
                    continue

                # stream_mode == "updates"
                image_filename = extract_generated_image_filename(chunk)
                if image_filename:
                    yield format_sse(
                        "image", {"url": f"/generated-images/{image_filename}"}
                    )
                    continue

                text = extract_updates_text(chunk)
                if text:
                    yield format_sse("text", {"text": text})

            yield format_sse("done", {})

        except Exception as exc:
            # Validation errors from parse_input_messages, Groq/network
            # failures not already caught inside assistant_node, etc.
            # Never let an exception crash the HTTP response silently —
            # always tell the client the turn failed.
            yield format_sse(
                "error",
                {"message": f"{type(exc).__name__}: {exc}"},
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# FILE UPLOAD + INGESTION
# ============================================================

def _validate_upload(
    filename: str,
    file_bytes: bytes,
    *,
    allowed_extensions: set[str],
    max_mb: float,
) -> None:

    if not filename:
        raise ValueError("The uploaded file has no filename.")

    suffix = Path(filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise ValueError(f"Unsupported file type: {filename}")

    if not file_bytes:
        raise ValueError(f"File '{filename}' appears to be empty.")

    size_mb = len(file_bytes) / (1024 * 1024)

    if size_mb > max_mb:
        raise ValueError(
            f"File '{filename}' is {size_mb:.1f} MB. "
            f"Maximum allowed is {max_mb:g} MB."
        )


def _save_upload(
    filename: str,
    file_bytes: bytes,
    thread_id: str,
    user_id: str,
) -> Path:
    """
    Saves under UPLOAD_DIR/<user>/<thread>/. Both folder names are
    sanitised, so a crafted thread_id such as "../../x" can never
    write outside the upload directory.
    """

    thread_dir = _user_upload_dir(user_id) / _safe_path_part(thread_id)
    thread_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(filename).name or "uploaded-file"
    saved_path = thread_dir / f"{uuid.uuid4().hex[:12]}_{safe_name}"

    saved_path.write_bytes(file_bytes)

    return saved_path


@app.post("/files/upload", response_model=list[UploadedFileResult])
async def upload_files(
    request: Request,
    thread_id: str = Form(...),
    files: list[UploadFile] = File(...),
) -> list[UploadedFileResult]:
    """
    Save each file, then ingest it into FAISS via rag.ingest.ingest_file()
    exactly as app.py does. Files are processed independently — one
    file's validation/ingestion failure does not block the others, and
    the response reports a per-file result.
    """

    user_id = request.state.user_id

    if not thread_id.strip():
        raise HTTPException(status_code=400, detail="A valid thread_id is required.")

    if not files:
        raise HTTPException(status_code=400, detail="No files were provided.")

    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"You can upload at most {MAX_UPLOAD_FILES} files at once.",
        )

    results: list[UploadedFileResult] = []

    for uploaded_file in files:

        filename = uploaded_file.filename or "uploaded-file"

        try:
            file_bytes = await uploaded_file.read()
        except Exception as exc:
            results.append(
                UploadedFileResult(
                    filename=filename,
                    error=f"Could not read the upload: {exc}",
                )
            )
            continue

        try:
            _validate_upload(
                filename,
                file_bytes,
                allowed_extensions=ALLOWED_FILE_EXTENSIONS,
                max_mb=MAX_FILE_MB,
            )
        except ValueError as exc:
            results.append(UploadedFileResult(filename=filename, error=str(exc)))
            continue

        try:
            saved_path = _save_upload(filename, file_bytes, thread_id, user_id)
        except OSError as exc:
            results.append(
                UploadedFileResult(
                    filename=filename,
                    error=f"Could not save the file: {exc}",
                )
            )
            continue

        # Collect ingestion progress messages for this file. progress_context
        # is a contextvars.ContextVar read by rag.ingest.emit_ingest_event();
        # setting it here lets us capture the same progress log app.py's
        # st.status widget shows, without changing ingest.py at all.
        progress_log: list[str] = []

        def _on_progress(event: dict, _log=progress_log) -> None:
            message = event.get("message")
            if isinstance(message, str):
                _log.append(message)

        progress_token = progress_context.set(_on_progress)

        try:
            chunk_count = await run_in_threadpool(
                ingest_file, str(saved_path), user_id
            )

            results.append(
                UploadedFileResult(
                    filename=filename,
                    file_path=str(saved_path),
                    chunks=chunk_count,
                    progress=progress_log,
                )
            )

        except FileLockTimeout:
            results.append(
                UploadedFileResult(
                    filename=filename,
                    file_path=str(saved_path),
                    error=(
                        "The knowledge base is busy processing another "
                        "file. Please try again in a moment."
                    ),
                    progress=progress_log,
                )
            )

        except (ValueError, FileNotFoundError, PermissionError) as exc:
            results.append(
                UploadedFileResult(
                    filename=filename,
                    file_path=str(saved_path),
                    error=str(exc),
                    progress=progress_log,
                )
            )

        except Exception as exc:
            results.append(
                UploadedFileResult(
                    filename=filename,
                    file_path=str(saved_path),
                    error=f"Unexpected error: {type(exc).__name__}: {exc}",
                    progress=progress_log,
                )
            )

        finally:
            progress_context.reset(progress_token)

    return results


# ============================================================
# VOICE
# ============================================================

@app.post("/voice/transcribe", response_model=TranscribeResponse)
async def voice_transcribe(audio: UploadFile = File(...)) -> TranscribeResponse:

    try:
        audio_bytes = await audio.read()
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not read the recording: {exc}"
        ) from exc

    try:
        text = await run_in_threadpool(transcribe_audio, audio_bytes)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Could not transcribe this recording "
                f"({type(exc).__name__}: {exc})."
            ),
        ) from exc

    return TranscribeResponse(text=text)


@app.post("/voice/speak")
async def voice_speak(request: SpeakRequest) -> Response:

    text = (request.text or "").strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        audio_bytes = await run_in_threadpool(text_to_speech, text)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Could not generate audio "
                f"({type(exc).__name__}: {exc})."
            ),
        ) from exc

    if not audio_bytes:
        raise HTTPException(
            status_code=502, detail="Audio generation returned no data."
        )

    return Response(content=audio_bytes, media_type="audio/mpeg")