from __future__ import annotations

import hashlib
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from agent.graph import (
    CHECKPOINT_DB_PATH,
    create_checkpointer,
    get_all_saved_threads,
    get_thread_history,
    remove_last_turn,
    stream_agent,
)
from agent.voice import (
    text_to_speech,
    transcribe_audio,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

UI_DB_PATH = Path(
    os.getenv(
        "UI_DB_PATH",
        str(DATA_DIR / "ui.sqlite"),
    )
)

UPLOAD_DIR = Path(
    os.getenv(
        "UPLOAD_DIR",
        str(DATA_DIR / "upload"),
    )
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MAX_IMAGE_MB = float(os.getenv("MAX_IMAGE_MB", "10"))
MAX_FILE_MB = float(os.getenv("MAX_FILE_MB", "25"))
MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", "10"))
MAX_IMAGES_PER_REQUEST = int(os.getenv("MAX_IMAGES_PER_REQUEST", "3"))
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "8000"))

# gTTS gets noticeably slower and more failure-prone on very long text, and
# a multi-minute audio clip is not useful for a "listen to this reply"
# button anyway. Text beyond this length is truncated before synthesis.
MAX_TTS_CHARS = int(os.getenv("MAX_TTS_CHARS", "1500"))

ALLOWED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
}

ALLOWED_FILE_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".xlsx",
    ".xls",
}


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# UI DATABASE
# ============================================================

def get_ui_connection():
    """
    Create a connection to the UI metadata database.
    """

    connection = sqlite3.connect(
        str(UI_DB_PATH),
        check_same_thread=False,
    )

    return connection


def initialize_ui_database():
    """
    Create and safely migrate the UI database.

    This supports older versions of ui.sqlite that may not
    contain pinned_at / deleted_at columns.
    """

    connection = get_ui_connection()

    try:

        # ----------------------------------------------------
        # Create pinned_threads table if it doesn't exist
        # ----------------------------------------------------

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS pinned_threads (
                thread_id TEXT PRIMARY KEY,
                pinned_at TEXT
            )
            """
        )

        # ----------------------------------------------------
        # Create deleted_threads table if it doesn't exist
        # ----------------------------------------------------

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS deleted_threads (
                thread_id TEXT PRIMARY KEY,
                deleted_at TEXT
            )
            """
        )

        # ----------------------------------------------------
        # Check existing pinned_threads columns
        # ----------------------------------------------------

        pinned_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(pinned_threads)"
            ).fetchall()
        }

        if "pinned_at" not in pinned_columns:

            connection.execute(
                """
                ALTER TABLE pinned_threads
                ADD COLUMN pinned_at TEXT
                """
            )

        # ----------------------------------------------------
        # Check existing deleted_threads columns
        # ----------------------------------------------------

        deleted_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(deleted_threads)"
            ).fetchall()
        }

        if "deleted_at" not in deleted_columns:

            connection.execute(
                """
                ALTER TABLE deleted_threads
                ADD COLUMN deleted_at TEXT
                """
            )

        # ----------------------------------------------------
        # Give existing pinned rows a timestamp
        # ----------------------------------------------------

        connection.execute(
            """
            UPDATE pinned_threads
            SET pinned_at = COALESCE(
                pinned_at,
                datetime('now')
            )
            WHERE pinned_at IS NULL
            """
        )

        # ----------------------------------------------------
        # Give existing deleted rows a timestamp
        # ----------------------------------------------------

        connection.execute(
            """
            UPDATE deleted_threads
            SET deleted_at = COALESCE(
                deleted_at,
                datetime('now')
            )
            WHERE deleted_at IS NULL
            """
        )

        connection.commit()

    finally:

        connection.close()


# ============================================================
# PINNED CHAT FUNCTIONS
# ============================================================

def pin_thread(
    thread_id: str,
):
    """
    Pin a conversation.
    """

    if not isinstance(thread_id, str) or not thread_id.strip():
        return

    connection = get_ui_connection()

    try:

        connection.execute(
            """
            INSERT OR REPLACE INTO pinned_threads
            (
                thread_id,
                pinned_at
            )
            VALUES (
                ?,
                datetime('now')
            )
            """,
            (thread_id.strip(),),
        )

        connection.commit()

    finally:

        connection.close()


def unpin_thread(
    thread_id: str,
):
    """
    Remove a conversation from pinned chats.
    """

    if not isinstance(thread_id, str) or not thread_id.strip():
        return

    connection = get_ui_connection()

    try:

        connection.execute(
            """
            DELETE FROM pinned_threads
            WHERE thread_id = ?
            """,
            (thread_id.strip(),),
        )

        connection.commit()

    finally:

        connection.close()


def get_pinned_threads() -> set[str]:
    """
    Return pinned thread IDs.

    Uses COALESCE so older migrated rows also work.
    """

    connection = get_ui_connection()

    try:

        rows = connection.execute(
            """
            SELECT thread_id
            FROM pinned_threads
            ORDER BY
                COALESCE(pinned_at, '') DESC,
                rowid DESC
            """
        ).fetchall()

        return {
            row[0]
            for row in rows
        }

    finally:

        connection.close()


# ============================================================
# DELETED CHAT FUNCTIONS
# ============================================================

def mark_thread_deleted(
    thread_id: str,
):
    """
    Mark a thread as deleted from the UI.
    """

    if not isinstance(thread_id, str) or not thread_id.strip():
        return

    connection = get_ui_connection()

    try:

        connection.execute(
            """
            INSERT OR REPLACE INTO deleted_threads
            (
                thread_id,
                deleted_at
            )
            VALUES (
                ?,
                datetime('now')
            )
            """,
            (thread_id.strip(),),
        )

        connection.execute(
            """
            DELETE FROM pinned_threads
            WHERE thread_id = ?
            """,
            (thread_id.strip(),),
        )

        connection.commit()

    finally:

        connection.close()


def get_deleted_threads() -> set[str]:
    """
    Return threads hidden from the UI.
    """

    connection = get_ui_connection()

    try:

        rows = connection.execute(
            """
            SELECT thread_id
            FROM deleted_threads
            """
        ).fetchall()

        return {
            row[0]
            for row in rows
        }

    finally:

        connection.close()


def permanently_delete_checkpoint_thread(
    thread_id: str,
) -> tuple[bool, str | None]:
    """Best-effort physical checkpoint cleanup.

    The UI tombstone remains the source of truth for deletion, so a
    LangGraph/SQLite schema difference cannot make a deleted chat reappear.
    """

    if not thread_id or not CHECKPOINT_DB_PATH.exists():
        return True, None

    connection = None
    cleanup_error = None

    try:
        checkpointer, connection = create_checkpointer()

        delete_method = getattr(checkpointer, "delete_thread", None)

        if callable(delete_method):
            delete_method(thread_id)
            return True, None

    except Exception as exc:
        cleanup_error = str(exc)
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    # Do not depend on LangGraph's private SQLite schema. The UI deletion
    # is already committed and will hide the thread from the application.
    return False, cleanup_error or (
        "The installed checkpointer does not expose delete_thread()."
    )


def delete_thread(
    thread_id: str,
) -> tuple[bool, str | None]:
    """Delete a conversation from the UI and best-effort checkpoint storage."""

    if not isinstance(thread_id, str) or not thread_id.strip():
        return False, "Invalid thread ID."

    # Commit the UI tombstone first. This guarantees that the conversation
    # disappears from Previous/Pinned Chats even if checkpoint cleanup fails.
    try:
        mark_thread_deleted(thread_id.strip())
    except Exception as exc:
        return False, f"Could not mark the chat as deleted: {exc}"

    cleanup_ok, cleanup_error = permanently_delete_checkpoint_thread(
        thread_id.strip()
    )

    return True, cleanup_error if not cleanup_ok else None


# ============================================================
# THREAD HELPERS
# ============================================================

def create_thread_id() -> str:
    """
    Create a unique conversation ID.
    """

    return str(
        uuid.uuid4()
    )


def load_threads() -> dict[str, str]:
    """
    Load saved conversations while excluding deleted chats.
    """

    try:

        threads = get_all_saved_threads()

    except Exception:

        threads = {}

    try:

        deleted = get_deleted_threads()

    except Exception:

        deleted = set()

    return {
        thread_id: title
        for thread_id, title in threads.items()
        if thread_id not in deleted
    }


# ============================================================
# MESSAGE NORMALIZATION
# ============================================================

def normalize_message(
    message: Any,
) -> dict[str, str] | None:
    """
    Convert a LangChain message into the simple format used
    by Streamlit.
    """

    message_type = type(
        message
    ).__name__

    content = getattr(
        message,
        "content",
        "",
    )

    # --------------------------------------------------------
    # Normal string content
    # --------------------------------------------------------

    if isinstance(
        content,
        str,
    ):

        text = content

    # --------------------------------------------------------
    # Multimodal content
    # --------------------------------------------------------

    elif isinstance(
        content,
        list,
    ):

        text_parts = []

        for block in content:

            if not isinstance(
                block,
                dict,
            ):
                continue

            if block.get(
                "type"
            ) == "text":

                value = block.get(
                    "text",
                    "",
                )

                if value:

                    text_parts.append(
                        str(value)
                    )

        text = "\n".join(
            text_parts
        )

    else:

        text = str(
            content
        )

    if not text.strip():
        return None

    if message_type == "HumanMessage":

        role = "user"

    elif message_type == "AIMessage":

        role = "assistant"

    else:

        return None

    return {
        "role": role,
        "content": text,
    }


# ============================================================
# LOAD HISTORY
# ============================================================

def load_history(
    thread_id: str,
) -> list[dict[str, str]]:
    """
    Load persistent history.

    First uses graph.get_thread_history().
    Then falls back directly to the checkpoint if needed.
    """

    if not thread_id:
        return []

    try:

        history = get_thread_history(
            thread_id
        )

        if history:
            return history

    except Exception:
        pass

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    connection = None

    try:

        checkpointer, connection = (
            create_checkpointer()
        )

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        checkpoint = (
            checkpointer.get_tuple(
                config
            )
        )

        if checkpoint is None:
            return []

        values = (
            checkpoint.checkpoint.get(
                "channel_values",
                {},
            )
        )

        messages = values.get(
            "messages",
            [],
        )

        history = []

        for message in messages:

            normalized = normalize_message(
                message
            )

            if normalized:

                history.append(
                    normalized
                )

        return history

    except Exception:

        return []

    finally:

        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state():

    if "thread_id" not in st.session_state:

        st.session_state.thread_id = (
            create_thread_id()
        )

    if "messages" not in st.session_state:

        st.session_state.messages = []

    if "loaded_thread" not in st.session_state:

        st.session_state.loaded_thread = (
            st.session_state.thread_id
        )

    if "delete_confirmation" not in st.session_state:

        st.session_state.delete_confirmation = None

    if "delete_cleanup_warning" not in st.session_state:

        st.session_state.delete_cleanup_warning = False

    if "image_uploader_key" not in st.session_state:

        st.session_state.image_uploader_key = 0

    if "file_uploader_key" not in st.session_state:

        st.session_state.file_uploader_key = 0

    # --------------------------------------------------------
    # Edit-message state.
    #
    # edit_draft holds the text being edited; when it is not None the
    # main input area shows an edit box instead of the normal chat
    # input. Nothing is removed from the checkpoint until the edit is
    # actually sent (see render_edit_or_chat_input), so Cancel is
    # always safe/free of side effects.
    # --------------------------------------------------------

    if "edit_draft" not in st.session_state:

        st.session_state.edit_draft = None

    if "edit_uploader_key" not in st.session_state:

        st.session_state.edit_uploader_key = 0

    # --------------------------------------------------------
    # Text-to-speech playback cache.
    #
    # Keyed by message index within the current thread so clicking
    # "Listen" on one message, or an unrelated rerun (e.g. opening the
    # Copy expander), doesn't re-synthesize audio that's already been
    # generated for this session.
    # --------------------------------------------------------

    if "tts_audio_cache" not in st.session_state:

        st.session_state.tts_audio_cache = {}

    # --------------------------------------------------------
    # Speech-to-text (voice message) state.
    #
    # voice_uploader_key resets the st.audio_input widget after a
    # recording has been processed, the same way image/file uploader
    # keys are rotated elsewhere in this file. last_voice_audio_hash
    # prevents re-transcribing/resending the same clip on reruns that
    # aren't caused by a new recording.
    # --------------------------------------------------------

    if "voice_uploader_key" not in st.session_state:

        st.session_state.voice_uploader_key = 0

    if "last_voice_audio_hash" not in st.session_state:

        st.session_state.last_voice_audio_hash = None


# ============================================================
# SELECT THREAD
# ============================================================

def select_thread(
    thread_id: str,
):
    """
    Switch to another conversation.

    History is loaded BEFORE changing the active thread.
    """

    if not thread_id:
        return

    history = load_history(
        thread_id
    )

    st.session_state.thread_id = (
        thread_id
    )

    st.session_state.messages = (
        history
    )

    st.session_state.loaded_thread = (
        thread_id
    )

    st.session_state.delete_confirmation = None

    st.session_state.edit_draft = None

    st.session_state.tts_audio_cache = {}

    st.session_state.last_voice_audio_hash = None

    st.rerun()


# ============================================================
# NEW CHAT
# ============================================================

def start_new_chat():
    """
    Start a fresh conversation without deleting old chats.
    """

    new_thread = create_thread_id()

    st.session_state.thread_id = (
        new_thread
    )

    st.session_state.messages = []

    st.session_state.loaded_thread = (
        new_thread
    )

    st.session_state.delete_confirmation = None

    st.session_state.edit_draft = None

    st.session_state.tts_audio_cache = {}

    st.session_state.last_voice_audio_hash = None

    st.rerun()


# ============================================================
# CHAT TITLE
# ============================================================

def get_chat_title(
    thread_id: str,
    fallback: str = "New Chat",
) -> str:
    """
    Use the first user message as the conversation title.
    """

    history = load_history(
        thread_id
    )

    for message in history:

        if message.get(
            "role"
        ) == "user":

            content = message.get(
                "content",
                "",
            ).strip()

            if content:

                if len(content) > 42:

                    return (
                        content[:42]
                        + "..."
                    )

                return content

    return fallback


# ============================================================
# UPLOAD VALIDATION
# ============================================================

def validate_uploaded_items(
    uploaded_images,
    uploaded_files,
) -> None:
    """Validate upload count, extensions, and sizes before processing."""

    image_count = len(uploaded_images or [])
    file_count = len(uploaded_files or [])

    if image_count > MAX_IMAGES_PER_REQUEST:
        raise ValueError(
            f"You can upload at most {MAX_IMAGES_PER_REQUEST} images per request. "
            f"You selected {image_count}. Groq currently supports a maximum of 3 images per request."
        )

    if image_count + file_count > MAX_UPLOAD_FILES:
        raise ValueError(
            f"You can upload at most {MAX_UPLOAD_FILES} items at once. "
            f"You selected {image_count + file_count}."
        )

    for uploaded_image in uploaded_images or []:

        if not getattr(uploaded_image, "name", None):
            raise ValueError("One of the uploaded images has no filename.")

        suffix = Path(uploaded_image.name).suffix.lower()
        if suffix not in ALLOWED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image type: {uploaded_image.name}"
            )

        image_bytes = uploaded_image.getvalue()

        if not image_bytes:
            raise ValueError(
                f"Image '{uploaded_image.name}' appears to be empty."
            )

        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > MAX_IMAGE_MB:
            raise ValueError(
                f"Image '{uploaded_image.name}' is {size_mb:.1f} MB. "
                f"Maximum allowed is {MAX_IMAGE_MB:g} MB."
            )

    for uploaded_file in uploaded_files or []:

        if not getattr(uploaded_file, "name", None):
            raise ValueError("One of the uploaded files has no filename.")

        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix not in ALLOWED_FILE_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {uploaded_file.name}"
            )

        file_bytes = uploaded_file.getvalue()

        if not file_bytes:
            raise ValueError(
                f"File '{uploaded_file.name}' appears to be empty."
            )

        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > MAX_FILE_MB:
            raise ValueError(
                f"File '{uploaded_file.name}' is {size_mb:.1f} MB. "
                f"Maximum allowed is {MAX_FILE_MB:g} MB."
            )


# ============================================================
# FILE HELPERS
# ============================================================

def save_uploaded_file(
    uploaded_file,
    thread_id: str,
) -> Path:
    """
    Save an uploaded file locally.

    This gives the RAG ingestion layer a persistent file path
    if it needs to process the uploaded document.
    """

    thread_dir = (
        UPLOAD_DIR / thread_id
    )

    thread_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_name = (
        Path(
            uploaded_file.name
        ).name
    )

    if not safe_name:
        safe_name = "uploaded-file"

    filename = (
        f"{uuid.uuid4().hex[:12]}_"
        f"{safe_name}"
    )

    path = (
        thread_dir / filename
    )

    path.write_bytes(
        uploaded_file.getvalue()
    )

    return path


# ============================================================
# DISPLAY UPLOADED IMAGES
# ============================================================

def render_uploaded_images(
    uploaded_files,
):
    """
    Display all selected images.
    """

    if not uploaded_files:
        return

    count = len(
        uploaded_files
    )

    columns = st.columns(
        min(
            count,
            4,
        )
    )

    for index, uploaded_file in enumerate(
        uploaded_files
    ):

        with columns[
            index % len(columns)
        ]:

            st.image(
                uploaded_file,
                caption=uploaded_file.name,
                width="stretch",
            )


# ============================================================
# EXPORT CONVERSATION
# ============================================================

def build_chat_transcript(
    messages: list[dict[str, str]],
    thread_id: str,
) -> str:
    """
    Render the current conversation as a plain Markdown transcript
    suitable for downloading.
    """

    title = get_chat_title(
        thread_id,
        "AI Assistant conversation",
    )

    lines = [
        f"# {title}",
        "",
        f"_Exported {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
    ]

    if not messages:
        lines.append("_This conversation has no messages yet._")

    for message in messages:

        role = message.get("role", "")
        content = (message.get("content") or "").strip()

        if not content:
            continue

        speaker = "You" if role == "user" else "Assistant"

        lines.append(f"**{speaker}:**")
        lines.append("")
        lines.append(content)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_export_button():
    """
    Offer the current conversation as a downloadable Markdown file.

    st.download_button triggers a direct browser download and needs no
    server-side file handling, matching how uploads are already handled
    client-side in this app.
    """

    transcript = build_chat_transcript(
        st.session_state.messages,
        st.session_state.thread_id,
    )

    st.download_button(
        "⬇️ Export chat (.md)",
        data=transcript,
        file_name=f"chat_{st.session_state.thread_id[:8]}.md",
        mime="text/markdown",
        width="stretch",
        disabled=not st.session_state.messages,
        key=f"export_{st.session_state.thread_id}",
    )


# ============================================================
# TEXT-TO-SPEECH ("LISTEN")
# ============================================================

def synthesize_message_audio(
    index: int,
    content: str,
):
    """
    Generate audio for one assistant message via agent.voice.text_to_speech
    and cache it under its message index.

    Errors (network issues, gTTS failures) are surfaced with st.error
    rather than raised, since this is a supplementary playback feature
    and must never interrupt the chat itself.
    """

    text = (content or "").strip()

    if not text:
        st.warning("There's no text to read for this message.")
        return

    truncated = len(text) > MAX_TTS_CHARS

    if truncated:
        text = text[:MAX_TTS_CHARS].rstrip()

    with st.spinner("Generating audio..."):

        try:
            audio_bytes = text_to_speech(text)

        except ValueError as exc:
            st.error(f"Nothing to read aloud: {exc}")
            return

        except RuntimeError as exc:
            st.error(f"Text-to-speech is not configured correctly: {exc}")
            return

        except Exception as exc:
            st.error(
                "Could not generate audio for this message "
                f"({type(exc).__name__}: {exc}). "
                "This is usually a temporary network issue — please try again."
            )
            return

    if not audio_bytes:
        st.error("Audio generation returned no data. Please try again.")
        return

    st.session_state.tts_audio_cache[index] = audio_bytes

    if truncated:
        st.info(
            f"Only the first {MAX_TTS_CHARS} characters were read aloud "
            "(message was truncated for playback)."
        )


# ============================================================
# DISPLAY CHAT MESSAGES
# ============================================================

def render_messages():
    """
    Render current conversation.

    The most recent user message gets an "Edit" affordance (editing
    older messages out of order is not supported — see remove_last_turn
    in agent/graph.py). Assistant messages get a "Copy" affordance using
    st.code(), which renders Streamlit's built-in one-click copy icon
    with no extra dependencies.
    """

    messages = st.session_state.messages

    last_user_index = None

    for index, message in enumerate(messages):

        if message.get("role") == "user":

            last_user_index = index

    editing = st.session_state.edit_draft is not None

    for index, message in enumerate(messages):

        role = message.get(
            "role"
        )

        content = message.get(
            "content",
            "",
        )

        if role == "user":

            with st.chat_message(
                "user"
            ):

                st.markdown(
                    content
                )

                is_last_user_message = (
                    index == last_user_index
                )

                if is_last_user_message and not editing:

                    if st.button(
                        "✏️ Edit",
                        key=f"edit_{st.session_state.thread_id}_{index}",
                        help="Edit and resend this message.",
                    ):

                        begin_message_edit(
                            content
                        )

        elif role == "assistant":

            with st.chat_message(
                "assistant"
            ):

                st.markdown(
                    content
                )

                if content:

                    if st.button(
                        "🔊 Listen",
                        key=f"listen_{st.session_state.thread_id}_{index}",
                        help="Play this reply as audio.",
                    ):

                        synthesize_message_audio(
                            index,
                            content,
                        )

                    cached_audio = st.session_state.tts_audio_cache.get(
                        index
                    )

                    if cached_audio:

                        st.audio(
                            cached_audio,
                            format="audio/mp3",
                        )

                    with st.expander(
                        "📋 Copy",
                        expanded=False,
                    ):

                        st.code(
                            content,
                            language=None,
                        )


# ============================================================
# EDIT MESSAGE FLOW
# ============================================================

def begin_message_edit(
    original_text: str,
):
    """
    Enter edit mode for the most recent user message.

    Nothing is removed from the checkpoint yet — only the draft text is
    staged — so cancelling an edit is always a no-op.
    """

    st.session_state.edit_draft = original_text
    st.rerun()


def cancel_message_edit():

    st.session_state.edit_draft = None
    st.session_state.edit_uploader_key += 1
    st.rerun()


def submit_message_edit(
    new_text: str,
):
    """
    Resend an edited message.

    The previous last turn (the old user message plus its assistant
    reply) is removed from the checkpoint first, then the edited text
    is sent as a normal new turn. If the checkpoint removal fails for
    any reason, the edit is aborted rather than silently duplicating
    the exchange.
    """

    new_text = (new_text or "").strip()

    if not new_text:
        st.error("Please enter a message before sending.")
        return

    removed = remove_last_turn(
        st.session_state.thread_id
    )

    if not removed:
        st.error(
            "Could not update this conversation for editing. "
            "Please try again, or start a new chat."
        )
        return

    # Drop the old turn locally too (last user message, and its
    # assistant reply if one was already generated).
    messages = st.session_state.messages

    last_user_index = None

    for index in range(len(messages) - 1, -1, -1):

        if messages[index].get("role") == "user":
            last_user_index = index
            break

    if last_user_index is not None:
        st.session_state.messages = messages[:last_user_index]

    st.session_state.edit_draft = None

    process_user_message(
        new_text,
        None,
        None,
    )


def render_edit_or_chat_input():
    """
    Show the edit box when a message is being edited, otherwise the
    normal upload + chat_input flow.

    Editing intentionally does not carry the original message's images
    or files forward — a resend replaces the text of the last turn, and
    re-attaching uploads is out of scope for this flow. This is a known
    limitation, not a bug: the person can always start a new message
    with fresh attachments instead of editing.
    """

    if st.session_state.edit_draft is not None:

        st.info(
            "Editing your last message. Attachments from the original "
            "message are not carried over — send new ones as a fresh "
            "message instead if needed."
        )

        edited_text = st.text_area(
            "Edit message",
            value=st.session_state.edit_draft,
            key=f"edit_box_{st.session_state.edit_uploader_key}",
            label_visibility="collapsed",
            height=120,
        )

        button_columns = st.columns(2)

        with button_columns[0]:

            if st.button(
                "Send edited message",
                type="primary",
                width="stretch",
            ):

                submit_message_edit(
                    edited_text
                )

        with button_columns[1]:

            if st.button(
                "Cancel",
                width="stretch",
            ):

                cancel_message_edit()

        return

    # ----------------------------------------------------------------
    # Normal upload + chat input flow
    # ----------------------------------------------------------------

    upload_columns = st.columns(
        2
    )

    with upload_columns[0]:

        uploaded_images = st.file_uploader(
            "🖼️ Images",
            type=[
                "png",
                "jpg",
                "jpeg",
                "webp",
                "gif",
            ],
            accept_multiple_files=True,
            key=(
                f"image_uploader_"
                f"{st.session_state.image_uploader_key}"
            ),
            help=(
                "Upload one or multiple images "
                "for visual analysis."
            ),
        )

    with upload_columns[1]:

        uploaded_files = st.file_uploader(
            "📎 Files for RAG",
            type=[
                "pdf",
                "doc",
                "docx",
                "txt",
                "md",
                "csv",
                "json",
                "xlsx",
                "xls",
            ],
            accept_multiple_files=True,
            key=(
                f"file_uploader_"
                f"{st.session_state.file_uploader_key}"
            ),
            help=(
                "Upload one or multiple documents "
                "for knowledge/RAG processing."
            ),
        )

    if uploaded_images:

        st.caption(
            f"{len(uploaded_images)} image(s) selected"
        )

        render_uploaded_images(
            uploaded_images
        )

    if uploaded_files:

        st.caption(
            f"{len(uploaded_files)} file(s) selected"
        )

        for uploaded_file in uploaded_files:

            st.write(
                f"📎 {uploaded_file.name}"
            )

    # ----------------------------------------------------------------
    # Voice message (speech-to-text)
    #
    # Recording is transcribed via agent.voice.transcribe_audio (Groq
    # Whisper) and then sent through the exact same process_user_message
    # path as typed text, so it gets the same validation, streaming,
    # and history handling. Any images/files already selected above are
    # carried along, same as a typed message would.
    # ----------------------------------------------------------------

    voice_recording = st.audio_input(
        "🎤 Or record a voice message",
        key=f"voice_input_{st.session_state.voice_uploader_key}",
    )

    if voice_recording is not None:

        voice_bytes = voice_recording.getvalue()

        if not voice_bytes:
            st.warning("The recording appears to be empty. Please try again.")

        else:

            voice_hash = hashlib.sha256(voice_bytes).hexdigest()

            # Only act on a genuinely new recording — st.audio_input keeps
            # returning the same value across reruns triggered by other
            # widgets, and re-transcribing/resending it every time would
            # silently duplicate the message.
            if voice_hash != st.session_state.last_voice_audio_hash:

                st.session_state.last_voice_audio_hash = voice_hash

                transcribed_text = ""
                transcription_failed = False

                with st.spinner("Transcribing your voice message..."):

                    try:
                        transcribed_text = transcribe_audio(voice_bytes)

                    except ValueError as exc:
                        st.error(f"Could not process the recording: {exc}")
                        transcription_failed = True

                    except RuntimeError as exc:
                        st.error(
                            f"Speech-to-text is not configured correctly: {exc}"
                        )
                        transcription_failed = True

                    except Exception as exc:
                        st.error(
                            "Could not transcribe this recording "
                            f"({type(exc).__name__}: {exc}). "
                            "This is usually a temporary network issue — "
                            "please try again."
                        )
                        transcription_failed = True

                transcribed_text = (transcribed_text or "").strip()

                if transcribed_text:

                    # Reset the recorder so the same clip isn't reprocessed
                    # after process_user_message()'s own st.rerun().
                    st.session_state.voice_uploader_key += 1

                    process_user_message(
                        transcribed_text,
                        uploaded_images,
                        uploaded_files,
                    )

                elif not transcription_failed:
                    # No exception was raised, but there was no speech to
                    # transcribe — a distinct case from the error branches
                    # above, so it gets its own message instead of a
                    # duplicate/contradictory one.
                    st.warning(
                        "No speech was detected in that recording. "
                        "Please try again."
                    )

    prompt = st.chat_input(
        "Message AI Assistant..."
    )

    if (
        prompt
        or uploaded_images
        or uploaded_files
    ):

        process_user_message(
            prompt or "",
            uploaded_images,
            uploaded_files,
        )


# ============================================================
# STREAM RESPONSE TEXT
# ============================================================

def extract_streamed_ai_text(
    node_update: Any,
) -> str:
    """
    Extract assistant text from a LangGraph "updates" chunk.

    node_update is the payload for a single ("updates", node_update)
    event from stream_agent — e.g. {"assistant": {"messages": [...]}}.
    """

    if not isinstance(
        node_update,
        dict,
    ):
        return ""

    # --------------------------------------------------------
    # Assistant node
    # --------------------------------------------------------

    assistant_update = node_update.get(
        "assistant"
    )

    if not isinstance(
        assistant_update,
        dict,
    ):
        return ""

    messages = assistant_update.get(
        "messages",
        [],
    )

    if not messages:
        return ""

    last_message = messages[-1]

    content = getattr(
        last_message,
        "content",
        "",
    )

    if isinstance(
        content,
        str,
    ):

        return content

    if isinstance(
        content,
        list,
    ):

        text_parts = []

        for block in content:

            if not isinstance(
                block,
                dict,
            ):
                continue

            if block.get(
                "type"
            ) == "text":

                text = block.get(
                    "text",
                    "",
                )

                if text:

                    text_parts.append(
                        str(text)
                    )

        return "\n".join(
            text_parts
        )

    return ""


def extract_activity_label(
    custom_event: Any,
) -> str:
    """
    Extract a human-readable label from a LangGraph "custom" chunk.

    These are the events emitted by agent.tools.emit_tool_event()
    (tool started/completed/error) and rag.ingest.emit_ingest_event()
    (ingestion progress) via langgraph.config.get_stream_writer(). This
    lets the UI show live backend activity ("Searching the web...",
    "Searching indexed documents...") instead of a silent wait.
    """

    if isinstance(custom_event, dict):

        message = custom_event.get("message")

        if isinstance(message, str) and message.strip():
            return message.strip()

    return ""


# ============================================================
# PROCESS USER MESSAGE
# ============================================================

def process_user_message(
    prompt: str,
    uploaded_images,
    uploaded_files,
):
    """
    Send one new message to LangGraph.

    Supports:

        text

        multiple images

        multiple files

        text + multiple images

        text + multiple files

        text + images + files
    """

    prompt = (
        prompt or ""
    ).strip()

    if len(prompt) > MAX_PROMPT_CHARS:
        st.error(
            f"Your message is too long ({len(prompt)} characters). "
            f"Please keep it under {MAX_PROMPT_CHARS} characters."
        )
        return

    has_images = bool(
        uploaded_images
    )

    has_files = bool(
        uploaded_files
    )

    # --------------------------------------------------------
    # Validate uploads before changing chat state
    # --------------------------------------------------------

    try:
        validate_uploaded_items(
            uploaded_images,
            uploaded_files,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    # --------------------------------------------------------
    # Nothing to send
    # --------------------------------------------------------

    if not prompt and not has_images and not has_files:

        return

    # --------------------------------------------------------
    # User-visible text
    # --------------------------------------------------------

    display_text = prompt

    if not display_text:

        if has_images and has_files:

            display_text = (
                "Please analyze the uploaded images "
                "and files together."
            )

        elif has_images:

            display_text = (
                "Please analyze these uploaded images."
            )

        elif has_files:

            display_text = (
                "Please analyze the uploaded files "
                "and provide the relevant information."
            )

    # --------------------------------------------------------
    # Display user message immediately
    # --------------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            display_text
        )

        if has_images:

            render_uploaded_images(
                uploaded_images
            )

        if has_files:

            st.caption(
                "📎 "
                + ", ".join(
                    file.name
                    for file in uploaded_files
                )
            )

    # --------------------------------------------------------
    # Build agent message
    # --------------------------------------------------------

    agent_message = {
        "role": "user",
        "content": prompt,
    }

    # --------------------------------------------------------
    # Multiple images
    # --------------------------------------------------------

    if has_images:

        agent_message[
            "images"
        ] = []

        for uploaded_image in uploaded_images:

            agent_message[
                "images"
            ].append(
                {
                    "image_bytes": (
                        uploaded_image.getvalue()
                    ),
                    "image_type": (
                        uploaded_image.type
                        or "image/png"
                    ),
                    "filename": (
                        uploaded_image.name
                    ),
                }
            )

    # --------------------------------------------------------
    # Multiple files
    # --------------------------------------------------------

    if has_files:

        agent_message[
            "files"
        ] = []

        for uploaded_file in uploaded_files:

            # Save a persistent copy for the RAG layer.
            try:
                saved_path = save_uploaded_file(
                    uploaded_file,
                    st.session_state.thread_id,
                )
            except OSError as exc:
                st.error(
                    f"Could not save '{uploaded_file.name}': {exc}"
                )
                return

            agent_message[
                "files"
            ].append(
                {
                    "file_bytes": (
                        uploaded_file.getvalue()
                    ),
                    "file_type": (
                        uploaded_file.type
                        or "application/octet-stream"
                    ),
                    "filename": (
                        uploaded_file.name
                    ),
                    "file_path": str(
                        saved_path
                    ),
                }
            )

    # --------------------------------------------------------
    # Save user message in local UI history
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": display_text,
        }
    )

    # --------------------------------------------------------
    # Assistant response
    # --------------------------------------------------------

    assistant_placeholder = None
    streamed_text = ""

    with st.chat_message(
        "assistant"
    ):

        # ------------------------------------------------
        # Live backend-activity panel.
        #
        # Shows tool/ingestion progress events ("custom" chunks)
        # as they happen, so a multi-second tool call (web search,
        # RAG search, calculator, etc.) has visible progress instead
        # of looking frozen. Collapses to a short summary once the
        # response is ready.
        # ------------------------------------------------

        activity_status = st.status(
            "Working...",
            expanded=False,
        )

        assistant_placeholder = st.empty()

        activity_seen = False

        try:

            for stream_mode, chunk in stream_agent(
                user_id="local",
                messages=[agent_message],
                thread_id=st.session_state.thread_id,
            ):

                if stream_mode == "custom":

                    label = extract_activity_label(
                        chunk
                    )

                    if label:

                        activity_seen = True

                        activity_status.update(
                            label=label
                        )

                    continue

                # stream_mode == "updates"

                new_text = (
                    extract_streamed_ai_text(
                        chunk
                    )
                )

                if new_text:

                    streamed_text = new_text

                    assistant_placeholder.markdown(
                        streamed_text
                    )

            if activity_seen:

                activity_status.update(
                    label="Done",
                    state="complete",
                )

            else:

                # No tool activity for this turn (a plain chat reply) —
                # collapse the panel instead of leaving "Working..." up.
                activity_status.update(
                    label="Done",
                    state="complete",
                )

            # ------------------------------------------------
            # If streaming didn't expose final text,
            # reload the persistent checkpoint.
            # ------------------------------------------------

            if not streamed_text:

                history = load_history(
                    st.session_state.thread_id
                )

                for message in reversed(
                    history
                ):

                    if message.get(
                        "role"
                    ) == "assistant":

                        streamed_text = (
                            message.get(
                                "content",
                                "",
                            )
                        )

                        break

                if streamed_text:

                    assistant_placeholder.markdown(
                        streamed_text
                    )
                else:

                    streamed_text = (
                        "Sorry, I didn't get a response. "
                        "Please try again."
                    )

                    assistant_placeholder.warning(
                        streamed_text
                    )

        except Exception as exc:

            try:
                activity_status.update(
                    label="Failed",
                    state="error",
                )
            except Exception:
                pass

            streamed_text = (
                "Sorry, something went wrong while "
                "processing your request."
            )

            assistant_placeholder.error(
                f"{streamed_text}\n\n"
                f"{type(exc).__name__}: {exc}"
            )

    # --------------------------------------------------------
    # Save assistant response
    # --------------------------------------------------------

    if streamed_text:

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": streamed_text,
            }
        )

    # --------------------------------------------------------
    # Reset uploaders safely.
    #
    # Do NOT modify the actual Streamlit widget's
    # session_state value.
    # --------------------------------------------------------

    st.session_state.image_uploader_key += 1

    st.session_state.file_uploader_key += 1

    st.rerun()


# ============================================================
# DELETE CONFIRMATION DIALOG
# ============================================================

@st.dialog("Delete chat?")
def confirm_delete_dialog(thread_id: str):
    """Show a modal confirmation before deleting a chat."""

    title = get_chat_title(thread_id, "this chat")

    st.write(f'Are you sure you want to delete "{title}"?')
    st.caption(
        "This removes the chat from your chat history. "
        "This action cannot be undone from the interface."
    )

    confirm_columns = st.columns(2)

    with confirm_columns[0]:
        if st.button(
            "Delete",
            type="primary",
            width="stretch",
            key=f"confirm_delete_{thread_id}",
        ):
            deleted_ok, cleanup_error = delete_thread(thread_id)

            if not deleted_ok:
                st.error(
                    cleanup_error or "The chat could not be deleted."
                )
                return

            if st.session_state.thread_id == thread_id:
                st.session_state.thread_id = create_thread_id()
                st.session_state.messages = []
                st.session_state.loaded_thread = st.session_state.thread_id

            st.session_state.delete_confirmation = None
            st.session_state.delete_cleanup_warning = bool(cleanup_error)
            st.rerun()

    with confirm_columns[1]:
        if st.button(
            "Cancel",
            width="stretch",
            key=f"cancel_delete_{thread_id}",
        ):
            st.session_state.delete_confirmation = None
            st.rerun()


# ============================================================
# SIDEBAR CHAT ITEM
# ============================================================

def render_chat_item(
    thread_id: str,
    title: str,
    pinned: bool,
):
    """
    Render one chat in the sidebar.
    """

    row = st.columns(
        [0.70, 0.15, 0.15]
    )

    # --------------------------------------------------------
    # Open chat
    # --------------------------------------------------------

    with row[0]:

        label = title

        if pinned:

            label = "📌 " + label

        if st.button(
            label,
            key=f"open_{thread_id}",
            width="stretch",
        ):

            select_thread(
                thread_id
            )

    # --------------------------------------------------------
    # Pin
    # --------------------------------------------------------

    with row[1]:

        if pinned:

            if st.button(
                "📌",
                key=f"unpin_{thread_id}",
                help="Unpin chat",
            ):

                unpin_thread(
                    thread_id
                )

                st.rerun()

        else:

            if st.button(
                "📍",
                key=f"pin_{thread_id}",
                help="Pin chat",
            ):

                pin_thread(
                    thread_id
                )

                st.rerun()

    # --------------------------------------------------------
    # Delete
    # --------------------------------------------------------

    with row[2]:

        if st.button(
            "🗑️",
            key=f"delete_{thread_id}",
            help="Delete chat",
        ):

            st.session_state.delete_confirmation = (
                thread_id
            )

            st.rerun()


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    with st.sidebar:

        st.title(
            "AI Assistant"
        )

        # ----------------------------------------------------
        # New Chat
        # ----------------------------------------------------

        if st.button(
            "＋ New Chat",
            width="stretch",
        ):

            start_new_chat()

        st.divider()

        # ----------------------------------------------------
        # Load chats
        # ----------------------------------------------------

        threads = load_threads()

        try:

            pinned_threads = (
                get_pinned_threads()
            )

        except Exception:

            pinned_threads = set()

        # ----------------------------------------------------
        # Pinned chats
        # ----------------------------------------------------

        pinned_items = [
            (
                thread_id,
                threads[thread_id],
            )
            for thread_id in threads
            if thread_id in pinned_threads
        ]

        if pinned_items:

            st.subheader(
                "📌 Pinned Chats"
            )

            for thread_id, title in pinned_items:

                render_chat_item(
                    thread_id,
                    title,
                    True,
                )

            st.divider()

        # ----------------------------------------------------
        # Previous chats
        # ----------------------------------------------------

        st.subheader(
            "Previous Chats"
        )

        previous_items = [
            (
                thread_id,
                title,
            )
            for thread_id, title in threads.items()
            if thread_id not in pinned_threads
        ]

        if not previous_items:

            st.caption(
                "No previous chats yet."
            )

        else:

            for thread_id, title in previous_items:

                render_chat_item(
                    thread_id,
                    title,
                    False,
                )

        # ----------------------------------------------------
        # Delete confirmation dialog
        # ----------------------------------------------------

        delete_id = st.session_state.delete_confirmation

        if delete_id:
            confirm_delete_dialog(delete_id)


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        initialize_ui_database()

    except Exception as exc:

        st.error(
            "The app database could not be initialized: "
            f"{exc}"
        )
        st.stop()
        return

    initialize_session_state()

    # --------------------------------------------------------
    # Keep session history synchronized with active thread.
    # --------------------------------------------------------

    if (
        st.session_state.loaded_thread
        != st.session_state.thread_id
    ):

        st.session_state.messages = (
            load_history(
                st.session_state.thread_id
            )
        )

        st.session_state.loaded_thread = (
            st.session_state.thread_id
        )

    # --------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------

    render_sidebar()

    if st.session_state.get("delete_cleanup_warning"):
        st.info(
            "The chat was removed from the interface. "
            "Physical checkpoint cleanup was not available "
            "with the installed checkpointer."
        )
        st.session_state.delete_cleanup_warning = False

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    header_columns = st.columns(
        [0.8, 0.2]
    )

    with header_columns[0]:

        st.title(
            "AI Assistant"
        )

    with header_columns[1]:

        render_export_button()

    # --------------------------------------------------------
    # Conversation
    # --------------------------------------------------------

    render_messages()

    # --------------------------------------------------------
    # Upload area / edit box / chat input
    # --------------------------------------------------------

    st.divider()

    render_edit_or_chat_input()


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()