import os
import pickle
import threading
import time
import contextvars
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from PIL import Image
import pytesseract

import requests
from bs4 import BeautifulSoup

from filelock import FileLock, Timeout

from utils.security import (
    validate_file_security,
    sanitize_path,
    sanitize_user_input,
)

# LangGraph custom streaming.
# Import safely so ingest.py can still be used outside a LangGraph run.
try:
    from langgraph.config import get_stream_writer
except ImportError:
    get_stream_writer = None


# ---------------------------------------------------------------------------
# PATHS
#
# NOTE: previously FAISS_DIR was the literal relative string "data/faiss",
# and sanitize_path() was called with base_dir="data". Both resolve relative
# to the process's current working directory, which is only "data/" when the
# app happens to be launched from the project root. Any other launch method
# (a systemd unit, a different working directory, Streamlit Cloud, etc.)
# would silently point ingestion at the wrong place. Anchoring everything to
# this file's location makes it deployment-independent while keeping the
# same on-disk layout ("<project_root>/data/faiss").
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = str(PROJECT_ROOT / "data")
DATA_DIR = os.getenv("DATA_DIR", DEFAULT_DATA_DIR)

FAISS_DIR = os.path.join(DATA_DIR, "faiss")
METADATA_PATH = os.path.join(FAISS_DIR, "metadata.pkl")
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
UPLOAD_DIR = os.path.join(DATA_DIR, "upload")

# Concurrent ingestion (multi-file upload) can otherwise race on the same
# FAISS index: two calls doing load -> add_documents -> save_local at once
# can corrupt the index or silently drop one file's chunks. All writers to
# FAISS_DIR take this lock. 60s covers a large batch of files/large chunks;
# a run that legitimately needs longer will simply raise Timeout rather than
# hang forever, which is surfaced to the caller as a clear error below.
FAISS_LOCK_PATH = os.path.join(FAISS_DIR, ".faiss.lock")
FAISS_LOCK_TIMEOUT = float(os.getenv("FAISS_LOCK_TIMEOUT", "60"))

# Embedding chunks in one call gives no visibility into progress and can look
# "stuck" on large files. Batching lets us emit real progress between calls.
EMBED_BATCH_SIZE = int(os.getenv("INGEST_EMBED_BATCH_SIZE", "20"))

# How often the model-load heartbeat ticks while HuggingFaceEmbeddings
# downloads/initializes (a step that can't be subdivided into real progress).
HEARTBEAT_INTERVAL_SECONDS = 2.5


# ---------------------------------------------------------------------------
# STREAMING / BACKEND STATUS
# ---------------------------------------------------------------------------

# Lets ingest_file()/ingest_url() report progress when called directly
# (e.g. from a Streamlit upload handler) rather than only from inside a
# LangGraph node. A caller sets this to a callable before ingesting and
# resets it afterwards; emit_ingest_event fans out to both destinations
# without any other function in this module needing to change.
progress_context: "contextvars.ContextVar[Any]" = contextvars.ContextVar(
    "ingest_progress_callback",
    default=None,
)


def emit_ingest_event(
    message: str,
    *,
    event: str,
    stage: str,
    **metadata: Any,
) -> None:
    """
    Sends a real backend progress event through LangGraph, and/or to a
    directly-registered progress callback (see progress_context).

    This function does NOT control tool selection or agent routing.
    It only reports what the ingestion pipeline is currently doing.
    """

    payload = {
        "message": message,
        "event": event,
        "stage": stage,
        "source": "rag_ingestion",
        **metadata,
    }

    if get_stream_writer is not None:
        try:
            writer = get_stream_writer()
            writer(payload)
        except Exception:
            # Streaming progress must never break ingestion.
            pass

    callback = progress_context.get()

    if callback is not None:
        try:
            callback(payload)
        except Exception:
            # A broken UI callback must never break ingestion.
            pass


class _Heartbeat:
    """Emits periodic 'still working' events during a step with no
    natural sub-progress (e.g. first-time embedding model load).

    Used as a context manager so the ticking thread is always stopped,
    even if the wrapped step raises.
    """

    def __init__(self, message: str, *, stage: str, **metadata: Any):
        self._message = message
        self._stage = stage
        self._metadata = metadata
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        elapsed = 0
        while not self._stop_event.wait(HEARTBEAT_INTERVAL_SECONDS):
            elapsed += HEARTBEAT_INTERVAL_SECONDS
            emit_ingest_event(
                f"{self._message} ({elapsed:.0f}s)",
                event="stage_heartbeat",
                stage=self._stage,
                **self._metadata,
            )

    def __enter__(self) -> "_Heartbeat":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=HEARTBEAT_INTERVAL_SECONDS)


def _faiss_lock() -> FileLock:
    os.makedirs(FAISS_DIR, exist_ok=True)
    return FileLock(FAISS_LOCK_PATH, timeout=FAISS_LOCK_TIMEOUT)


# ---------------------------------------------------------------------------
# EMBEDDINGS
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns the initialized HuggingFace embedding model.

    Cached so the embedding model is loaded only once per process.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def _get_embeddings_with_progress(*, source_name: str) -> HuggingFaceEmbeddings:
    """Load (or reuse) the embedding model, with a heartbeat on first load.

    lru_cache means this is only slow the very first time in a process;
    the heartbeat only actually ticks during that first, real load.
    """

    if get_embeddings.cache_info().currsize > 0:
        # Already loaded in this process - effectively instant.
        return get_embeddings()

    with _Heartbeat(
        "🧠 Loading embedding model (first run may take a moment)",
        stage="embeddings",
        filename=source_name,
    ):
        return get_embeddings()


# ---------------------------------------------------------------------------
# IMAGE OCR
# ---------------------------------------------------------------------------

def load_image_text(file_path: str) -> List[Document]:
    """
    Extract text from JPG/JPEG/PNG image files using Tesseract OCR.
    """

    filename = os.path.basename(file_path)

    emit_ingest_event(
        f"🔎 Extracting text from image: {filename}",
        event="stage_started",
        stage="ocr",
        filename=filename,
    )

    try:
        image = Image.open(file_path)

        extracted_text = pytesseract.image_to_string(image)

        if not extracted_text or not extracted_text.strip():
            raise ValueError(
                "No readable text found in the image."
            )

        emit_ingest_event(
            f"✓ Text extracted from image: {filename}",
            event="stage_completed",
            stage="ocr",
            filename=filename,
        )

        return [
            Document(
                page_content=extracted_text.strip(),
                metadata={
                    "source": filename,
                    "page": 1,
                },
            )
        ]

    except Exception as err:
        emit_ingest_event(
            f"✗ Image text extraction failed: {filename}",
            event="stage_error",
            stage="ocr",
            filename=filename,
            error=str(err),
        )

        raise ValueError(
            f"Could not extract text from image "
            f"({filename}): {err}. "
            "Ensure Tesseract-OCR is installed on your operating system."
        )


# ---------------------------------------------------------------------------
# DOCUMENT LOADING
# ---------------------------------------------------------------------------

def load_document(file_path: str) -> List[Document]:
    """
    Select the appropriate document loader based on file extension.
    """

    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)

    if ext in [".jpg", ".jpeg", ".png"]:
        return load_image_text(file_path)

    if ext == ".pdf":
        emit_ingest_event(
            f"📄 Extracting text from PDF: {filename}",
            event="stage_started",
            stage="text_extraction",
            filename=filename,
        )

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        emit_ingest_event(
            f"✓ PDF text extraction completed: {filename}",
            event="stage_completed",
            stage="text_extraction",
            filename=filename,
            pages=len(documents),
        )

        return documents

    if ext == ".csv":
        emit_ingest_event(
            f"📊 Reading CSV file: {filename}",
            event="stage_started",
            stage="text_extraction",
            filename=filename,
        )

        loader = CSVLoader(file_path)
        documents = loader.load()

        emit_ingest_event(
            f"✓ CSV reading completed: {filename}",
            event="stage_completed",
            stage="text_extraction",
            filename=filename,
            rows=len(documents),
        )

        return documents

    if ext in [".docx", ".doc"]:
        emit_ingest_event(
            f"📝 Extracting text from Word document: {filename}",
            event="stage_started",
            stage="text_extraction",
            filename=filename,
        )

        try:
            loader = Docx2txtLoader(file_path)
            documents = loader.load()

            emit_ingest_event(
                f"✓ Word document extraction completed: {filename}",
                event="stage_completed",
                stage="text_extraction",
                filename=filename,
            )

            return documents

        except Exception as err:
            raise ValueError(
                f"Could not load Word file: {err}"
            )

    # Plain text / markdown / Python / extensionless files.
    emit_ingest_event(
        f"📖 Reading text file: {filename}",
        event="stage_started",
        stage="text_extraction",
        filename=filename,
    )

    try:
        loader = TextLoader(
            file_path,
            encoding="utf-8",
        )

        documents = loader.load()

        emit_ingest_event(
            f"✓ Text extraction completed: {filename}",
            event="stage_completed",
            stage="text_extraction",
            filename=filename,
        )

        return documents

    except UnicodeDecodeError:

        try:
            loader = TextLoader(
                file_path,
                encoding="latin-1",
            )

            documents = loader.load()

            emit_ingest_event(
                f"✓ Text extraction completed: {filename}",
                event="stage_completed",
                stage="text_extraction",
                filename=filename,
            )

            return documents

        except Exception:

            with open(
                file_path,
                "r",
                encoding="utf-8",
                errors="ignore",
            ) as f:
                content = f.read()

            if content.strip():
                return [
                    Document(
                        page_content=content,
                        metadata={
                            "source": filename,
                        },
                    )
                ]

            return []

    except Exception as err:

        try:
            with open(
                file_path,
                "r",
                encoding="utf-8",
                errors="ignore",
            ) as f:
                content = f.read()

            if content.strip():
                return [
                    Document(
                        page_content=content,
                        metadata={
                            "source": filename,
                        },
                    )
                ]

        except Exception:
            pass

        raise ValueError(
            f"Could not read text file ({ext}): {err}"
        )


# ---------------------------------------------------------------------------
# METADATA REGISTRY
# ---------------------------------------------------------------------------

def _load_metadata_registry() -> List[Dict[str, Any]]:
    """
    Load stored chunk metadata from disk.
    """

    if not os.path.exists(METADATA_PATH):
        return []

    try:
        with open(METADATA_PATH, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, list):
            return data

        return []

    except Exception:
        return []


def _save_metadata_registry(
    metadata_list: List[Dict[str, Any]],
) -> None:
    """
    Save updated chunk metadata to disk.
    """

    os.makedirs(
        FAISS_DIR,
        exist_ok=True,
    )

    with open(
        METADATA_PATH,
        "wb",
    ) as f:
        pickle.dump(
            metadata_list,
            f,
        )


# ---------------------------------------------------------------------------
# FAISS HELPERS
# ---------------------------------------------------------------------------

def _load_vectorstore(
    embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    """
    Load the existing FAISS vector store.
    """

    return FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def _chunk_batches(
    chunks: List[Document],
    batch_size: int,
) -> List[List[Document]]:
    if batch_size < 1:
        batch_size = len(chunks) or 1

    return [
        chunks[i : i + batch_size]
        for i in range(0, len(chunks), batch_size)
    ]


def _save_chunks_to_faiss(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    *,
    source_name: str = "",
) -> None:
    """
    Append chunks to an existing FAISS index or create a new one.

    Embeds in batches (instead of one call for the whole file) so callers
    get real incremental progress events on large files rather than a
    single opaque call with no feedback until it finishes.

    Must be called while holding the FAISS lock (see ingest_file/ingest_url).
    """

    os.makedirs(
        FAISS_DIR,
        exist_ok=True,
    )

    total = len(chunks)
    batches = _chunk_batches(chunks, EMBED_BATCH_SIZE)

    vectorstore = None
    if os.path.exists(INDEX_PATH):
        vectorstore = _load_vectorstore(embeddings)

    embedded_so_far = 0

    for batch in batches:

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

        embedded_so_far += len(batch)

        emit_ingest_event(
            f"🧠 Embedded {embedded_so_far}/{total} chunks...",
            event="stage_progress",
            stage="vectorization",
            filename=source_name,
            embedded=embedded_so_far,
            total=total,
        )

    # vectorstore is guaranteed non-None here because ingest_file/ingest_url
    # both validate chunks is non-empty before calling this function.
    vectorstore.save_local(FAISS_DIR)


def _register_chunks(
    source_name: str,
    chunks: List[Document],
) -> None:
    """
    Add chunk metadata to the persistent registry.
    """

    registry = _load_metadata_registry()

    for chunk_index, chunk in enumerate(chunks):

        registry.append(
            {
                "source": source_name,
                "page": chunk.metadata.get(
                    "page",
                    0,
                ),
                "chunk_id": chunk.metadata.get("chunk_id", chunk_index),
                "text": chunk.page_content,
                "user_id": chunk.metadata.get("user_id"),
            }
        )

    _save_metadata_registry(
        registry
    )


def _index_chunks_locked(
    source_name: str,
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
) -> None:
    """
    Save chunks to FAISS and register their metadata as one locked,
    all-or-nothing step.

    Both writes happen under the same lock acquisition so that:
      - concurrent ingestions (multi-file upload) cannot race on the
        FAISS index, and
      - the vector index and the metadata registry can never end up out
        of sync with each other (previously these were two independent,
        unlocked calls).
    """

    try:
        with _faiss_lock():
            _save_chunks_to_faiss(
                chunks,
                embeddings,
                source_name=source_name,
            )
            _register_chunks(source_name, chunks)

    except Timeout as exc:
        raise ValueError(
            "The knowledge base is busy processing another file right now. "
            "Please try again in a moment."
        ) from exc


# ---------------------------------------------------------------------------
# FILE INGESTION
# ---------------------------------------------------------------------------

def ingest_file(file_path: str,user_id: str | None = None) -> int:
    """
    Securely load a document, extract text, split it into chunks,
    generate embeddings, update FAISS, and save metadata.

    Returns:
        Number of chunks added.
    """

    clean_input = sanitize_user_input(
        file_path
    )

    if not clean_input:
        raise ValueError("A valid file path is required.")

    safe_path_obj = sanitize_path(
        clean_input,
        base_dir=DATA_DIR,
    )

    safe_file_path = str(
        safe_path_obj
    )

    source_name = os.path.basename(
        safe_file_path
    )

    if not source_name:
        raise ValueError("The file path does not resolve to a filename.")

    emit_ingest_event(
        f"🔐 Validating file: {source_name}",
        event="stage_started",
        stage="security_validation",
        filename=source_name,
    )

    # -------------------------------------------------------
    # SECURITY VALIDATION
    #
    # NOTE: previously a ValueError here was silently swallowed whenever
    # the file had no extension, on the assumption that
    # validate_file_security() would always reject extensionless files.
    # That let ANY validation failure (oversized file, empty file, etc.)
    # through for extensionless files, not just the extension check.
    # Extensionless files are now an explicitly allowed extension in
    # utils.security.ALLOWED_EXTENSIONS, so validation always runs for
    # real and any failure here is a genuine, unconditional rejection.
    # -------------------------------------------------------

    try:
        validate_file_security(
            safe_file_path
        )

    except ValueError as sec_err:

        emit_ingest_event(
            f"✗ File validation failed: {source_name}",
            event="stage_error",
            stage="security_validation",
            filename=source_name,
            error=str(sec_err),
        )
        raise
    except FileNotFoundError as not_found_err:
        emit_ingest_event(
            f"✗ File not found: {source_name}",
            event="stage_error",
            stage="security_validation",
            filename=source_name,
            error=str(not_found_err),
        )
        raise

    emit_ingest_event(
        f"✓ File validation passed: {source_name}",
        event="stage_completed",
        stage="security_validation",
        filename=source_name,
    )

    os.makedirs(
        FAISS_DIR,
        exist_ok=True,
    )

    # -------------------------------------------------------
    # TEXT EXTRACTION
    # -------------------------------------------------------

    emit_ingest_event(
        f"📥 Loading file content: {source_name}",
        event="stage_started",
        stage="loading",
        filename=source_name,
    )

    documents = []

    try:
        documents = load_document(
            safe_file_path
        )

    except Exception as err:

        emit_ingest_event(
            f"⚠️ Standard loader failed. Trying fallback reader: {source_name}",
            event="stage_warning",
            stage="loading",
            filename=source_name,
            error=str(err),
        )

        documents = []

    # Fallback text reader.
    if not documents:

        try:

            with open(
                safe_file_path,
                "r",
                encoding="utf-8",
                errors="ignore",
            ) as f:
                content = f.read()

            if content.strip():

                documents = [
                    Document(
                        page_content=content,
                        metadata={
                            "source": source_name
                        },
                    )
                ]

        except Exception as err:

            emit_ingest_event(
                f"✗ Could not read file: {source_name}",
                event="stage_error",
                stage="loading",
                filename=source_name,
                error=str(err),
            )

            raise ValueError(
                f"Could not read content from this file: {err}"
            )

    if not documents:

        emit_ingest_event(
            f"✗ No readable text found: {source_name}",
            event="stage_error",
            stage="loading",
            filename=source_name,
        )

        raise ValueError(
            "The file contains no readable text."
        )

    emit_ingest_event(
        f"✓ File content loaded: {source_name}",
        event="stage_completed",
        stage="loading",
        filename=source_name,
        documents=len(documents),
    )

    # -------------------------------------------------------
    # CHUNKING
    # -------------------------------------------------------

    emit_ingest_event(
        f"✂️ Splitting {source_name} into searchable chunks...",
        event="stage_started",
        stage="chunking",
        filename=source_name,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(
        documents
    )

    if not chunks:
        emit_ingest_event(
            f"✗ No chunks could be created: {source_name}",
            event="stage_error",
            stage="chunking",
            filename=source_name,
        )
        raise ValueError(
            "No text chunks could be extracted from this file."
        )

    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata["source"] = source_name
        chunk.metadata["chunk_id"] = chunk_index
        if user_id:
            chunk.metadata["user_id"] = user_id

    emit_ingest_event(
        f"✓ Created {len(chunks)} searchable chunks.",
        event="stage_completed",
        stage="chunking",
        filename=source_name,
        chunks=len(chunks),
    )

    # -------------------------------------------------------
    # EMBEDDINGS
    # -------------------------------------------------------

    emit_ingest_event(
        f"🧠 Preparing embedding model for {source_name}...",
        event="stage_started",
        stage="embeddings",
        filename=source_name,
        chunks=len(chunks),
    )

    try:
        embeddings = _get_embeddings_with_progress(source_name=source_name)
    except Exception as err:
        emit_ingest_event(
            f"✗ Could not load the embedding model: {source_name}",
            event="stage_error",
            stage="embeddings",
            filename=source_name,
            error=str(err),
        )
        raise ValueError(
            f"Could not load the embedding model: {err}"
        )

    emit_ingest_event(
        f"✓ Embedding model ready. Creating vectors for {len(chunks)} chunks...",
        event="stage_started",
        stage="vectorization",
        filename=source_name,
        chunks=len(chunks),
    )

    # -------------------------------------------------------
    # FAISS + METADATA (locked, all-or-nothing)
    # -------------------------------------------------------

    try:
        _index_chunks_locked(
            source_name,
            chunks,
            embeddings,
        )
    except Exception as err:
        emit_ingest_event(
            f"✗ Could not update the knowledge base: {source_name}",
            event="stage_error",
            stage="vectorization",
            filename=source_name,
            error=str(err),
        )
        raise

    emit_ingest_event(
        f"✓ FAISS vector index and metadata updated with {len(chunks)} chunks.",
        event="stage_completed",
        stage="vectorization",
        filename=source_name,
        chunks=len(chunks),
    )

    # -------------------------------------------------------
    # COMPLETE
    # -------------------------------------------------------

    emit_ingest_event(
        f"✅ {source_name} is ready for RAG search.",
        event="ingestion_completed",
        stage="complete",
        filename=source_name,
        chunks=len(chunks),
    )

    return len(chunks)


# ---------------------------------------------------------------------------
# MULTI-FILE COMPATIBILITY WRAPPER
# ---------------------------------------------------------------------------

def ingest_multiple_files(
    file_paths: List[str] | tuple[str, ...] | str,
    *,
    user_id: str | None = None,
) -> Dict[str, List[str]]:
    """Compatibility wrapper for older tests and callers.

    It ingests each file individually while preserving the current multi-user
    file-ingestion logic, and returns a summary map with successful and failed
    sources.
    """

    if isinstance(file_paths, str):
        file_paths = [file_paths]

    successful: List[str] = []
    failed: List[str] = []

    for file_path in file_paths:
        try:
            ingest_file(file_path, user_id=user_id)
            successful.append(str(file_path))
        except Exception:
            failed.append(str(file_path))

    return {
        "successful": successful,
        "failed": failed,
    }


# ---------------------------------------------------------------------------
# LIST INDEXED DOCUMENTS
# ---------------------------------------------------------------------------

def list_indexed_documents() -> List[Dict[str, Any]]:
    """
    Return all unique documents currently represented
    in the metadata registry.
    """

    registry = _load_metadata_registry()

    if not registry:
        return []

    doc_counts = {}

    for entry in registry:

        src = entry.get(
            "source",
            "Unknown",
        )

        doc_counts[src] = (
            doc_counts.get(src, 0) + 1
        )

    return [
        {
            "source": src,
            "chunk_count": count,
        }
        for src, count in doc_counts.items()
    ]


# ---------------------------------------------------------------------------
# DELETE DOCUMENT
# ---------------------------------------------------------------------------

def delete_document(
    source_name: str,
) -> bool:
    """
    Remove a document from the metadata registry and
    rebuild FAISS using the remaining documents.
    """

    if not isinstance(source_name, str) or not source_name.strip():
        return False

    try:
        with _faiss_lock():

            registry = _load_metadata_registry()

            if not registry:
                return False

            clean_source = os.path.basename(
                source_name
            ).strip()

            remaining_entries = [
                item
                for item in registry
                if item.get("source") != clean_source
            ]

            if len(remaining_entries) == len(registry):
                return False

            emit_ingest_event(
                f"🗑️ Removing {clean_source} from the knowledge base...",
                event="stage_started",
                stage="deletion",
                filename=clean_source,
            )

            # No documents remain.
            if not remaining_entries:

                if os.path.exists(INDEX_PATH):
                    os.remove(INDEX_PATH)

                idx_pkl = os.path.join(
                    FAISS_DIR,
                    "index.pkl",
                )

                if os.path.exists(idx_pkl):
                    os.remove(idx_pkl)

                if os.path.exists(METADATA_PATH):
                    os.remove(METADATA_PATH)

                emit_ingest_event(
                    f"✓ Removed {clean_source} and cleared the empty index.",
                    event="stage_completed",
                    stage="deletion",
                    filename=clean_source,
                )

                return True

            # Rebuild remaining vector store.
            rebuild_docs = [
                Document(
                    page_content=item["text"],
                    metadata={
                        "source": item["source"],
                        "page": item.get(
                            "page",
                            0,
                        ),
                    },
                )
                for item in remaining_entries
            ]

            embeddings = _get_embeddings_with_progress(source_name=clean_source)

            vectorstore = FAISS.from_documents(
                rebuild_docs,
                embeddings,
            )

            vectorstore.save_local(
                FAISS_DIR
            )

            _save_metadata_registry(
                remaining_entries
            )

            emit_ingest_event(
                f"✓ Removed {clean_source} and rebuilt the FAISS index.",
                event="stage_completed",
                stage="deletion",
                filename=clean_source,
            )

            return True

    except Timeout:
        emit_ingest_event(
            f"✗ Could not remove {source_name}: knowledge base is busy.",
            event="stage_error",
            stage="deletion",
            filename=source_name,
        )
        raise ValueError(
            "The knowledge base is busy processing another file. "
            "Please try again in a moment."
        )


# ---------------------------------------------------------------------------
# RAG SEARCH
# ---------------------------------------------------------------------------

def search_documents(
    query: str,
    k: int = 5,
) -> List[Document]:
    """
    Search the FAISS vector store for the top matching chunks.
    """

    if not isinstance(
        query,
        str,
    ) or not query.strip():

        return []

    if not os.path.exists(
        INDEX_PATH
    ):

        return []

    try:

        embeddings = get_embeddings()

        vectorstore = _load_vectorstore(
            embeddings
        )

        results = vectorstore.similarity_search(
            query.strip(),
            k=k,
        )

        return results

    except Exception as err:

        print(
            f"Error searching vector database: {err}"
        )

        return []


# ---------------------------------------------------------------------------
# URL INGESTION
# ---------------------------------------------------------------------------

def ingest_url(
    url: str,
) -> int:
    """
    Fetch readable text from a web URL, split it into chunks,
    create embeddings, and add it to FAISS.
    """

    if not isinstance(
        url,
        str,
    ) or not url.strip():

        raise ValueError(
            "A valid URL is required."
        )

    url = url.strip()

    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(
            "Only http:// and https:// URLs are supported."
        )

    emit_ingest_event(
        f"🌐 Fetching web page...",
        event="stage_started",
        stage="url_fetch",
        url=url,
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36"
        )
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=10,
        )

        response.raise_for_status()

    except Exception as err:

        emit_ingest_event(
            "✗ Failed to fetch web page.",
            event="stage_error",
            stage="url_fetch",
            url=url,
            error=str(err),
        )

        raise ValueError(
            f"Could not fetch URL: {err}"
        )

    emit_ingest_event(
        "✓ Web page downloaded. Extracting readable text...",
        event="stage_completed",
        stage="url_fetch",
        url=url,
    )

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    # Remove non-content elements.
    for element in soup(
        [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "noscript",
        ]
    ):
        element.extract()

    text = soup.get_text(
        separator="\n"
    )

    lines = (
        line.strip()
        for line in text.splitlines()
    )

    clean_text = "\n".join(
        chunk
        for chunk in lines
        if chunk
    )

    if not clean_text.strip():
        emit_ingest_event(
            "✗ Could not extract readable text from this web link.",
            event="stage_error",
            stage="url_fetch",
            url=url,
        )
        raise ValueError(
            "Could not extract readable text from this web link."
        )

    doc = Document(
        page_content=clean_text,
        metadata={
            "source": url,
            "page": 1,
        },
    )

    # -------------------------------------------------------
    # CHUNKING
    # -------------------------------------------------------

    emit_ingest_event(
        "✂️ Splitting web content into searchable chunks...",
        event="stage_started",
        stage="chunking",
        url=url,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(
        [doc]
    )

    if not chunks:
        emit_ingest_event(
            "✗ No text chunks could be extracted from this URL.",
            event="stage_error",
            stage="chunking",
            url=url,
        )
        raise ValueError(
            "No text chunks could be extracted from this URL."
        )

    emit_ingest_event(
        f"✓ Created {len(chunks)} chunks from the web page.",
        event="stage_completed",
        stage="chunking",
        url=url,
        chunks=len(chunks),
    )

    # -------------------------------------------------------
    # EMBEDDINGS
    # -------------------------------------------------------

    emit_ingest_event(
        "🧠 Preparing embedding model for web content...",
        event="stage_started",
        stage="embeddings",
        url=url,
        chunks=len(chunks),
    )

    try:
        embeddings = _get_embeddings_with_progress(source_name=url)
    except Exception as err:
        emit_ingest_event(
            "✗ Could not load the embedding model.",
            event="stage_error",
            stage="embeddings",
            url=url,
            error=str(err),
        )
        raise ValueError(
            f"Could not load the embedding model: {err}"
        )

    emit_ingest_event(
        "🧠 Creating embeddings for web content...",
        event="stage_started",
        stage="vectorization",
        url=url,
        chunks=len(chunks),
    )

    # -------------------------------------------------------
    # FAISS + METADATA (locked, all-or-nothing)
    # -------------------------------------------------------

    try:
        with _faiss_lock():
            _save_chunks_to_faiss(
                chunks,
                embeddings,
                source_name=url,
            )

            registry = _load_metadata_registry()

            for chunk in chunks:

                registry.append(
                    {
                        "source": url,
                        "page": 1,
                        "text": chunk.page_content,
                        "user_id": chunk.metadata.get("user_id"),
                    }
                )

            _save_metadata_registry(
                registry
            )

    except Timeout as exc:
        emit_ingest_event(
            "✗ Could not update the knowledge base: busy with another file.",
            event="stage_error",
            stage="vectorization",
            url=url,
        )
        raise ValueError(
            "The knowledge base is busy processing another file. "
            "Please try again in a moment."
        ) from exc
    except Exception as err:
        emit_ingest_event(
            "✗ Could not update the knowledge base.",
            event="stage_error",
            stage="vectorization",
            url=url,
            error=str(err),
        )
        raise

    emit_ingest_event(
        f"✓ FAISS updated with {len(chunks)} web-content chunks.",
        event="stage_completed",
        stage="vectorization",
        url=url,
        chunks=len(chunks),
    )

    emit_ingest_event(
        "💾 Web-page metadata saved.",
        event="stage_completed",
        stage="metadata",
        url=url,
    )

    emit_ingest_event(
        f"✅ Web page is ready for RAG search.",
        event="ingestion_completed",
        stage="complete",
        url=url,
        chunks=len(chunks),
    )

    return len(chunks)


# ---------------------------------------------------------------------------
# INDEXED SOURCES
# ---------------------------------------------------------------------------

def get_indexed_sources() -> List[Dict[str, Any]]:
    """
    Return details for files currently stored in data/upload.

    Web URLs are represented through the metadata registry.
    """

    sources = []

    if os.path.exists(UPLOAD_DIR):

        for fname in os.listdir(
            UPLOAD_DIR
        ):

            fpath = os.path.join(
                UPLOAD_DIR,
                fname,
            )

            if os.path.isfile(fpath):

                size_kb = (
                    os.path.getsize(fpath)
                    / 1024
                )

                sources.append(
                    {
                        "name": fname,
                        "type": "File",
                        "size": f"{size_kb:.1f} KB",
                    }
                )

    # Add URLs from metadata.
    registry = _load_metadata_registry()

    existing_names = {
        item["name"]
        for item in sources
    }

    url_sources = set()

    for entry in registry:

        source = entry.get(
            "source",
            "",
        )

        if isinstance(
            source,
            str,
        ) and source.startswith(
            (
                "http://",
                "https://",
            )
        ):

            url_sources.add(source)

    for url in sorted(url_sources):

        if url not in existing_names:

            sources.append(
                {
                    "name": url,
                    "type": "URL",
                    "size": "Indexed",
                }
            )

    return sources


# ---------------------------------------------------------------------------
# DELETE INDEXED SOURCE
# ---------------------------------------------------------------------------

def delete_indexed_source(
    source_name: str,
) -> bool:
    """
    Remove a file or URL from the knowledge base.
    """

    clean_source = (
        source_name.strip()
        if isinstance(source_name, str)
        else ""
    )

    if not clean_source:
        return False

    # File source.
    if not clean_source.startswith(
        (
            "http://",
            "https://",
        )
    ):

        fpath = os.path.join(
            UPLOAD_DIR,
            os.path.basename(
                clean_source
            ),
        )

        if os.path.exists(fpath):
            os.remove(fpath)

        return delete_document(
            os.path.basename(
                clean_source
            )
        )

    # URL source.
    try:
        with _faiss_lock():

            registry = _load_metadata_registry()

            remaining_entries = [
                item
                for item in registry
                if item.get("source") != clean_source
            ]

            if len(remaining_entries) == len(
                registry
            ):
                return False

            if not remaining_entries:

                if os.path.exists(INDEX_PATH):
                    os.remove(INDEX_PATH)

                idx_pkl = os.path.join(
                    FAISS_DIR,
                    "index.pkl",
                )

                if os.path.exists(idx_pkl):
                    os.remove(idx_pkl)

                if os.path.exists(METADATA_PATH):
                    os.remove(METADATA_PATH)

                return True

            rebuild_docs = [
                Document(
                    page_content=item["text"],
                    metadata={
                        "source": item["source"],
                        "page": item.get(
                            "page",
                            0,
                        ),
                    },
                )
                for item in remaining_entries
            ]

            embeddings = _get_embeddings_with_progress(source_name=clean_source)

            vectorstore = FAISS.from_documents(
                rebuild_docs,
                embeddings,
            )

            vectorstore.save_local(
                FAISS_DIR
            )

            _save_metadata_registry(
                remaining_entries
            )

            return True

    except Timeout:
        raise ValueError(
            "The knowledge base is busy processing another file. "
            "Please try again in a moment."
        )