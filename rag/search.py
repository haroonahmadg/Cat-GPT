import os
import pickle

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


FAISS_DIR = "data/faiss"
METADATA_PATH = os.path.join(FAISS_DIR, "metadata.pkl")


def search_pdf(
    query: str,
    k: int = 4,
    source_filter: list[str] | None = None,user_id: str | None = None
) -> list:
    """
    Search the local FAISS index and return the most relevant document
    chunks.

    source_filter: optional list of filenames (as stored in each chunk's
    metadata["source"] by rag/ingest.py). When given, only chunks from
    those specific files are considered — this is what lets a caller
    scope a search to "the files attached to this conversation" instead
    of matching against every document ever ingested. When omitted,
    behavior is unchanged from before (searches the whole index).
    """

    if not isinstance(query, str):
        raise ValueError("Search query must be a string.")

    query = query.strip()

    if not query:
        raise ValueError("Search query cannot be empty.")

    if len(query) > 500:
        raise ValueError(
            "Search query is too long. Maximum length is 500 characters."
        )

    if not isinstance(k, int) or isinstance(k, bool):
        raise ValueError("k must be an integer.")

    if k < 1 or k > 10:
        raise ValueError("k must be between 1 and 10.")

    if source_filter is not None:
        if not isinstance(source_filter, list) or not all(
            isinstance(item, str) for item in source_filter
        ):
            raise ValueError("source_filter must be a list of filenames.")

    index_file = os.path.join(
        FAISS_DIR,
        "index.faiss",
    )

    index_pickle = os.path.join(
        FAISS_DIR,
        "index.pkl",
    )

    if not os.path.isfile(index_file) or not os.path.isfile(index_pickle):
        raise FileNotFoundError(
            "FAISS index not found. Ingest a PDF before searching."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    if source_filter:
        allowed = set(source_filter)

        # Over-fetch before filtering, since FAISS's similarity_search
        # ranks first and only then would filter — pulling k candidates
        # per allowed source keeps enough results to still return up to
        # k after narrowing to the allowed filenames.
        raw_k = min(k * max(len(allowed), 1) * 3, 50)

        candidates = vectorstore.similarity_search(query, k=raw_k)

        documents = [
            doc
            for doc in candidates
            if doc.metadata.get("source") in allowed
        ][:k]
    else:
        documents = vectorstore.similarity_search(query, k=k)

    results = []

    for document in documents:
        results.append(
            {
                "content": document.page_content,
                "metadata": document.metadata,
            }
        )

    return results


def get_document_preview(
    source_filter: list[str],
    k: int = 4,
    user_id: str | None = None,
) -> list:
    """
    Return the first k chunks of the given file(s), in original document
    order (not similarity-ranked).

    Used as a fallback for generic requests like "summarize this file"
    or "tell me about the attached document" — a vague query often has
    poor semantic overlap with any single chunk, so similarity_search
    can legitimately return nothing even though the file is right
    there. This bypasses relevance ranking entirely and just returns
    the start of the document(s), which is what a generic "describe
    this" request actually wants.

    Reads the same metadata registry rag/ingest.py writes (chunk order
    there matches insertion order, which matches document order — see
    ingest.py's _register_chunks, which appends chunks for a file in
    the order the text splitter produced them).
    """

    if not isinstance(source_filter, list) or not source_filter:
        raise ValueError("source_filter must be a non-empty list of filenames.")

    if not all(isinstance(item, str) for item in source_filter):
        raise ValueError("source_filter must be a list of filenames.")

    if not isinstance(k, int) or isinstance(k, bool):
        raise ValueError("k must be an integer.")

    if k < 1 or k > 20:
        raise ValueError("k must be between 1 and 20.")

    if not os.path.isfile(METADATA_PATH):
        raise FileNotFoundError(
            "No document metadata found. Ingest a file before requesting a preview."
        )

    try:
        with open(METADATA_PATH, "rb") as f:
            registry = pickle.load(f)
    except Exception as exc:
        raise ValueError(f"Could not read document metadata: {exc}") from exc

    if not isinstance(registry, list):
        return []

    allowed = set(source_filter)

    matching = [
        entry
        for entry in registry
        if isinstance(entry, dict) and entry.get("source") in allowed and (user_id is None or entry.get("user_id") == user_id)
    ][:k]

    results = []

    for entry in matching:
        text = (entry.get("text") or "").strip()

        if not text:
            continue

        results.append(
            {
                "content": text,
                "metadata": {
                    "source": entry.get("source"),
                    "page": entry.get("page", 0),
                },
            }
        )

    return results