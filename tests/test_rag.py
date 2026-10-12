import os
import pytest
from rag.ingest import (
    ingest_file,
    ingest_multiple_files,
    search_documents,
    list_indexed_documents,
    delete_document,
    FAISS_DIR,
)


@pytest.fixture(autouse=True)
def setup_rag_dir(tmp_path, monkeypatch):
    test_faiss_dir = str(tmp_path / "faiss")
    monkeypatch.setattr("rag.ingest.FAISS_DIR", test_faiss_dir)
    monkeypatch.setattr("rag.ingest.INDEX_PATH", os.path.join(test_faiss_dir, "index.faiss"))
    monkeypatch.setattr("rag.ingest.METADATA_PATH", os.path.join(test_faiss_dir, "metadata.pkl"))
    yield


def test_ingest_and_search_single_file(tmp_path):
    sample_file = tmp_path / "data" / "knowledge.txt"
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    sample_file.write_text("Project Alpha code name is Omega-99. System port is 9090.")

    chunks = ingest_file(str(sample_file))
    assert chunks > 0

    results = search_documents("What is the system port?", k=1)
    assert len(results) > 0
    assert "9090" in results[0].page_content


def test_ingest_multiple_files(tmp_path):
    dir_path = tmp_path / "data"
    dir_path.mkdir(parents=True, exist_ok=True)

    file1 = dir_path / "doc1.txt"
    file2 = dir_path / "doc2.txt"

    file1.write_text("Python is a dynamic programming language.")
    file2.write_text("SQLite is an embedded relational database.")

    summary = ingest_multiple_files([str(file1), str(file2)])
    assert len(summary["successful"]) == 2
    assert len(summary["failed"]) == 0

    indexed_docs = list_indexed_documents()
    assert len(indexed_docs) == 2


def test_delete_document_and_rebuild(tmp_path):
    dir_path = tmp_path / "data"
    dir_path.mkdir(parents=True, exist_ok=True)

    file1 = dir_path / "keep.txt"
    file2 = dir_path / "remove.txt"

    file1.write_text("Keep this document content in FAISS index.")
    file2.write_text("Remove this document content from index completely.")

    ingest_multiple_files([str(file1), str(file2)])

    removed = delete_document("remove.txt")
    assert removed is True

    indexed = list_indexed_documents()
    sources = [item["source"] for item in indexed]

    assert "keep.txt" in sources
    assert "remove.txt" not in sources