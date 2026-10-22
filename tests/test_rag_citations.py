import importlib
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(monkeypatch, tmp_path):
    monkeypatch.setenv("UI_DB_PATH", str(tmp_path / "ui.sqlite"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "upload"))
    monkeypatch.setenv("SESSION_SECRET", "test-citations-session-secret")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id.apps.googleusercontent.com")
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite"))
    import api.main as api_main

    importlib.reload(api_main)
    with TestClient(api_main.app) as client:
        yield client, api_main


def test_search_pdf_returns_scoped_source_metadata(tmp_path, monkeypatch):
    import rag.search as rag_search_module

    faiss_dir = tmp_path / "faiss"
    faiss_dir.mkdir()
    (faiss_dir / "index.faiss").touch()
    (faiss_dir / "index.pkl").touch()
    monkeypatch.setattr(rag_search_module, "FAISS_DIR", str(faiss_dir))

    documents = [
        SimpleNamespace(
            page_content="Private report content for Alice.",
            metadata={
                "source": "report.pdf",
                "page": 2,
                "chunk_id": 4,
                "user_id": "alice",
            },
        ),
        SimpleNamespace(
            page_content="Private report content for Bob.",
            metadata={
                "source": "report.pdf",
                "page": 2,
                "chunk_id": 4,
                "user_id": "bob",
            },
        ),
    ]

    class FakeVectorStore:
        def similarity_search(self, _query, k):
            assert k >= 5
            return documents

    monkeypatch.setattr(rag_search_module, "HuggingFaceEmbeddings", lambda **_: object())
    monkeypatch.setattr(
        rag_search_module.FAISS,
        "load_local",
        lambda *args, **kwargs: FakeVectorStore(),
    )

    results = rag_search_module.search_pdf("report", user_id="alice")

    assert len(results) == 1
    assert results[0]["content"] == "Private report content for Alice."
    assert results[0]["metadata"] == {
        "source": "report.pdf",
        "page": 2,
        "chunk_id": 4,
        "user_id": "alice",
        "preview": "Private report content for Alice.",
    }


def test_stream_emits_deduplicated_citations_once(api_client, monkeypatch):
    client, api_main = api_client
    registered = client.post(
        "/auth/register",
        json={
            "email": "citation-owner@example.com",
            "password": "Password123",
            "name": "Citation Owner",
        },
    )
    assert registered.status_code == 200, registered.text

    citation = {
        "file": "report.pdf",
        "page": 3,
        "preview": "Revenue increased during the quarter.",
    }

    def fake_stream_agent(**_kwargs):
        yield "custom", {"event": "citation", "citation": citation}
        yield "custom", {"event": "citation", "citation": citation}
        yield "custom", {
            "event": "citation",
            "citation": {**citation, "page": 4, "preview": "Operating costs decreased."},
        }
        yield "updates", {"assistant": {"messages": [{"role": "assistant", "content": "Summary."}]}}

    monkeypatch.setattr(api_main, "stream_agent", fake_stream_agent)
    response = client.post(
        "/threads/thread-citations/messages/stream",
        json={"content": "Summarize the report."},
        headers={"Authorization": f"Bearer {registered.json()['token']}"},
    )

    assert response.status_code == 200
    frames = [frame for frame in response.text.split("\n\n") if frame]
    citation_frames = [frame for frame in frames if frame.startswith("event: citations")]
    assert len(citation_frames) == 1
    citation_payload = json.loads(citation_frames[0].split("data: ", 1)[1])
    assert citation_payload["items"] == [
        citation,
        {**citation, "page": 4, "preview": "Operating costs decreased."},
    ]
    assert frames[-1] == "event: done\ndata: {}"


def test_rag_search_emits_compact_citation_marker(monkeypatch):
    import agent.tools as agent_tools

    monkeypatch.setattr(
        agent_tools,
        "search_pdf",
        lambda *_args, **_kwargs: [
            {
                "content": "The report states revenue rose.",
                "metadata": {
                    "source": "report.pdf",
                    "page": 2,
                    "preview": "Revenue rose across all regions.",
                },
            }
        ],
    )
    events = []
    monkeypatch.setattr(agent_tools, "emit_tool_event", lambda *args, **kwargs: events.append(kwargs))

    result = agent_tools.rag_search.invoke({"query": "What happened to revenue?"})

    assert "[source: report.pdf p.3]" in result
    citation_events = [event for event in events if event.get("event") == "citation"]
    assert citation_events == [
        {
            "event": "citation",
            "tool": "rag_search",
            "citation": {
                "file": "report.pdf",
                "page": 3,
                "preview": "Revenue rose across all regions.",
            },
        }
    ]
