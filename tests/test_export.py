import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(monkeypatch, tmp_path):
    monkeypatch.setenv("UI_DB_PATH", str(tmp_path / "ui.sqlite"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "upload"))
    monkeypatch.setenv("SESSION_SECRET", "test-export-session-secret")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id.apps.googleusercontent.com")
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite"))
    import api.main as api_main

    importlib.reload(api_main)
    with TestClient(api_main.app) as client:
        yield client, api_main


def test_export_markdown_requires_owner_and_formats_transcript(api_client, monkeypatch):
    client, api_main = api_client
    from api.store import get_user_by_email

    owner = client.post(
        "/auth/register",
        json={
            "email": "export-owner@example.com",
            "password": "Password123",
            "name": "Export Owner",
        },
    )
    assert owner.status_code == 200, owner.text
    owner_id = get_user_by_email("export-owner@example.com")["user_id"]
    non_owner = client.post(
        "/auth/register",
        json={
            "email": "export-other@example.com",
            "password": "Password123",
            "name": "Other User",
        },
    )
    assert non_owner.status_code == 200, non_owner.text

    monkeypatch.setattr(
        api_main,
        "get_all_saved_threads",
        lambda user_id: {"thread-123": "Original title"} if user_id == owner_id else {},
    )
    monkeypatch.setattr(
        api_main,
        "get_renamed_titles",
        lambda user_id: {"thread-123": "Custom title"},
    )
    monkeypatch.setattr(
        api_main,
        "get_thread_history",
        lambda **kwargs: [
            {"role": "user", "content": "Summarize the report."},
            {
                "role": "assistant",
                "content": "The report says revenue grew.\n\n[source: report.pdf p.3]",
            },
        ],
    )

    response = client.get(
        "/threads/thread-123/export.md",
        headers={"Authorization": f"Bearer {owner.json()['token']}"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.headers["content-disposition"].endswith('filename="chat_thread-1.md"')
    assert response.text.startswith("# Custom title\n\n_Exported ")
    assert "## You\n\nSummarize the report." in response.text
    assert "## Assistant\n\nThe report says revenue grew." in response.text
    assert "[source: report.pdf p.3]" in response.text

    forbidden = client.get(
        "/threads/thread-123/export.md",
        headers={"Authorization": f"Bearer {non_owner.json()['token']}"},
    )
    assert forbidden.status_code == 404
    assert "Conversation not found" in forbidden.json()["detail"]
