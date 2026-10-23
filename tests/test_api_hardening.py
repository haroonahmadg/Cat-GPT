import importlib

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError


@pytest.fixture
def api_client(monkeypatch, tmp_path):
    monkeypatch.setenv("UI_DB_PATH", str(tmp_path / "ui.sqlite"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "upload"))
    monkeypatch.setenv("SESSION_SECRET", "test-api-hardening-secret")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id.apps.googleusercontent.com")
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite"))
    import api.main as api_main

    importlib.reload(api_main)
    with TestClient(api_main.app) as client:
        yield client, api_main


def _auth_headers(client):
    response = client.post(
        "/auth/register",
        json={"email": "hardening@example.com", "password": "Password123"},
    )
    assert response.status_code == 200
    token = response.json()["token"]
    return {"Authorization": "Bearer " + token}


def test_speak_request_limits_text_length():
    from api.schemas import SpeakRequest

    assert len(SpeakRequest(text="x" * 5000).text) == 5000
    with pytest.raises(ValidationError):
        SpeakRequest(text="x" * 5001)


def test_thread_list_returns_server_error_when_checkpoint_store_fails(api_client, monkeypatch):
    client, api_main = api_client
    headers = _auth_headers(client)

    def fail_listing(_user_id, *, raise_errors=False):
        raise OSError("database unavailable")

    monkeypatch.setattr(api_main, "get_all_saved_threads", fail_listing)
    response = client.get("/threads", headers=headers)

    assert response.status_code == 500
    assert response.json()["detail"] == "Could not load conversations right now."
    assert "database unavailable" not in response.text


def test_upload_rejects_oversized_content_length_before_file_read(api_client):
    client, api_main = api_client
    headers = _auth_headers(client)
    max_bytes = int(
        (api_main.MAX_FILE_MB * api_main.MAX_UPLOAD_FILES + 1) * 1024 * 1024
    )
    headers["Content-Length"] = str(max_bytes + 1)

    response = client.post(
        "/files/upload",
        headers=headers,
        data={"thread_id": "thread-1"},
        files=[("files", ("small.txt", b"small", "text/plain"))],
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "The total upload size exceeds the allowed limit."
