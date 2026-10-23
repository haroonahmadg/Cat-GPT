import importlib
import io
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def voice_client(monkeypatch, tmp_path):
    monkeypatch.setenv("UI_DB_PATH", str(tmp_path / "ui.sqlite"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "upload"))
    monkeypatch.setenv("SESSION_SECRET", "test-voice-session-secret")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id.apps.googleusercontent.com")
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key")
    import api.main as api_main

    importlib.reload(api_main)
    with TestClient(api_main.app) as client:
        yield client, api_main


import uuid


def _auth_headers(client):
    email = f"voice-{uuid.uuid4().hex[:8]}@example.com"
    reg = client.post(
        "/auth/register",
        json={
            "email": email,
            "password": "Password123",
            "name": "Voice User",
        },
    )
    assert reg.status_code == 200
    token = reg.json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_voice_transcribe_unauth(voice_client):
    client, _ = voice_client
    files = {"audio": ("test.webm", b"audio data", "audio/webm")}
    res = client.post("/voice/transcribe", files=files)
    assert res.status_code == 401


def test_voice_transcribe_empty(voice_client):
    client, _ = voice_client
    headers = _auth_headers(client)
    files = {"audio": ("test.webm", b"", "audio/webm")}
    res = client.post("/voice/transcribe", files=files, headers=headers)
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_voice_transcribe_bad_mime(voice_client):
    client, _ = voice_client
    headers = _auth_headers(client)
    files = {"audio": ("test.txt", b"plain text data", "text/plain")}
    res = client.post("/voice/transcribe", files=files, headers=headers)
    assert res.status_code == 400
    assert "unsupported" in res.json()["detail"].lower()


def test_voice_transcribe_oversize(voice_client):
    client, _ = voice_client
    headers = _auth_headers(client)
    large_data = b"x" * (10 * 1024 * 1024 + 1)
    files = {"audio": ("test.webm", large_data, "audio/webm")}
    res = client.post("/voice/transcribe", files=files, headers=headers)
    assert res.status_code == 400
    assert "limit" in res.json()["detail"].lower() or "10 mb" in res.json()["detail"].lower()


def test_voice_transcribe_success(voice_client, monkeypatch):
    client, api_main = voice_client
    headers = _auth_headers(client)

    import agent.voice as agent_voice
    mock_groq = MagicMock()
    mock_groq.return_value.audio.transcriptions.create.return_value = "Hello, this is a transcribed voice message."
    monkeypatch.setattr(agent_voice, "Groq", mock_groq)

    files = {"audio": ("recording.webm", b"valid-webm-audio-bytes", "audio/webm")}
    res = client.post("/voice/transcribe", files=files, headers=headers)
    assert res.status_code == 200
    assert res.json() == {"text": "Hello, this is a transcribed voice message."}


def test_voice_transcribe_groq_failure(voice_client, monkeypatch):
    client, api_main = voice_client
    headers = _auth_headers(client)

    import agent.voice as agent_voice
    mock_groq = MagicMock()
    mock_groq.return_value.audio.transcriptions.create.side_effect = RuntimeError("Groq API rate limit or outage")
    monkeypatch.setattr(agent_voice, "Groq", mock_groq)

    files = {"audio": ("recording.webm", b"valid-webm-audio-bytes", "audio/webm")}
    res = client.post("/voice/transcribe", files=files, headers=headers)
    assert res.status_code in (500, 502)
    assert res.json()["detail"] == "An unexpected upstream error occurred."
    assert "Groq API rate limit or outage" not in res.json()["detail"]
