import importlib
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="ai-auth-")
    tmp_path = Path(tmpdir)
    monkeypatch.setenv("UI_DB_PATH", str(tmp_path / "ui.sqlite"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "upload"))
    monkeypatch.setenv("SESSION_SECRET", "test-secret-value")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id.apps.googleusercontent.com")
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite"))

    import api.main as api_main
    importlib.reload(api_main)

    with TestClient(api_main.app) as client:
        yield client, api_main


def test_register_login_and_me(api_client):
    client, _ = api_client
    response = client.post(
        "/auth/register",
        json={"email": "user@example.com", "password": "Password123", "name": "User One"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["email"] == "user@example.com"
    assert payload["has_password"] is True
    assert payload["token"]

    me = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {payload['token']}"},
    )
    assert me.status_code == 200, me.text
    assert me.json()["email"] == "user@example.com"
    assert me.json()["has_password"] is True

    login = client.post(
        "/auth/login",
        json={"email": "user@example.com", "password": "Password123"},
    )
    assert login.status_code == 200, login.text
    assert login.json()["email"] == "user@example.com"


def test_google_login_rejected_for_invalid_token(api_client, monkeypatch):
    client, main_module = api_client

    def boom(_credential: str):
        raise ValueError("bad google credential")

    monkeypatch.setattr(main_module, "verify_google_credential", boom)

    response = client.post(
        "/auth/google",
        json={"credential": "invalid"},
    )
    assert response.status_code == 401
    assert "bad google credential" in response.json()["detail"]


def test_google_identity_reuses_existing_email_account(api_client):
    _, _ = api_client
    from api.store import create_password_user, get_or_create_user, get_user_by_email

    password_user = create_password_user(
        "person@example.com",
        "Existing Person",
        "existing-password-hash",
    )

    google_user = get_or_create_user(
        "google-subject-123",
        "PERSON@example.com",
        "Google Person",
    )
    stored_user = get_user_by_email("person@example.com")

    assert google_user["user_id"] == password_user["user_id"]
    assert google_user["password_hash"] == "existing-password-hash"
    assert stored_user["user_id"] == password_user["user_id"]
    assert stored_user["email"] == "person@example.com"


def test_google_identity_creates_account_once(api_client):
    _, _ = api_client
    from api.store import get_or_create_user

    created = get_or_create_user("google-subject-456", "new@example.com", "New User")
    signed_in = get_or_create_user("google-subject-456", "new@example.com", "New User")

    assert created["user_id"] == signed_in["user_id"]
    assert created["email"] == "new@example.com"
