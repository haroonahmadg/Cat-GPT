import importlib
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def image_api(monkeypatch):
    tmp_path = Path(tempfile.mkdtemp(prefix="ai-images-"))
    monkeypatch.setenv("UI_DB_PATH", str(tmp_path / "ui.sqlite"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "upload"))
    monkeypatch.setenv("SESSION_SECRET", "image-test-secret")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id.apps.googleusercontent.com")
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite"))

    import agent.tools as agent_tools
    import api.main as api_main

    monkeypatch.setattr(agent_tools, "GENERATED_IMAGE_DIR", tmp_path / "generated")
    importlib.reload(api_main)
    with TestClient(api_main.app) as client:
        yield client, api_main, agent_tools


def _create_account(client: TestClient, email: str) -> str:
    response = client.post(
        "/auth/register",
        json={"email": email, "password": "Password123", "name": "Image User"},
    )
    assert response.status_code == 200, response.text
    return response.json()["token"]


def test_image_library_is_account_scoped(image_api):
    client, api_main, agent_tools = image_api
    first_token = _create_account(client, "first@example.com")
    second_token = _create_account(client, "second@example.com")
    first_user = api_main.verify_session_token(first_token)
    second_user = api_main.verify_session_token(second_token)
    first_name = f"{uuid.uuid4().hex[:16]}.png"
    second_name = f"{uuid.uuid4().hex[:16]}.png"
    first_path = agent_tools.get_generated_image_dir(first_user, create=True) / first_name
    second_path = agent_tools.get_generated_image_dir(second_user, create=True) / second_name
    first_path.write_bytes(b"first-image")
    second_path.write_bytes(b"second-image")

    first_headers = {"Authorization": f"Bearer {first_token}"}
    second_headers = {"Authorization": f"Bearer {second_token}"}

    first_library = client.get("/images", headers=first_headers)
    second_library = client.get("/images", headers=second_headers)
    assert first_library.status_code == 200
    assert [image["filename"] for image in first_library.json()] == [first_name]
    assert [image["filename"] for image in second_library.json()] == [second_name]

    assert client.get(f"/generated-images/{first_name}", headers=first_headers).content == b"first-image"
    assert client.get(f"/generated-images/{first_name}", headers=second_headers).status_code == 404
    assert client.get(f"/generated-images/{first_name}").status_code == 401


def test_library_includes_account_uploaded_files_and_media_is_private(image_api):
    client, api_main, _ = image_api
    token = _create_account(client, "library@example.com")
    other_token = _create_account(client, "other-library@example.com")
    user_id = api_main.verify_session_token(token)
    folder = "thread-folder"
    user_folder = api_main._user_upload_dir(user_id) / folder
    user_folder.mkdir(parents=True)
    filename = "document_123456.txt"
    (user_folder / filename).write_text("account document", encoding="utf-8")

    response = client.get(
        "/library/media",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.text
    assert response.json()[0]["filename"] == filename
    assert response.json()[0]["folder"] == folder

    item = client.get(
        f"/library/media/{folder}/{filename}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert item.status_code == 200
    assert item.text == "account document"
    assert client.get(
        f"/library/media/{folder}/{filename}",
        headers={"Authorization": f"Bearer {other_token}"},
    ).status_code == 404


def test_image_generation_uses_authenticated_account(image_api, monkeypatch):
    client, api_main, agent_tools = image_api
    token = _create_account(client, "generate@example.com")
    user_id = api_main.verify_session_token(token)
    filename = f"{uuid.uuid4().hex[:16]}.png"
    calls = []

    def fake_generate(args):
        calls.append(agent_tools.active_user_id.get())
        image_dir = agent_tools.get_generated_image_dir(
            agent_tools.active_user_id.get(), create=True
        )
        (image_dir / filename).write_bytes(b"generated-image")
        return f"{agent_tools.GENERATED_IMAGE_MARKER}{filename}"

    monkeypatch.setattr(api_main, "generate_image", SimpleNamespace(invoke=fake_generate))
    response = client.post(
        "/images/generate",
        headers={"Authorization": f"Bearer {token}"},
        json={"prompt": "A cat in a garden"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["filename"] == filename
    assert calls == [user_id]


def test_generate_image_uses_free_provider_and_saves_valid_image(image_api, monkeypatch):
    _, _, agent_tools = image_api
    import base64
    from io import BytesIO
    from PIL import Image

    monkeypatch.setenv("HF_API_KEY", "test-key")
    user_token = agent_tools.active_user_id.set("free-provider-user")
    image_bytes = BytesIO()
    Image.new("RGB", (2, 2), color="red").save(image_bytes, format="PNG")
    calls = []

    class FakePostResponse:
        headers = {"content-type": "application/json"}

        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                import httpx

                request = httpx.Request("POST", agent_tools.HF_IMAGE_URL)
                response = httpx.Response(self.status_code, request=request)
                raise httpx.HTTPStatusError(
                    f"HTTP {self.status_code}",
                    request=request,
                    response=response,
                )

        def json(self):
            return self._payload

    def fake_post(url, **kwargs):
        calls.append(("post", url, kwargs))
        return FakePostResponse({
            "data": [{"b64_json": base64.b64encode(image_bytes.getvalue()).decode()}]
        })

    monkeypatch.setattr(agent_tools.httpx, "post", fake_post)
    monkeypatch.setattr(agent_tools, "emit_tool_event", lambda *args, **kwargs: None)
    try:
        result = agent_tools.generate_image.func("A cat in a garden")
    finally:
        agent_tools.active_user_id.reset(user_token)

    assert result.startswith(agent_tools.GENERATED_IMAGE_MARKER)
    assert calls[0][1] == agent_tools.HF_IMAGE_URL
    assert calls[0][2]["json"]["model"] == agent_tools.HF_IMAGE_MODELS[0]
    assert calls[0][2]["json"]["prompt"].startswith(agent_tools.HF_IMAGE_STYLE_PREFIX)
    image_path = (
        agent_tools.get_generated_image_dir("free-provider-user")
        / f"{result.removeprefix(agent_tools.GENERATED_IMAGE_MARKER)}"
    )
    with Image.open(image_path) as saved:
        assert saved.size == (2, 2)


def test_generate_image_retries_schnell_after_dev_failure(image_api, monkeypatch):
    _, _, agent_tools = image_api
    import base64
    from io import BytesIO
    from PIL import Image
    import httpx

    monkeypatch.setenv("HF_API_KEY", "test-key")
    user_token = agent_tools.active_user_id.set("fallback-image-user")
    image_bytes = BytesIO()
    Image.new("RGB", (2, 2), color="blue").save(image_bytes, format="PNG")
    requested_models = []

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __init__(self, status_code):
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                request = httpx.Request("POST", agent_tools.HF_IMAGE_URL)
                response = httpx.Response(self.status_code, request=request)
                raise httpx.HTTPStatusError(
                    f"HTTP {self.status_code}",
                    request=request,
                    response=response,
                )

        def json(self):
            return {
                "data": [{
                    "b64_json": base64.b64encode(image_bytes.getvalue()).decode()
                }]
            }

    def fake_post(url, **kwargs):
        model = kwargs["json"]["model"]
        requested_models.append(model)
        return FakeResponse(429 if model == agent_tools.HF_IMAGE_MODELS[0] else 200)

    monkeypatch.setattr(agent_tools.httpx, "post", fake_post)
    monkeypatch.setattr(agent_tools, "emit_tool_event", lambda *args, **kwargs: None)
    try:
        result = agent_tools.generate_image.func("A blue cat")
    finally:
        agent_tools.active_user_id.reset(user_token)

    assert result.startswith(agent_tools.GENERATED_IMAGE_MARKER)
    assert requested_models == list(agent_tools.HF_IMAGE_MODELS)


def test_generate_image_returns_plain_message_when_both_models_fail(image_api, monkeypatch):
    _, _, agent_tools = image_api
    import httpx

    monkeypatch.setenv("HF_API_KEY", "test-key")
    requested_models = []

    def fake_post(url, **kwargs):
        requested_models.append(kwargs["json"]["model"])
        request = httpx.Request("POST", url)
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError("Not found", request=request, response=response)

    def fake_get(url, **kwargs):
        request = httpx.Request("GET", url)
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError("Not found", request=request, response=response)

    monkeypatch.setattr(agent_tools.httpx, "post", fake_post)
    monkeypatch.setattr(agent_tools.httpx, "get", fake_get)
    monkeypatch.setattr(agent_tools, "emit_tool_event", lambda *args, **kwargs: None)

    result = agent_tools.generate_image.func("A cat")

    assert requested_models == list(agent_tools.HF_IMAGE_MODELS)
    assert result == "Image generation error: image service returned status 404."


def test_generate_image_falls_back_to_pollinations_after_hugging_face_404(
    image_api, monkeypatch
):
    _, _, agent_tools = image_api
    import httpx
    from io import BytesIO
    from PIL import Image

    monkeypatch.setenv("HF_API_KEY", "test-key")
    user_token = agent_tools.active_user_id.set("pollinations-fallback-user")
    image_bytes = BytesIO()
    Image.new("RGB", (2, 2), color="green").save(image_bytes, format="JPEG")
    requested_models = []
    fallback_calls = []

    def fake_post(url, **kwargs):
        requested_models.append(kwargs["json"]["model"])
        request = httpx.Request("POST", url)
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError("Not found", request=request, response=response)

    class FakeImageResponse:
        headers = {"content-type": "image/jpeg"}
        content = image_bytes.getvalue()

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        fallback_calls.append((url, kwargs))
        return FakeImageResponse()

    monkeypatch.setattr(agent_tools.httpx, "post", fake_post)
    monkeypatch.setattr(agent_tools.httpx, "get", fake_get)
    monkeypatch.setattr(agent_tools, "emit_tool_event", lambda *args, **kwargs: None)
    try:
        result = agent_tools.generate_image.func("A green cat")
    finally:
        agent_tools.active_user_id.reset(user_token)

    assert result.startswith(agent_tools.GENERATED_IMAGE_MARKER)
    assert requested_models == list(agent_tools.HF_IMAGE_MODELS)
    assert fallback_calls[0][0].startswith(agent_tools.POLLINATIONS_IMAGE_URL)
    saved_path = (
        agent_tools.get_generated_image_dir("pollinations-fallback-user")
        / result.removeprefix(agent_tools.GENERATED_IMAGE_MARKER)
    )
    with Image.open(saved_path) as saved:
        assert saved.size == (2, 2)


def test_generate_image_without_hugging_face_key_is_plain_and_skips_request(
    image_api, monkeypatch
):
    _, _, agent_tools = image_api
    monkeypatch.delenv("HF_API_KEY", raising=False)
    monkeypatch.delenv("HF_API_TOKEN", raising=False)
    monkeypatch.setattr(
        agent_tools.httpx,
        "post",
        lambda *args, **kwargs: pytest.fail("HTTP request should not be made without a key"),
    )

    import httpx

    def failed_pollinations(url, **kwargs):
        request = httpx.Request("GET", url)
        response = httpx.Response(503, request=request)
        raise httpx.HTTPStatusError("Unavailable", request=request, response=response)

    monkeypatch.setattr(agent_tools.httpx, "get", failed_pollinations)
    monkeypatch.setattr(agent_tools, "emit_tool_event", lambda *args, **kwargs: None)

    result = agent_tools.generate_image.func("A cat")

    assert result == "Image generation error: image service returned status 503."


@pytest.mark.parametrize(
    ("prompt", "expected_status"),
    [("   ", 400), ("x" * 501, 422)],
)
def test_image_generation_rejects_invalid_prompt(image_api, prompt, expected_status):
    client, _, _ = image_api
    token = _create_account(client, f"invalid-{len(prompt)}@example.com")
    response = client.post(
        "/images/generate",
        headers={"Authorization": f"Bearer {token}"},
        json={"prompt": prompt},
    )
    assert response.status_code == expected_status
