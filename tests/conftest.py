import sys

import pytest


@pytest.fixture(autouse=True)
def reset_test_state(monkeypatch, tmp_path_factory):
    # Use a unique temp working directory per test run, then purge any
    # previously imported api/agent modules so they re-read these env values.
    test_dir = tmp_path_factory.mktemp("catgpt-test")
    monkeypatch.setenv("UI_DB_PATH", str(test_dir / "ui.sqlite"))
    monkeypatch.setenv("UPLOAD_DIR", str(test_dir / "upload"))
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id.apps.googleusercontent.com")
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(test_dir / "checkpoints.sqlite"))

    for module_name in (
        "api.config",
        "api.store",
        "api.auth",
        "api.rate_limit",
        "api.main",
        "agent.graph",
        "agent.tools",
    ):
        sys.modules.pop(module_name, None)

    yield

    for module_name in (
        "api.config",
        "api.store",
        "api.auth",
        "api.rate_limit",
        "api.main",
        "agent.graph",
        "agent.tools",
    ):
        sys.modules.pop(module_name, None)
