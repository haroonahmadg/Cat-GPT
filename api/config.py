"""
Configuration for the FastAPI backend.

NOTE ON DUPLICATION WITH app.py:
app.py (the existing Streamlit frontend) defines its own copies of the
upload-validation constants and the UI/thread database path below. app.py
cannot be imported from here, because importing it executes Streamlit-only
calls (st.set_page_config, st.sidebar, etc.) outside of a Streamlit runtime,
which fails. These values are therefore intentionally duplicated as the
single source of truth for the API, rather than refactoring them out of
app.py — app.py is left completely untouched per the existing-architecture
constraint. If both frontends need to share these values later, extracting
them into a small shared module both files import would be the natural
follow-up, but that is out of scope for this phase.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Same on-disk locations app.py already uses, so both frontends see the
# same threads, pins, deletions, and uploaded files.
UI_DB_PATH = Path(
    os.getenv("UI_DB_PATH", str(DATA_DIR / "ui.sqlite"))
)

UPLOAD_DIR = Path(
    os.getenv("UPLOAD_DIR", str(DATA_DIR / "upload"))
)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGE_MB = float(os.getenv("MAX_IMAGE_MB", "10"))
MAX_FILE_MB = float(os.getenv("MAX_FILE_MB", "25"))
MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", "10"))
MAX_IMAGES_PER_REQUEST = int(os.getenv("MAX_IMAGES_PER_REQUEST", "3"))
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "8000"))

ALLOWED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
}

ALLOWED_FILE_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".xlsx",
    ".xls",
}

# Comma-separated list of allowed frontend origins for CORS, e.g.
# "http://localhost:3000,https://myapp.example.com"
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGIN", "http://localhost:3000"
    ).split(",")
    if origin.strip()
]

# ============================================================
# IMAGE GENERATION (Hugging Face Inference API)
# ============================================================

HF_API_KEY = os.getenv("HF_API_KEY", "").strip() or os.getenv("HF_API_TOKEN", "").strip()
HF_IMAGE_MODEL = os.getenv(
    "HF_IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"
).strip()

GENERATED_IMAGE_DIR = DATA_DIR / "generated_images"
GENERATED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
# ============================================================
# AUTH (Phase 4a — single shared password, not multi-user accounts)
#
# APP_PASSWORD: the one password that unlocks the whole app. Required —
# the API refuses to start without it, so a deployment can never
# accidentally go live with no password set.
#
# SESSION_SECRET: used to sign session tokens (HMAC), so a token can't
# be forged without knowing this value. Required for the same reason
# as APP_PASSWORD — no silent "auth that doesn't actually check anything"
# fallback.
#
# SESSION_TTL_SECONDS: how long a session token stays valid before the
# person has to log in again. Defaults to 7 days.
# ============================================================




# ============================================================
# AUTH (Google sign-in)
# ============================================================

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
SESSION_SECRET = os.getenv("SESSION_SECRET", "").strip()
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(7 * 24 * 60 * 60)))





# ============================================================
# AUTH SETTINGS  (append this block to the END of api/config.py)
#
# Every value can be overridden in .env; the defaults below apply
# when a variable is not set. auth.py, rate_limit.py and main.py
# read these names from here instead of hardcoding numbers, the
# same way GOOGLE_CLIENT_ID / SESSION_SECRET / SESSION_TTL_SECONDS
# already work.
# ============================================================

import os as _os


def _auth_env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read a whole-number setting from the environment, with validation."""

    raw = _os.getenv(name)

    if raw is None or not raw.strip():
        return default

    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a whole number.") from exc

    if value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}.")

    return value


# Input limits
EMAIL_MAX_LENGTH = _auth_env_int("EMAIL_MAX_LENGTH", 254)
NAME_MAX_LENGTH = _auth_env_int("NAME_MAX_LENGTH", 100)
PASSWORD_MIN_LENGTH = _auth_env_int("PASSWORD_MIN_LENGTH", 8)
PASSWORD_MAX_LENGTH = _auth_env_int("PASSWORD_MAX_LENGTH", 128)

if PASSWORD_MIN_LENGTH > PASSWORD_MAX_LENGTH:
    raise RuntimeError(
        "PASSWORD_MIN_LENGTH cannot be greater than PASSWORD_MAX_LENGTH."
    )

# Brute-force protection (see api/rate_limit.py)
LOGIN_MAX_ATTEMPTS_PER_EMAIL = _auth_env_int("LOGIN_MAX_ATTEMPTS_PER_EMAIL", 5)
LOGIN_MAX_ATTEMPTS_PER_IP = _auth_env_int("LOGIN_MAX_ATTEMPTS_PER_IP", 20)
LOGIN_WINDOW_SECONDS = _auth_env_int("LOGIN_WINDOW_SECONDS", 15 * 60)

REGISTER_MAX_PER_IP = _auth_env_int("REGISTER_MAX_PER_IP", 10)
REGISTER_WINDOW_SECONDS = _auth_env_int("REGISTER_WINDOW_SECONDS", 60 * 60)