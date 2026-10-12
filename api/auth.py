"""
Google sign-in authentication.

A session token is an HMAC-signed string: "<user_id>.<expiry>.<signature>".
Signing with SESSION_SECRET means a token can't be forged without
knowing that value. No server-side session storage is needed — the
token itself carries everything (who, and until when).
"""

from __future__ import annotations

import hashlib
import hmac
import re
import time
from typing import TYPE_CHECKING

from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests

# argon2-cffi hashes passwords. The import is guarded so a missing
# package never stops the whole API from starting: the password
# endpoints report it clearly instead.
try:
    from argon2 import PasswordHasher
except ImportError:
    PasswordHasher = None

if TYPE_CHECKING:
    # Type-only alias: PasswordHasher above may be None at runtime, so
    # it cannot be used directly in a type annotation.
    from argon2 import PasswordHasher as PasswordHasherType

from api.config import (
    EMAIL_MAX_LENGTH,
    GOOGLE_CLIENT_ID,
    NAME_MAX_LENGTH,
    PASSWORD_MAX_LENGTH,
    PASSWORD_MIN_LENGTH,
    SESSION_SECRET,
    SESSION_TTL_SECONDS,
)
from api.store import (
    AccountExistsError,
    create_password_user,
    get_or_create_user,
    get_user_by_email,
    get_user_by_id,
    set_password_hash,
)


def _require_configured() -> None:
    if not GOOGLE_CLIENT_ID:
        raise RuntimeError(
            "GOOGLE_CLIENT_ID is not configured. Set it in your .env file "
            "before starting the API."
        )

    if not SESSION_SECRET:
        raise RuntimeError(
            "SESSION_SECRET is not configured. Set it in your .env file "
            "(any long random string) before starting the API."
        )

    if SESSION_TTL_SECONDS <= 0:
        raise RuntimeError(
            "SESSION_TTL_SECONDS must be a positive number of seconds."
        )


def verify_google_credential(credential: str) -> dict:
    """
    Verify a Google Identity Services credential (an ID token JWT) and
    return the matching app user, creating one on first sign-in.

    Raises ValueError if the credential is invalid or doesn't belong
    to this app's GOOGLE_CLIENT_ID.
    """

    _require_configured()

    if not isinstance(credential, str) or not credential.strip():
        raise ValueError("A Google credential is required.")

    try:
        payload = google_id_token.verify_oauth2_token(
            credential.strip(),
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )
    except Exception as exc:
        raise ValueError(f"Invalid Google credential: {exc}") from exc

    google_id = payload.get("sub")
    email = payload.get("email")
    name = payload.get("name")

    if not google_id or not email:
        raise ValueError("Google credential did not include the expected identity fields.")

    # Linking by email is safe only when Google explicitly confirms ownership.
    if payload.get("email_verified") is not True:
        raise ValueError("Your Google email address is not verified.")

    if not isinstance(name, str):
        name = None

    return get_or_create_user(google_id, email, name)


def _sign(payload: str) -> str:
    _require_configured()

    return hmac.new(
        SESSION_SECRET.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def create_session_token(user_id: str) -> str:
    """Create a signed session token for this user, valid for SESSION_TTL_SECONDS."""

    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("A valid user_id is required.")

    clean_user_id = user_id.strip()

    # "." separates the token's three parts, so it can't appear in the id.
    if "." in clean_user_id:
        raise ValueError("user_id must not contain '.'.")

    expires_at = str(int(time.time()) + SESSION_TTL_SECONDS)
    payload = f"{clean_user_id}.{expires_at}"
    signature = _sign(payload)

    return f"{payload}.{signature}"


def verify_session_token(token: str) -> str | None:
    """
    Returns the user_id if the token is well-formed, correctly signed,
    and not expired — otherwise None. Never raises on malformed input.
    """

    if not isinstance(token, str) or token.count(".") != 2:
        return None

    user_id, expires_at_raw, signature = token.split(".")

    if not user_id or not expires_at_raw or not signature:
        return None

    try:
        expected_signature = _sign(f"{user_id}.{expires_at_raw}")
    except RuntimeError:
        # SESSION_SECRET missing: nothing can be verified, so nothing
        # is accepted. Login reports the configuration problem itself.
        return None

    # Compare as bytes: compare_digest raises TypeError on non-ASCII
    # str input, which a crafted token could otherwise use to cause a
    # server error instead of a clean rejection.
    if not hmac.compare_digest(
        signature.encode("utf-8"),
        expected_signature.encode("utf-8"),
    ):
        return None

    try:
        expires_at = int(expires_at_raw)
    except ValueError:
        return None

    if time.time() >= expires_at:
        return None

    return user_id


# ============================================================
# EMAIL + PASSWORD ACCOUNTS
#
# Password accounts produce the same session token as Google
# sign-in (create_session_token), so the middleware and every
# per-user feature work unchanged.
# ============================================================

# Limits (email/name/password lengths) come from api/config.py, which
# reads them from .env. PASSWORD_MAX_LENGTH also stops absurdly long
# passwords being used to waste CPU on hashing.

_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class InvalidCredentialsError(ValueError):
    """Wrong email/password combination (deliberately not more specific)."""


class UserNotFoundError(Exception):
    """The signed-in user no longer exists."""


def normalize_email(email: str) -> str:
    """Trim and lowercase an email; raise ValueError if it is not valid."""

    if not isinstance(email, str) or not email.strip():
        raise ValueError("An email address is required.")

    clean = email.strip().lower()

    if len(clean) > EMAIL_MAX_LENGTH or not _EMAIL_PATTERN.match(clean):
        raise ValueError("Please enter a valid email address.")

    return clean


def _validate_password(password: str, email: str | None = None) -> str:
    """Return the password unchanged if acceptable; raise ValueError if not."""

    if not isinstance(password, str) or not password:
        raise ValueError("A password is required.")

    if not password.strip():
        raise ValueError("The password cannot be only spaces.")

    if len(password) < PASSWORD_MIN_LENGTH:
        raise ValueError(
            f"The password must be at least {PASSWORD_MIN_LENGTH} characters."
        )

    if len(password) > PASSWORD_MAX_LENGTH:
        raise ValueError(
            f"The password must be at most {PASSWORD_MAX_LENGTH} characters."
        )

    if email and password.strip().lower() == email.strip().lower():
        raise ValueError("The password must be different from your email.")

    return password


def _clean_name(name: str | None) -> str | None:
    if name is None:
        return None

    if not isinstance(name, str):
        raise ValueError("The name must be text.")

    clean = name.strip()

    if len(clean) > NAME_MAX_LENGTH:
        raise ValueError(
            f"The name must be at most {NAME_MAX_LENGTH} characters."
        )

    return clean or None


def _require_hasher() -> "PasswordHasherType":
    if PasswordHasher is None:
        raise RuntimeError(
            "Password accounts need the 'argon2-cffi' package. "
            "Run: pip install argon2-cffi"
        )

    return PasswordHasher()


def hash_password(password: str) -> str:
    return _require_hasher().hash(password)


def verify_password(password: str, stored_hash: str | None) -> bool:
    """
    True only when password matches stored_hash. Never raises on bad
    input or a malformed hash; those simply return False. A missing
    argon2 package still raises RuntimeError, so the problem is not
    mistaken for a wrong password.
    """

    hasher = _require_hasher()

    if not isinstance(password, str) or not isinstance(stored_hash, str):
        return False

    try:
        return bool(hasher.verify(stored_hash, password))
    except Exception:
        return False


_dummy_hash: str | None = None


def _burn_password_check(password: str) -> None:
    """
    Do a real hash verification against a throwaway hash, so a login for
    an unknown email takes about as long as one for a real email and
    cannot be used to discover which emails are registered.
    """

    global _dummy_hash

    if _dummy_hash is None:
        _dummy_hash = hash_password("timing-equalisation-placeholder")

    verify_password(password, _dummy_hash)


def register_with_password(
    email: str,
    password: str,
    name: str | None = None,
) -> dict:
    """
    Create an email + password account and return the user dict.

    Raises ValueError for invalid input and AccountExistsError when the
    email is already registered.
    """

    clean_email = normalize_email(email)
    _validate_password(password, clean_email)
    clean_name = _clean_name(name)

    password_hash = hash_password(password)

    return create_password_user(clean_email, clean_name, password_hash)


def login_with_password(email: str, password: str) -> dict:
    """
    Verify an email + password pair and return the user dict.

    Every failure raises InvalidCredentialsError with the same message,
    whether the email is unknown, belongs to a Google-only account, or
    the password is wrong.
    """

    generic_error = InvalidCredentialsError("Invalid email or password.")

    if (
        not isinstance(email, str)
        or not isinstance(password, str)
        or not email.strip()
        or not password
        or len(email) > EMAIL_MAX_LENGTH
        or len(password) > PASSWORD_MAX_LENGTH
    ):
        raise generic_error

    user = get_user_by_email(email.strip())

    if not user or not user.get("password_hash"):
        _burn_password_check(password)
        raise generic_error

    if not verify_password(password, user["password_hash"]):
        raise generic_error

    # Re-hash with the library's current settings when the stored hash
    # was made with older ones. A failure here must never block a
    # login that was already verified.
    try:
        if _require_hasher().check_needs_rehash(user["password_hash"]):
            set_password_hash(user["user_id"], hash_password(password))
    except Exception:
        pass

    return {
        "user_id": user["user_id"],
        "email": user["email"],
        "name": user["name"],
    }


def set_user_password(
    user_id: str,
    new_password: str,
    current_password: str | None = None,
) -> None:
    """
    Add or change the password of the signed-in user.

    An account that already has a password must prove it with
    current_password first, so a stolen session token alone cannot be
    used to take the account over by changing its password.

    Raises UserNotFoundError, InvalidCredentialsError (wrong current
    password) or ValueError (new password not acceptable).
    """

    user = get_user_by_id(user_id)

    if not user:
        raise UserNotFoundError("This account no longer exists.")

    _validate_password(new_password, user["email"])

    if user.get("password_hash"):
        if not current_password or not verify_password(
            current_password, user["password_hash"]
        ):
            raise InvalidCredentialsError("The current password is incorrect.")

    set_password_hash(user["user_id"], hash_password(new_password))