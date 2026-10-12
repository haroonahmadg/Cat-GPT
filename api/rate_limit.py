"""
Small in-memory attempt limiter used to slow down password guessing
and mass sign-ups.

Limits are tracked per process and reset when the server restarts.
That is fine for a single-process development setup; with several
workers or servers, move this state to a shared store (a database
table or Redis).

Behind a reverse proxy, request.client.host is the proxy's address,
so per-IP limits would apply to everyone at once. Configure the
proxy headers (for example uvicorn --proxy-headers) before relying
on the per-IP limits in production.
"""

from __future__ import annotations

import math
import threading
import time

from api.config import (
    LOGIN_MAX_ATTEMPTS_PER_EMAIL,
    LOGIN_MAX_ATTEMPTS_PER_IP,
    LOGIN_WINDOW_SECONDS,
    REGISTER_MAX_PER_IP,
    REGISTER_WINDOW_SECONDS,
)


class AttemptLimiter:
    """
    Sliding-window counter: allows at most `max_attempts` recorded
    attempts per key within `window_seconds`.
    """

    def __init__(
        self,
        max_attempts: int,
        window_seconds: int,
        max_keys: int = 10_000,
    ) -> None:

        if max_attempts <= 0 or window_seconds <= 0 or max_keys <= 0:
            raise ValueError("Limiter settings must be positive numbers.")

        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.max_keys = max_keys

        self._attempts: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, key: str, now: float) -> list[float]:
        """Attempts for key still inside the window (lock must be held)."""

        recent = [
            stamp
            for stamp in self._attempts.get(key, [])
            if now - stamp < self.window_seconds
        ]

        if recent:
            self._attempts[key] = recent
        else:
            self._attempts.pop(key, None)

        return recent

    def retry_after(self, key: str) -> int:
        """Seconds until this key may try again, or 0 when not limited."""

        now = time.monotonic()

        with self._lock:
            recent = self._recent(key, now)

            if len(recent) < self.max_attempts:
                return 0

            # The key is free again once the max_attempts-th most
            # recent attempt falls out of the window.
            unblock_at = recent[-self.max_attempts] + self.window_seconds

            return max(1, math.ceil(unblock_at - now))

    def record(self, key: str) -> None:
        """Count one attempt for this key."""

        now = time.monotonic()

        with self._lock:
            self._recent(key, now)
            self._attempts.setdefault(key, []).append(now)

            if len(self._attempts) > self.max_keys:
                self._purge(now)

    def reset(self, key: str) -> None:
        """Forget this key's attempts (for example after a successful login)."""

        with self._lock:
            self._attempts.pop(key, None)

    def _purge(self, now: float) -> None:
        """Bound memory use (lock must be held)."""

        for key in list(self._attempts):
            self._recent(key, now)

        # Still too many live keys: drop the ones idle the longest.
        overflow = len(self._attempts) - self.max_keys

        if overflow > 0:
            oldest_first = sorted(
                self._attempts,
                key=lambda k: self._attempts[k][-1],
            )
            for key in oldest_first[:overflow]:
                self._attempts.pop(key, None)


# Limits come from api/config.py (set in .env). Failed logins are
# counted per email and per IP; account creations per IP.
LOGIN_EMAIL_LIMITER = AttemptLimiter(
    max_attempts=LOGIN_MAX_ATTEMPTS_PER_EMAIL,
    window_seconds=LOGIN_WINDOW_SECONDS,
)
LOGIN_IP_LIMITER = AttemptLimiter(
    max_attempts=LOGIN_MAX_ATTEMPTS_PER_IP,
    window_seconds=LOGIN_WINDOW_SECONDS,
)
REGISTER_IP_LIMITER = AttemptLimiter(
    max_attempts=REGISTER_MAX_PER_IP,
    window_seconds=REGISTER_WINDOW_SECONDS,
)