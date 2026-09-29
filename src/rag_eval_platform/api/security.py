"""API key check and a per-client sliding-window rate limiter (no extra dependency)."""

import hmac
import time
from collections import deque
from collections.abc import Callable


def check_api_key(given: str | None, expected: str | None) -> bool:
    """True when no key is configured, or the given one matches (constant-time compare)."""
    if expected is None:
        return True
    return given is not None and hmac.compare_digest(given.encode(), expected.encode())


class RateLimiter:
    """At most ``limit`` requests per client in any ``window_s`` seconds."""

    def __init__(
        self, limit: int, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._limit, self._window, self._clock = limit, window_s, clock
        self._seen: dict[str, deque[float]] = {}

    def allow(self, client: str) -> tuple[bool, float]:
        """(allowed, seconds to wait before the next request is allowed)."""
        now = self._clock()
        seen = self._seen.setdefault(client, deque())
        while seen and now - seen[0] >= self._window:
            seen.popleft()
        if len(seen) >= self._limit:
            return False, round(seen[0] + self._window - now, 3)
        seen.append(now)
        return True, 0.0
