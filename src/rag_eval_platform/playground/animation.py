"""Play-once animation helpers for the playground (pure, no Streamlit).

A chart animates when it receives data it has not shown before, and draws its final frame
directly on every later rerun (Streamlit reruns the page on every click).
"""

import hashlib
from collections.abc import MutableMapping

_SLOT = "_animated:"


def ease_steps(steps: int) -> tuple[float, ...]:
    """Progress values in (0, 1] with a cubic ease-out, ending exactly at 1.0."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    return tuple(1 - (1 - i / steps) ** 3 for i in range(1, steps + 1))


def fingerprint(*parts: object) -> str:
    """A short stable id for a chart's data."""
    return hashlib.sha1(repr(parts).encode(), usedforsecurity=False).hexdigest()[:16]


def should_animate(state: MutableMapping[str, object], key: str, data_fingerprint: str) -> bool:
    """True the first time ``key`` sees ``data_fingerprint``; records it in ``state``."""
    slot = _SLOT + key
    if state.get(slot) == data_fingerprint:
        return False
    state[slot] = data_fingerprint
    return True
