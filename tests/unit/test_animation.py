"""Tests for playground.animation."""

from itertools import pairwise

import pytest

from rag_eval_platform.playground.animation import ease_steps, fingerprint, should_animate


def test_ease_steps_rise_to_exactly_one() -> None:
    steps = ease_steps(8)

    assert len(steps) == 8
    assert steps[-1] == 1.0
    assert all(0 < a < b for a, b in pairwise(steps))


def test_ease_steps_are_front_loaded() -> None:
    first, *_ = ease_steps(4)

    assert first > 0.25  # ease-out: fast start, gentle finish


def test_ease_steps_needs_at_least_one_step() -> None:
    with pytest.raises(ValueError, match="steps"):
        ease_steps(0)


def test_fingerprint_is_stable_and_sensitive() -> None:
    assert fingerprint("a", 1, (0.5,)) == fingerprint("a", 1, (0.5,))
    assert fingerprint("a", 1) != fingerprint("a", 2)


def test_should_animate_once_per_new_data() -> None:
    state: dict[str, object] = {}

    assert should_animate(state, "chart", "v1") is True
    assert should_animate(state, "chart", "v1") is False  # same data: no replay
    assert should_animate(state, "chart", "v2") is True  # new data: animate again
    assert should_animate(state, "other", "v2") is True  # each chart tracks its own data
