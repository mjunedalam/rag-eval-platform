"""Tests for api.security: the API key check and the rate limiter."""

from rag_eval_platform.api.security import RateLimiter, check_api_key


def test_no_expected_key_means_everyone_is_allowed() -> None:
    assert check_api_key(None, None)
    assert check_api_key("anything", None)


def test_the_key_must_match_exactly() -> None:
    assert check_api_key("s3cret", "s3cret")
    assert not check_api_key("s3cre", "s3cret")
    assert not check_api_key(None, "s3cret")


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_rate_limit_blocks_after_the_limit_then_slides() -> None:
    clock = Clock()
    limiter = RateLimiter(limit=2, window_s=60, clock=clock)

    assert limiter.allow("a") == (True, 0.0)
    clock.now += 10
    assert limiter.allow("a") == (True, 0.0)
    allowed, retry = limiter.allow("a")
    assert not allowed
    assert retry == 50.0  # the first request leaves the window in 50 s

    clock.now += 50.001
    assert limiter.allow("a")[0]  # the window has slid


def test_clients_are_limited_separately() -> None:
    limiter = RateLimiter(limit=1, window_s=60, clock=Clock())

    assert limiter.allow("a")[0]
    assert not limiter.allow("a")[0]
    assert limiter.allow("b")[0]
