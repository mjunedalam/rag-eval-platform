"""Tests for observability.cost."""

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.observability.cost import Prices, cost_usd


def test_cost_is_tokens_times_price_per_million() -> None:
    prices = Prices(input_per_1m=0.15, output_per_1m=0.60)

    assert cost_usd(1_000_000, 500_000, prices) == 0.45
    assert cost_usd(None, None, prices) == 0.0


def test_local_models_are_free_by_default() -> None:
    prices = Prices.from_settings(Settings(_env_file=None))

    assert prices.is_free
    assert cost_usd(300, 40, prices) == 0.0
    assert not Prices(output_per_1m=1.0).is_free
