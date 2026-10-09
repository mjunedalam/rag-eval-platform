"""What an answer cost: tokens times the configured price (0 for local models)."""

from dataclasses import dataclass
from typing import Self

from rag_eval_platform.config.settings import Settings


@dataclass(frozen=True)
class Prices:
    input_per_1m: float = 0.0  # USD per million input tokens
    output_per_1m: float = 0.0

    @property
    def is_free(self) -> bool:
        return self.input_per_1m == 0 and self.output_per_1m == 0

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(settings.price_input_per_1m, settings.price_output_per_1m)


def cost_usd(input_tokens: int | None, output_tokens: int | None, prices: Prices) -> float:
    spent = (input_tokens or 0) * prices.input_per_1m + (output_tokens or 0) * prices.output_per_1m
    return spent / 1_000_000
