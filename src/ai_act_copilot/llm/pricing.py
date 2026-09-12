"""Token pricing, so every call can report what it cost.

Cost is computed from the usage the API returns rather than estimated from the prompt:
cache reads and writes are billed differently, and that difference is the whole point of
prompt caching. Prices are per million tokens, in US dollars.
"""

from dataclasses import dataclass

# Cached prefixes are cheap to read and slightly more expensive to write.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass(frozen=True, slots=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float


PRICES: dict[str, ModelPrice] = {
    "claude-opus-5": ModelPrice(5.00, 25.00),
    "claude-opus-4-8": ModelPrice(5.00, 25.00),
    "claude-sonnet-5": ModelPrice(2.00, 10.00),
    "claude-haiku-4-5": ModelPrice(1.00, 5.00),
}


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts as reported by the API."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


def cost_usd(model: str, usage: Usage) -> float:
    """What this call cost, or 0.0 for a model with no published price here."""
    price = PRICES.get(model)
    if price is None:
        return 0.0
    million = 1_000_000
    return (
        usage.input_tokens * price.input_per_mtok
        + usage.cache_read_tokens * price.input_per_mtok * CACHE_READ_MULTIPLIER
        + usage.cache_write_tokens * price.input_per_mtok * CACHE_WRITE_MULTIPLIER
        + usage.output_tokens * price.output_per_mtok
    ) / million
