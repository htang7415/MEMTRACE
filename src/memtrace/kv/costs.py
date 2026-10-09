"""Phase 6 M1: what prompt-cache retention costs and saves under providers' price structures.

Prices are multiples of the provider's base input-token price, so no absolute price is needed. Inputs come from
`memtrace.kv.retention`: total prompt tokens, tokens a cache of some lifetime would serve with perfect
placement (`retained`), and the token-hours that cache holds (its working set over the trace window).

- Write premium (Anthropic): 5-minute writes cost 1.25x, 1-hour writes 2x, reads 0.1x; no storage fee.
- Storage fee (Gemini explicit caching): billed per token-hour held. Its price is not modelled; instead
  `breakeven_storage` is the highest fee per token-hour (as a multiple of the input price per token) at which
  holding still costs less than recomputing, with reads at `read` and writes at 1x.
- OpenAI's retention adds no write premium or storage fee for the customer; its cost falls on the provider as
  the working set.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WritePremium:
    name: str
    lifetime: float  # seconds, refreshed on every read
    write: float
    read: float


ANTHROPIC = (
    WritePremium("anthropic-5min", 300.0, 1.25, 0.1),
    WritePremium("anthropic-1h", 3600.0, 2.0, 0.1),
)


def relative_cost(total_prompt: int, retained: int, pricing: WritePremium) -> float:
    """Cost with the cache, divided by the cost of sending every prompt token uncached at 1x."""
    if not total_prompt:
        return 0.0
    return (retained * pricing.read + (total_prompt - retained) * pricing.write) / total_prompt


def breakeven_storage(retained: int, token_hours: float, read: float = 0.1) -> float:
    """Storage fee per token-hour (x input price per token) at which caching just pays for itself."""
    return retained * (1 - read) / token_hours if token_hours else 0.0
