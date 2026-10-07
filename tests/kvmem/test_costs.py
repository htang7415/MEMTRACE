from __future__ import annotations

import pytest

from memtrace.kvmem.costs import ANTHROPIC, WritePremium, breakeven_storage, relative_cost


def test_relative_cost_prices_reads_and_writes() -> None:
    five_min, one_hour = ANTHROPIC
    # 1000 prompt tokens, 800 read from cache: 800 x 0.1 + 200 x write.
    assert relative_cost(1000, 800, five_min) == pytest.approx((80 + 200 * 1.25) / 1000)
    assert relative_cost(1000, 800, one_hour) == pytest.approx((80 + 200 * 2.0) / 1000)
    # Nothing reusable: caching only adds the write premium.
    assert relative_cost(1000, 0, five_min) == pytest.approx(1.25)
    assert relative_cost(0, 0, WritePremium("x", 1, 1, 1)) == 0.0


def test_breakeven_storage_is_savings_per_token_hour() -> None:
    # 900 tokens read at 0.1x save 810 input-token prices; 81 token-hours held -> 10 per token-hour.
    assert breakeven_storage(900, 81.0, read=0.1) == pytest.approx(10.0)
    assert breakeven_storage(900, 0.0) == 0.0
