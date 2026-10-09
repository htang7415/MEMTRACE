from __future__ import annotations

from typing import Any

from memtrace.kv.prefix_sim.sweep import frontier


def row(name: str, hit: float, gpu: float, ram: float = 0.0) -> dict[str, Any]:
    return {"name": name, "hit_share_of_reusable": hit, "gb_hours": {"gpu": gpu, "ram": ram}}


def test_frontier_drops_dominated_runs() -> None:
    rows = [
        row("small", 0.5, 10),
        row("big", 0.9, 40),
        row("worse-than-big", 0.8, 50),  # less hit, more GPU
        row("ram", 0.95, 10, 100),  # more lower-tier memory, but cheapest GPU at the top hit share
        row("same-as-small", 0.5, 10),  # ties do not dominate each other
    ]
    assert [r["name"] for r in frontier(rows)] == ["ram", "small", "same-as-small", "big"]
