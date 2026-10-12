from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest

from memtrace.kv import resume_decision
from memtrace.kv.resume_decision import decide, decide_capacity, decide_shared_tier, grade, next_step


def trial(
    arm: str, seed: int, recompute: float, p50: float, *, swapouts: float = 0.0, measured: float | None = None
) -> dict[str, Any]:
    metrics = {
        "resumed_recompute_share": recompute,
        "resumed_ttft_p50_ms": p50,
        "resumed_ttft_all_p95_ms": 2000.0,
        "all_ttft_p50_ms": 140.0,
        "host_slept_s": 0.0,
        "host_swapouts": swapouts,
        **({} if measured is None else {"host_swapouts_measured": measured}),
    }
    return {"cell_id": f"arm={arm}", "seed": seed, "status": "ok", "metrics": metrics}


def run(run_id: str, trials: list[dict[str, Any]]) -> dict[str, Any]:
    return {"run_id": run_id, "trials": trials}


R1 = run(
    "r1",
    [
        trial("stock", 0, 0.68, 900.0),
        trial("stock-equal-memory", 0, 0.33, 390.0, swapouts=72_770),
        trial("host-pool+ssd", 0, 0.04, 425.0, swapouts=4_524),
        *(trial("stock-equal-memory", s, 0.30, 400.0) for s in (1, 2)),
        *(trial("host-pool+ssd", s, 0.05, 380.0) for s in (1, 2)),
    ],
)
# The rerun paged while its engines started, not while it measured.
RERUN = run(
    "rerun",
    [
        trial("stock-equal-memory", 0, 0.35, 410.0, swapouts=3_000, measured=0.0),
        trial("host-pool+ssd", 0, 0.05, 400.0, swapouts=3_000, measured=0.0),
    ],
)


def test_paged_trials_need_a_rerun_and_a_valid_rerun_replaces_them() -> None:
    first = decide([R1], "stock-equal-memory", "host-pool+ssd", repeats=3)
    assert first["next_step"].startswith("incomplete: 2 of 3")
    assert [(r["arm"], r["seed"]) for r in first["to_rerun"]] == [("stock-equal-memory", 0), ("host-pool+ssd", 0)]
    assert first["to_rerun"][0]["reason"] == "host paged out 72770 pages"

    both = decide([R1, RERUN], "stock-equal-memory", "host-pool+ssd", repeats=3)
    assert both["to_rerun"] == [] and [r["seed"] for r in both["repeats"]] == [0, 1, 2]
    assert both["repeats"][0]["runs"] == {"stock-equal-memory": "rerun", "host-pool+ssd": "rerun"}
    assert both["grades"] == {
        "resumed_recompute_share": "A",  # 86% / 83% / 83% better
        "resumed_ttft_p50_ms": "B",  # 2% / 5% / 5% better
        "resumed_ttft_all_p95_ms": "B",
    }
    assert both["next_step"].startswith("R2 before S2")


def test_an_invalid_rerun_does_not_replace_a_valid_trial() -> None:
    later = run("later", [trial("host-pool+ssd", 1, 0.9, 9000.0, swapouts=10, measured=10)])
    result = decide([R1, RERUN, later], "stock-equal-memory", "host-pool+ssd", repeats=3)
    assert result["to_rerun"] == [] and result["repeats"][1]["runs"]["host-pool+ssd"] == "r1"


def test_grades_and_next_steps() -> None:
    assert grade([0.2, 0.15, 0.5]) == "A" and grade([-0.2, -0.3, -0.15]) == "C" and grade([0.2, 0.1, 0.3]) == "B"
    assert grade([0.2, -0.2, 0.2]) == "B"
    assert next_step({"resumed_recompute_share": "A", "resumed_ttft_p50_ms": "A"}).startswith("S2")
    assert next_step({"resumed_recompute_share": "B", "resumed_ttft_p50_ms": "A"}).startswith("stop")
    assert next_step({"resumed_recompute_share": "A", "resumed_ttft_p50_ms": "C"}).startswith("not covered")


def test_cli_reads_run_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dirs = []
    for r in (R1, RERUN):
        d = tmp_path / r["run_id"]
        d.mkdir()
        (d / "results.json").write_text(json.dumps(r))
        dirs.append(str(d))
    monkeypatch.setattr(sys, "argv", ["x", *dirs, "--out", str(tmp_path / "decision.json")])
    resume_decision.main()
    assert "next step: R2 before S2" in capsys.readouterr().out
    assert json.loads((tmp_path / "decision.json").read_text())["grades"]["resumed_recompute_share"] == "A"


def s2_trial(arm: str, seed: int, recompute: float, p95: float, *, measured: float = 0.0) -> dict[str, Any]:
    t = trial(arm, seed, recompute, 400.0, measured=measured)
    t["metrics"]["resumed_ttft_all_p95_ms"] = p95
    t["requests_per_endpoint"] = [600, 600]
    return t


def s2_run(shared: list[tuple[float, float]], spread_recompute: float = 0.40) -> dict[str, Any]:
    trials = []
    for seed, (recompute, p95) in enumerate(shared):
        trials += [
            s2_trial("affinity+private", seed, 0.10, 1500.0),
            s2_trial("round-robin+private", seed, spread_recompute, 2500.0),
            s2_trial("round-robin+shared", seed, recompute, p95),
        ]
    return run("s2", trials)


def test_s2_shared_tier_recovers_when_within_15_percent_of_affinity_in_every_repeat() -> None:
    result = decide_shared_tier([s2_run([(0.11, 1600.0), (0.10, 1700.0), (0.114, 1500.0)])], repeats=3)
    assert result["affinity_benefit"] and result["shared_recovers"]
    assert result["repeats"][0]["affinity_benefit"] == pytest.approx(0.75)  # 0.40 -> 0.10
    assert result["repeats"][1]["shared_vs_affinity"]["resumed_ttft_all_p95_ms"] == pytest.approx(-0.1333, abs=1e-3)
    assert result["repeats"][0]["calls_per_replica"]["round-robin+shared"] == [600, 600]
    assert result["next_step"] == "item 2: capacity with a shared tier"


def test_s2_one_repeat_outside_15_percent_means_no_recovery() -> None:
    result = decide_shared_tier([s2_run([(0.11, 1600.0), (0.10, 1800.0), (0.10, 1500.0)])], repeats=3)
    assert result["affinity_benefit"] and not result["shared_recovers"]  # p95 20% worse in repeat 1
    assert result["next_step"].startswith("item 2: one replica") and "does not recover" in result["next_step"]


def test_s2_without_an_affinity_benefit_there_is_nothing_to_recover() -> None:
    result = decide_shared_tier([s2_run([(0.10, 1500.0)] * 3, spread_recompute=0.11)], repeats=3)
    assert not result["affinity_benefit"] and "no benefit" in result["next_step"]


def test_s2_needs_valid_trials_in_all_three_arms() -> None:
    runs = s2_run([(0.10, 1500.0)] * 3)
    runs["trials"][5] = s2_trial("round-robin+shared", 1, 0.10, 1500.0, measured=900.0)
    result = decide_shared_tier([runs], repeats=3)
    assert result["next_step"].startswith("incomplete: 2 of 3")
    assert result["to_rerun"] == [{"arm": "round-robin+shared", "seed": 1, "reason": "host paged out 900 pages"}]


def sweep(p95: dict[tuple[str, int], list[float]], paged: tuple[str, int, int] | None = None) -> dict[str, Any]:
    """A capacity run: p95 per (arm, level), one value per repeat."""
    trials = []
    for (arm, level), values in p95.items():
        for seed, v in enumerate(values):
            measured = 500.0 if paged == (arm, level, seed) else 0.0
            t = trial(f"{arm},c={level}", seed, 0.1, 400.0, measured=measured)
            t["metrics"]["resumed_ttft_all_p95_ms"] = v
            trials.append(t)
    return run("cap", trials)


BASE, TIER = "stock-equal-memory", "host-pool+ssd"


def test_capacity_is_the_highest_level_within_the_slo_in_every_repeat() -> None:
    runs = sweep({
        (BASE, 12): [900, 950, 1000], (BASE, 24): [1900, 2100, 1800], (BASE, 36): [4000, 5000, 4500],
        (TIER, 12): [800, 850, 900], (TIER, 24): [1500, 1600, 1700], (TIER, 36): [1900, 1950, 1990],
    })  # fmt: skip
    result = decide_capacity([runs], BASE, TIER, repeats=3, slo_ms=2000)
    assert result["capacity"] == {BASE: 12, TIER: 36}  # base misses 24 in one repeat
    assert result["next_step"].startswith("win")


def test_same_or_lower_capacity_is_no_win_or_a_cost() -> None:
    same = sweep({(BASE, 12): [900] * 3, (BASE, 24): [2500] * 3, (TIER, 12): [800] * 3, (TIER, 24): [2100] * 3})
    assert decide_capacity([same], BASE, TIER, 3, 2000)["next_step"].startswith("no win")
    lower = sweep({(BASE, 12): [900] * 3, (BASE, 24): [1500] * 3, (TIER, 12): [800] * 3, (TIER, 24): [2100] * 3})
    result = decide_capacity([lower], BASE, TIER, 3, 2000)
    assert result["capacity"] == {BASE: 24, TIER: 12} and result["next_step"].startswith("the tier costs")
    none = sweep({(BASE, 12): [2500] * 3, (TIER, 12): [2500] * 3})
    assert decide_capacity([none], BASE, TIER, 3, 2000)["capacity"] == {BASE: 0, TIER: 0}


def test_capacity_needs_valid_trials_in_every_cell() -> None:
    runs = sweep({(BASE, 12): [900] * 3, (TIER, 12): [800] * 3}, paged=(TIER, 12, 1))
    result = decide_capacity([runs], BASE, TIER, 3, 2000)
    assert result["next_step"].startswith("incomplete: 2 of 3")
    assert result["to_rerun"] == [{"arm": "host-pool+ssd,c=12", "seed": 1, "reason": "host paged out 500 pages"}]
