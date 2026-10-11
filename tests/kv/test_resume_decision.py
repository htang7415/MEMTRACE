from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest

from memtrace.kv import resume_decision
from memtrace.kv.resume_decision import decide, grade, next_step


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
