from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from memtrace.harness import dashboard_export


def trial(arm: str, seed: int, recompute: float, *, measured: float = 0.0) -> dict[str, Any]:
    metrics = {"resumed_recompute_share": recompute, "resumed_ttft_p50_ms": 400.0, "host_swapouts_measured": measured}
    return {"cell_id": f"arm={arm}", "seed": seed, "status": "ok", "metrics": metrics}


def write_run(root: Path, run_id: str, trials: list[dict[str, Any]]) -> None:
    (root / run_id).mkdir(parents=True)
    (root / run_id / "results.json").write_text(json.dumps({"run_id": run_id, "trials": trials}))


def test_resume_track_shows_the_trial_each_rule_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_run(tmp_path, "a", [trial("stock", 0, 0.3), trial("ssd", 0, 0.05, measured=900), trial("ssd", 1, 0.04)])
    write_run(tmp_path, "b", [trial("ssd", 0, 0.06)])  # a paging-free rerun replaces the paged trial
    studies = (("Study", ("a", "b"), "verdict"), ("Not on disk", ("missing",), "none"))
    monkeypatch.setattr(dashboard_export, "RESUME_STUDIES", studies)
    (study,) = dashboard_export.resume_track(tmp_path)
    assert study["verdict"] == "verdict" and study["runs"] == ["a", "b"]
    got = [(r["cell"], r["repeat"], r["run_id"], r["excluded"], r["resumed_recompute_share"]) for r in study["rows"]]
    assert got == [("stock", 0, "a", None, 0.3), ("ssd", 0, "b", None, 0.06), ("ssd", 1, "a", None, 0.04)]
    assert study["rows"][0]["calls"] is None  # metrics a trial lacks are null


def test_a_paged_trial_without_a_valid_copy_is_shown_as_not_counting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_run(tmp_path, "a", [trial("stock", 0, 0.3, measured=500)])
    monkeypatch.setattr(dashboard_export, "RESUME_STUDIES", (("Study", ("a",), "v"),))
    (study,) = dashboard_export.resume_track(tmp_path)
    assert study["rows"][0]["excluded"] == "host paged out 500 pages"
