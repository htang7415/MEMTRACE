from __future__ import annotations

import json
from pathlib import Path

import pytest

from memtrace.harness import bundle as bundle_mod
from memtrace.harness.bundle import RunBundle
from memtrace.harness.results import ExperimentResult, from_dict


@pytest.fixture(autouse=True)
def _host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bundle_mod, "keep_awake", lambda: None)
    swaps = iter([{"swapouts": 10}, {"swapouts": 10}] * 3 + [{"swapouts": 10}, {"swapouts": 25}])
    monkeypatch.setattr(bundle_mod, "host_swap_pages", lambda: next(swaps))


def test_trials_record_host_checks_and_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    logs: list[str] = []
    b = RunBundle({"name": "unit", "description": "d"}, tmp_path, log=logs.append)
    assert (b.dir / "spec.yaml").exists() and b.run_id.endswith("-unit")
    with b.trial("arm=a", {"arm": "a"}, repeat=0, seed=7) as t:
        t.metrics["x"] = 1.0
        t.requests_per_endpoint = [3]
    monkeypatch.setattr(bundle_mod, "sleep_clock", lambda: lambda: 900.0)  # the host slept 15 min
    with b.trial("arm=a", {"arm": "a"}, repeat=1, seed=8) as t:
        t.metrics["x"] = 3.0
    with b.trial("arm=b", {"arm": "b"}, repeat=0, seed=7):
        raise RuntimeError("engine died")
    monkeypatch.setattr(bundle_mod, "sleep_clock", lambda: lambda: 0.0)
    with b.trial("arm=b", {"arm": "b"}, repeat=1, seed=8) as t:  # the host paged out 15 pages
        t.metrics["x"] = 5.0
    ok, slept, failed, paged = b.trials
    assert ok.status == "ok" and ok.quiet_host_ok and ok.metrics["host_swapouts"] == 0.0
    assert paged.status == "ok" and not paged.quiet_host_ok and paged.metrics["host_swapouts"] == 15.0
    assert slept.status == "ok" and not slept.quiet_host_ok and slept.metrics["host_slept_s"] == 900.0
    assert failed.status == "failed" and failed.error == "RuntimeError: engine died" and failed.metrics == {}
    assert any("WARNING the host slept" in m for m in logs) and any("FAILED" in m for m in logs)
    assert any("WARNING the host paged out 15 pages" in m for m in logs)
    b.write_requests("arm=a/r0", [{"k": 1}, {"k": 2}])
    out = b.finish({"day": "x"})
    result = from_dict(ExperimentResult, json.loads((out / "results.json").read_text()))
    assert result.provenance.tools["trials_completed"] == 4 and result.provenance.tools["day"] == "x"
    cells = {c.cell_id: c for c in result.cells}
    assert cells["arm=a"].metrics["x"].mean == 2.0 and cells["arm=b"].n_failed == 1
    assert [json.loads(line)["trial"] for line in (out / "requests.jsonl").read_text().splitlines()] == ["arm=a/r0"] * 2
