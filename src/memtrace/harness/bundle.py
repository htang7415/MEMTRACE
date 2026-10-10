"""One run's result bundle, and the host checks every trial in it records.

    bundle = RunBundle(spec, out_root)                 # data/runs/<run_id>/ with spec.yaml; host kept awake
    with bundle.trial(cell_id, params, repeat=r, seed=s) as t:
        t.metrics.update(...)                          # what the trial measured
        t.target = engine.describe()
    bundle.write_requests(label, rows)                 # appended to requests.jsonl
    bundle.finish()                                    # results.json (cells with 95% CIs, provenance)

Every trial waits for a quiet host when the spec asks for one, and records `host_slept_s` and `host_swapouts`.
A trial during which the host slept is not on a quiet host: client and engine pause together, but resume
throttled, so latencies across a sleep are not comparable. Nor is one during which the host paged anything out.
A trial that raises is recorded as failed, with its error, and the run continues.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable, Iterator, Mapping

import yaml

from memtrace.harness.provenance import make_provenance
from memtrace.harness.results import RESULT_SCHEMA_VERSION, ExperimentResult, TrialResult, aggregate_cells
from memtrace.harness.runner import wait_for_quiet_host
from memtrace.harness.spec import QuietHost
from memtrace.harness.stamps import utc_now_iso
from memtrace.harness.system_info import host_swap_pages, keep_awake, sleep_clock

MAX_SLEEP_S = 5.0  # a trial whose host slept longer is flagged as not quiet


def _stderr(message: str) -> None:
    print(message, file=sys.stderr)


@dataclass
class Trial:
    """What a trial body fills in; the bundle adds the host checks."""

    metrics: dict[str, float] = field(default_factory=dict)
    target: dict[str, Any] = field(default_factory=dict)
    requests_per_endpoint: list[int] = field(default_factory=list)


class RunBundle:
    def __init__(
        self,
        spec: Mapping[str, Any],
        out_root: Path,
        *,
        quiet_host: QuietHost | None = None,
        log: Callable[[str], None] = _stderr,
    ) -> None:
        self.spec = dict(spec)
        self.started_at = utc_now_iso()
        self.run_id = f"{datetime.now(tz=timezone.utc):%Y%m%dT%H%M%SZ}-{self.spec['name']}"
        self.dir = Path(out_root) / self.run_id
        self.dir.mkdir(parents=True)
        (self.dir / "spec.yaml").write_text(yaml.safe_dump(self.spec, sort_keys=False), encoding="utf-8")
        self.quiet_host = quiet_host
        self.log = log
        self.trials: list[TrialResult] = []
        self.cells: dict[str, dict[str, Any]] = {}
        keep_awake()

    def log_dir(self, label: str) -> Path:
        return self.dir / "logs" / label.replace("/", "_")

    @contextmanager
    def trial(
        self, cell_id: str, params: Mapping[str, Any], *, repeat: int, seed: int, label: str | None = None
    ) -> Iterator[Trial]:
        label = label or f"{cell_id}/r{repeat}"
        self.cells.setdefault(cell_id, dict(params))
        load, quiet = (wait_for_quiet_host(self.quiet_host)[:2]) if self.quiet_host else (0.0, True)
        started_at, t0, slept, swap_before = utc_now_iso(), time.perf_counter(), sleep_clock(), host_swap_pages()
        trial = Trial()
        error: str | None = None
        try:
            yield trial
        except Exception as exc:  # a failed trial is recorded, never silently dropped
            error = f"{type(exc).__name__}: {exc}"[:500]
            self.log(f"{label}: FAILED {error}")
        swap_after = host_swap_pages()
        trial.metrics["host_slept_s"] = slept()
        if swap_before and swap_after:
            trial.metrics["host_swapouts"] = float(swap_after["swapouts"] - swap_before["swapouts"])
        if trial.metrics["host_slept_s"] > MAX_SLEEP_S:
            quiet = False
            self.log(f"{label}: WARNING the host slept {trial.metrics['host_slept_s']:.0f} s during the trial")
        if trial.metrics.get("host_swapouts", 0.0) > 0:
            quiet = False
            self.log(f"{label}: WARNING the host paged out {trial.metrics['host_swapouts']:.0f} pages during the trial")
        if error is None:
            self.log(f"{label}: " + " ".join(f"{k}={v:.4g}" for k, v in trial.metrics.items()))
        self.trials.append(
            TrialResult(
                trial_id=label,
                cell_id=cell_id,
                repeat=repeat,
                seed=seed,
                status="ok" if error is None else "failed",
                started_at=started_at,
                duration_s=round(time.perf_counter() - t0, 3),
                host_load_1m_before=load,
                quiet_host_ok=quiet,
                metrics={k: round(v, 6) for k, v in trial.metrics.items()} if error is None else {},
                requests_per_endpoint=trial.requests_per_endpoint,
                target=trial.target,
                error=error,
            )
        )

    def write_requests(self, label: str, rows: Iterable[Mapping[str, Any]]) -> None:
        with (self.dir / "requests.jsonl").open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps({"trial": label, **row}) + "\n")

    def finish(self, tools: Mapping[str, Any] | None = None) -> Path:
        result = ExperimentResult(
            schema_version=RESULT_SCHEMA_VERSION,
            run_id=self.run_id,
            name=self.spec["name"],
            description=self.spec.get("description", ""),
            spec=self.spec,
            provenance=make_provenance(
                self.spec, self.started_at, {**(tools or {}), "trials_completed": len(self.trials)}
            ),
            trials=self.trials,
            cells=aggregate_cells(self.trials, self.cells),
        )
        (self.dir / "results.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
        self.log(f"wrote {self.dir}")
        return self.dir
