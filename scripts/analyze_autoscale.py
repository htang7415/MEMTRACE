"""Summarize an autoscaling burst run (`scripts/studies/platform_studies.sh burst`).

Reads `<run>/timeline.txt` (epoch, ready replicas, HPA desired replicas, once per second) and the
per-request records of each load phase, and reports: HPA reaction time and time to full capacity after
the burst starts, burst-phase latency, and replica-seconds (the capacity cost of the run).

    python scripts/analyze_autoscale.py data/autoscale data/autoscale_fixed1 data/autoscale_fixed4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


def summarize(run: Path) -> dict[str, Any]:
    timeline: list[tuple[float, int, int | None]] = []
    for line in (run / "timeline.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            timeline.append(
                (float(parts[0]), int(parts[1] or 0), int(parts[2]) if len(parts) > 2 and parts[2] else None)
            )
    phases: dict[str, list[dict[str, Any]]] = {}
    for phase_dir in sorted(run.glob("phase*/agent-sessions/c*")):
        records = [json.loads(line) for line in (phase_dir / "requests.jsonl").read_text().splitlines()]
        phases[phase_dir.parts[-3]] = records
    burst = phases["phase2"]
    burst_start = min(r["started_at"] for r in burst)
    burst_end = max(r["started_at"] + (r.get("e2e_seconds") or 0) for r in burst)
    peak = max(ready for _, ready, _ in timeline)

    def first_time(predicate: Callable[[int, int | None], bool]) -> float | None:
        return next(
            (t - burst_start for t, ready, desired in timeline if t >= burst_start and predicate(ready, desired)), None
        )

    ttft = np.array([r["ttft_seconds"] for r in burst if r.get("ok") and r.get("ttft_seconds") is not None])
    early = np.array(
        [
            r["ttft_seconds"]
            for r in burst
            if r.get("ok") and r.get("ttft_seconds") and r["started_at"] - burst_start < 20
        ]
    )
    seconds = [(t1 - t0) * ready for (t0, ready, _), (t1, _, _) in zip(timeline, timeline[1:])]
    starts = [r["started_at"] for records in phases.values() for r in records]
    ends = [r["started_at"] + (r.get("e2e_seconds") or 0) for records in phases.values() for r in records]
    load_seconds = [
        (t1 - t0) * ready
        for (t0, ready, _), (t1, _, _) in zip(timeline, timeline[1:])
        if min(starts) <= t0 <= max(ends)
    ]
    scale_down = next((t - burst_end for t, ready, _ in timeline if t > burst_end and ready < peak), None)
    return {
        "run": run.name,
        "peak_ready_replicas": peak,
        "hpa_reacts_after_s": first_time(lambda ready, desired: desired is not None and desired > timeline[0][1]),
        "full_capacity_after_s": first_time(lambda ready, desired: ready >= peak) if peak > timeline[0][1] else 0.0,
        "first_scale_down_after_burst_s": scale_down,
        "burst_duration_s": burst_end - burst_start,
        "burst_ttft_p50_s": float(np.percentile(ttft, 50)),
        "burst_ttft_p95_s": float(np.percentile(ttft, 95)),
        "burst_first_20s_ttft_p95_s": float(np.percentile(early, 95)) if len(early) else None,
        "burst_errors": sum(not r.get("ok") for r in burst),
        "replica_seconds_during_load": sum(load_seconds),
        "replica_seconds_total": sum(seconds),
    }


def main() -> None:
    rows = [summarize(Path(arg)) for arg in sys.argv[1:]]
    for row in rows:
        print(json.dumps(row))
    out = Path(sys.argv[1]) / "analysis.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
