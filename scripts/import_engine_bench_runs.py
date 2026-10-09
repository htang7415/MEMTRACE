"""Import the recorded MEMTRACE engine-bench runs into the harness result format, under the specs that now
define those studies, so every result reads the same way. One run directory per recorded batch, under
data/runs/ (git-ignored); local data only.

    python scripts/import_engine_bench_runs.py      # reads data/<study>/..., writes data/runs/<id>-imported/

Each recorded engine-bench level (summary.json + requests.jsonl) becomes one trial; its metrics are recomputed
from the request records by the harness's own summary (TTFT 2 s / TPOT 0.1 s SLO, as recorded). Pool counters,
engine prefix-cache counters and swap-outs go into the trial's target record. Provenance names the source
directory: these runs predate the harness and carry no commit.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from memtrace.harness.loadgen import RequestRecord, summarize
from memtrace.harness.planner import cell_id_for
from memtrace.harness.results import RESULT_SCHEMA_VERSION, ExperimentResult, Provenance, TrialResult, aggregate_cells
from memtrace.harness.runner import flatten_summary
from memtrace.harness.spec import load_spec
from memtrace.harness.stamps import stable_config_fingerprint

DATA = Path("data")
OUT = DATA / "runs"

# (spec, batch name, glob of level directories under data/, how to read cell values and the repeat from a path)
Batch = tuple[str, str, str, Any]


def _policy(path: Path) -> str:
    return next(p for p in path.parts if p.startswith(("hetero-", "sim4-"))).split("-", 1)[1]


def _rep(path: Path) -> int:
    m = re.search(r"/rep(\d+)/", path.as_posix())
    return int(m.group(1)) - 1 if m else 0


def _level(path: Path) -> int:
    return int(path.name.removeprefix("c"))


ENGINES = {"vllm-metal": "vllm_metal", "mlx-lm": "mlx_lm", "vllm-cpu": "vllm_cpu"}
BATCHES: list[Batch] = [
    (
        "e7_engines_sharegpt",
        "r2",
        "engine_bench_r2/*/sharegpt/c*",
        lambda p: ({"target.variant": ENGINES[p.parts[-3]], "workload.concurrency": _level(p)}, 0),
    ),
    (
        "e7b_engines_mooncake",
        "r2",
        "engine_bench_r2/*/mooncake-toolagent/c*",
        lambda p: ({"target.variant": ENGINES[p.parts[-3]], "workload.concurrency": _level(p)}, 0),
    ),
    (
        "e8_routing_kind_sims",
        "v2",
        "routing_study_v2/sim4-*/agent-sessions/c*",
        lambda p: ({"target.policy": _policy(p), "workload.concurrency": _level(p)}, 0),
    ),
    (
        "e9_hetero_pool",
        "reps",
        "hetero_reps/rep*/hetero-*/agent-sessions/c8",
        lambda p: ({"target.policy": _policy(p)}, _rep(p)),
    ),
    (
        "e9_hetero_pool",
        "reps2",
        "hetero_reps2/rep*/hetero-*/agent-sessions/c8",
        lambda p: ({"target.policy": _policy(p)}, _rep(p)),
    ),
    (
        "e10_hosted_overflow",
        "runs",
        "hosted/*/rep*/hetero-capacity/agent-sessions/c16",
        lambda p: ({"target.variant": p.parts[-5]}, _rep(p)),
    ),
    (
        "e11_copilot_replay",
        "runs",
        "copilot/rep*/hetero-*/copilot-agent/open",
        lambda p: ({"target.policy": _policy(p)}, _rep(p)),
    ),
    (
        "e11b_copilot_replay_4b",
        "runs",
        "copilot-4b*/rep*/hetero-*/copilot-agent/open",
        lambda p: ({"target.policy": _policy(p), "workload.sessions": 32 if "s32" in p.parts[-6] else 16}, _rep(p)),
    ),
    (
        "e11c_copilot_precise_index",
        "runs",
        "precise/rep*/hetero-*/copilot-agent/open",
        lambda p: ({"target.policy": _policy(p)}, _rep(p)),
    ),
    ("k11_engine_kv_check", "pilot", "m3/pilot/*/copilot-agent/open", lambda p: ({}, 0)),
]


def records(level: Path) -> list[RequestRecord]:
    out = []
    for line in (level / "requests.jsonl").read_text().splitlines():
        r = json.loads(line)
        session = r.get("session")
        rid = f"s{session}-c{r['call']}" if session is not None else f"r{r['index']}"
        ok = bool(r.get("ok"))
        out.append(
            RequestRecord(
                request_id=rid,
                session_id=f"s{session}" if session is not None else rid,
                endpoint=0,
                scheduled_s=r.get("started_at", 0.0),  # the earliest runs did not record send times
                status="ok" if ok else "error",
                ttft_s=r.get("ttft_seconds") if ok else None,
                e2e_s=r.get("e2e_seconds") if ok else None,
                prompt_tokens=int(r.get("prompt_tokens") or 0),
                cached_tokens=int(r.get("cached_prompt_tokens") or 0),
                completion_tokens=int(r.get("output_tokens") or 0),
                degraded=not ok,
                error=None if ok else str(r.get("error")),
            )
        )
    origin = min((rec.scheduled_s for rec in out), default=0.0)
    return [RequestRecord(**{**asdict(rec), "scheduled_s": rec.scheduled_s - origin}) for rec in out]


def trial(level: Path, cell: dict[str, Any], repeat: int, spec_seed: int) -> tuple[TrialResult, list[RequestRecord]]:
    summary = json.loads((level / "summary.json").read_text())
    manifest = (
        json.loads(next(level.parent.glob("manifest.json")).read_text())
        if (level.parent / "manifest.json").exists()
        else {}
    )
    recs = records(level)
    stats = summarize(recs, duration_s=summary["wall_seconds"], ttft_slo_s=2.0, e2e_slo_s=3600.0, tpot_slo_s=0.1)
    metrics = flatten_summary(stats)
    if "open_loop" in summary:
        metrics["peak_in_flight"] = float(summary["open_loop"]["peak_in_flight"])
    sent = [json.loads(line).get("started_at") for line in (level / "requests.jsonl").read_text().splitlines()]
    first = min((s for s in sent if s is not None), default=None)
    if first is None:  # no send times recorded: the run's start from its manifest
        first = datetime.fromisoformat(manifest["started_at"]).timestamp()
    cell_id = cell_id_for(cell)
    target = {
        "imported_from": level.as_posix(),
        "engine": manifest.get("engine"),
        "engine_version": manifest.get("engine_version"),
        "load_started_epoch_s": first,
        "collected": summary.get("pool") or {},
        "engine_prefix_cache": summary.get("prefix_cache"),
    }
    if summary.get("host_swap_pages"):
        target["host_swap_pages"] = summary["host_swap_pages"]
    started = datetime.fromtimestamp(first, tz=timezone.utc).replace(microsecond=0).isoformat()
    result = TrialResult(
        trial_id=f"{cell_id}/r{repeat}",
        cell_id=cell_id,
        repeat=repeat,
        seed=spec_seed,
        status="ok",
        started_at=started,
        duration_s=round(summary["wall_seconds"], 3),
        host_load_1m_before=0.0,
        quiet_host_ok=True,
        metrics=metrics,
        requests_per_endpoint=[len(recs)],
        target=target,
        error=None,
    )
    return result, recs


def import_batch(spec_name: str, batch: str, pattern: str, read: Any) -> Path | None:
    levels = sorted(p for p in DATA.glob(pattern) if (p / "summary.json").exists())
    if not levels:
        return None
    spec = load_spec(Path("experiments") / f"{spec_name}.yaml")
    spec_dict = spec.to_dict()
    trials, request_rows, cells = [], [], {}
    for level in levels:
        cell, repeat = read(level)
        result, recs = trial(level, cell, repeat, spec.seed)
        trials.append(result)
        cells[result.cell_id] = cell
        request_rows += [{"trial_id": result.trial_id, "cell_id": result.cell_id, **asdict(r)} for r in recs]
    present = {k: sorted({c[k] for c in cells.values()}, key=str) for k in next(iter(cells.values()))}
    spec_dict["matrix"] = present  # the cells this batch recorded
    starts = sorted(t.started_at for t in trials)
    run_id = f"{starts[0].replace('-', '').replace(':', '')[:15]}Z-{spec.name}-{batch}-imported"
    result = ExperimentResult(
        schema_version=RESULT_SCHEMA_VERSION,
        run_id=run_id,
        name=spec.name,
        description=spec.description,
        spec=spec_dict,
        provenance=Provenance(
            git_commit="unknown",
            git_dirty=False,
            spec_fingerprint=stable_config_fingerprint(spec_dict),
            started_at=starts[0],
            finished_at=starts[-1],
            host={},
            tools={"imported_from": pattern, "recorded_by": "MEMTRACE engine bench (before the harness)"},
        ),
        trials=sorted(trials, key=lambda t: (t.repeat, t.cell_id)),
        cells=aggregate_cells(trials, dict(sorted(cells.items()))),
    )
    out = OUT / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    (out / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in request_rows))
    return out


def main() -> None:
    for batch in BATCHES:
        out = import_batch(*batch)
        print(f"{batch[0]:28} {batch[1]:6} -> {out}")


if __name__ == "__main__":
    main()
