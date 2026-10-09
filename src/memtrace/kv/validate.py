"""Phase 6 M3: does the M2 simulator predict a real engine's prefix-cache hits?

Takes an engine-bench replay run (`requests.jsonl`, best with `--reuse full` so prompts are append-only as the
simulator assumes), rebuilds it as trace sessions from the measured start times, latencies and token counts, and
replays those through the simulator with one replica of the engine's KV capacity and LRU eviction. Per gap bin it
compares the engine's cached prompt tokens with the simulator's hits, both as shares of the reusable prefix.
Simulator hits are rounded down to whole blocks, as an engine caches only full blocks.

    memtrace report kv-validate --run data/m3/pilot/requests.jsonl --kv-tokens 16384
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from memtrace.kv.provenance import provenance
from memtrace.kv.retention import gap_bin, later_calls
from memtrace.kv.prefix_sim.engine import SimConfig, simulate
from memtrace.kv.prefix_sim.policies import RETENTION
from memtrace.kv.prefix_sim.tiers import Tier
from memtrace.kv.traces import TraceCall, TraceSession

TOLERANCE_POINTS = 5.0


def sessions_from_run(records: list[dict[str, Any]]) -> tuple[list[TraceSession], dict[tuple[int, int], int]]:
    """Trace sessions from a replay's records (successful calls only), and the engine's cached tokens per call,
    keyed by (session position, call position) in the returned list."""
    by_session: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("ok") and record.get("prompt_tokens") is not None:
            by_session[record["session"]].append(record)
    sessions, cached = [], {}
    for s, key in enumerate(sorted(by_session)):
        calls = []
        for c, r in enumerate(sorted(by_session[key], key=lambda r: r["call"])):
            start = r["started_at"]
            calls.append(
                TraceCall(start, start + r["e2e_seconds"], "m", r["prompt_tokens"], 0, r.get("output_tokens") or 0)
            )
            cached[(s, c)] = r.get("cached_prompt_tokens") or 0
        sessions.append(TraceSession(str(key), tuple(calls)))
    return sessions, cached


def compare(
    sessions: list[TraceSession], engine_cached: dict[tuple[int, int], int], kv_tokens: int, block_size: int = 16
) -> dict[str, Any]:
    config = SimConfig(
        replicas=1,
        tiers=(Tier("gpu", kv_tokens / 1e9, math.inf),),
        kv_bytes_per_token=1,
        retention=RETENTION["lru"],
        router="kv-aware",
        max_inflight=10**9,
    )
    sim = simulate(sessions, config, record_calls=True).call_hits
    bins: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "reusable": 0, "engine": 0, "sim": 0})
    position = {id(call): (s, c) for s, session in enumerate(sessions) for c, call in enumerate(session.calls)}
    for item in later_calls(sessions):
        reusable = item.retained(math.inf)  # same model always; zero after a compaction
        if not reusable:
            continue
        key = position[id(item.call)]
        b = bins[gap_bin(item.gap)]
        b["calls"] += 1
        b["reusable"] += reusable
        b["engine"] += min(engine_cached[key], reusable)
        b["sim"] += sim.get(key, 0) // block_size * block_size
    rows = {}
    for label, b in bins.items():
        engine, simulated = b["engine"] / b["reusable"], b["sim"] / b["reusable"]
        rows[label] = {
            "calls": b["calls"],
            "engine_hit_share": engine,
            "sim_hit_share": simulated,
            "diff_points": (simulated - engine) * 100,
        }
    total = {k: sum(b[k] for b in bins.values()) for k in ("calls", "reusable", "engine", "sim")}
    return {
        "kv_tokens": kv_tokens,
        "by_gap": rows,
        "overall": {
            "calls": total["calls"],
            "engine_hit_share": total["engine"] / total["reusable"] if total["reusable"] else None,
            "sim_hit_share": total["sim"] / total["reusable"] if total["reusable"] else None,
        },
        "within_tolerance": all(abs(r["diff_points"]) <= TOLERANCE_POINTS for r in rows.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="requests.jsonl of an engine-bench replay")
    parser.add_argument("--kv-tokens", type=int, required=True, help="engine KV cache capacity in tokens")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.run.read_text().splitlines() if line.strip()]
    sessions, cached = sessions_from_run(records)
    result = {
        **compare(sessions, cached, args.kv_tokens, args.block_size),
        "run": str(args.run),
        "provenance": provenance(),
    }
    summary = json.dumps(result, indent=2)
    print(summary)
    if args.out:
        args.out.write_text(summary + "\n")


if __name__ == "__main__":
    main()
