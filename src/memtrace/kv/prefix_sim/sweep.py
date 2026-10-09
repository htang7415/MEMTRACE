"""Phase 6 M2: sweep retention policy x placement x tier sizes over one trace day; one JSON line per run.

    memtrace report kv-sim --day 2026-06-06 --replicas 16 --out data/kvmem/m2-2026-06-06.jsonl

Tier bandwidths are nominal published figures (host RAM over PCIe Gen5 x16 ~50 GB/s usable, NVMe Gen4 SSD ~7 GB/s
sequential read), and `--prefill-tps` is an assumed recompute speed; both only decide whether a lower-tier hit is
worth loading. Labelled as assumptions wherever results are reported.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

from memtrace.kv.provenance import provenance
from memtrace.kv.retention import KV_BYTES_PER_TOKEN
from memtrace.kv.prefix_sim.engine import SimConfig, simulate
from memtrace.kv.prefix_sim.policies import RETENTION, ROUTERS
from memtrace.kv.prefix_sim.tiers import Tier
from memtrace.datasets.loaders.copilot import archive_path, read_sessions

RAM_GB_PER_S = 50.0
SSD_GB_PER_S = 7.0
LOWER_TIERS = {  # per replica
    "none": (),
    "ram256": (Tier("ram", 256, RAM_GB_PER_S),),
    "ram1024": (Tier("ram", 1024, RAM_GB_PER_S),),
    "ram1024+ssd4000": (Tier("ram", 1024, RAM_GB_PER_S), Tier("ssd", 4000, SSD_GB_PER_S)),
}


def frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Runs no other run beats: none keeps at least as much of the reusable prefix with no more GPU memory-time
    and no more lower-tier memory-time (strictly better in at least one). Sorted by GPU memory-time."""

    def cost(row: dict[str, Any]) -> tuple[float, float, float]:
        lower = sum(v for k, v in row["gb_hours"].items() if k != "gpu")
        return -row["hit_share_of_reusable"], row["gb_hours"]["gpu"], lower

    costs = [cost(r) for r in rows]
    kept = [
        r
        for r, c in zip(rows, costs)
        if not any(all(o <= m for o, m in zip(other, c)) and other != c for other in costs)
    ]
    return sorted(kept, key=lambda r: (r["gb_hours"]["gpu"], -r["hit_share_of_reusable"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--day", required=True, help="trace day, e.g. 2026-06-06")
    parser.add_argument("--model", choices=sorted(KV_BYTES_PER_TOKEN), default="Qwen3-4B")
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument("--max-inflight", type=int, default=8)
    parser.add_argument("--prefill-tps", type=float, default=10_000.0)
    parser.add_argument("--gpu-gb", type=float, nargs="+", default=[16, 32, 64])
    parser.add_argument("--lower", choices=sorted(LOWER_TIERS), nargs="+", default=list(LOWER_TIERS))
    parser.add_argument(
        "--retention", choices=sorted(RETENTION), nargs="+", default=["lru", "ttl-5min", "ttl-1h", "session"]
    )
    parser.add_argument("--router", choices=ROUTERS, nargs="+", default=list(ROUTERS))
    parser.add_argument(
        "--sticky-idle", type=float, nargs="+", default=[math.inf], help="sticky router: seconds idle to forget"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sessions = list(read_sessions([archive_path(args.day)]))
    made_by = provenance()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as out:
        grid = itertools.product(args.gpu_gb, args.lower, args.retention, args.router, args.sticky_idle)
        for gpu_gb, lower, retention, router, sticky_idle in grid:
            if router != "sticky" and sticky_idle != args.sticky_idle[0]:
                continue  # sticky_idle only changes the sticky router
            config = SimConfig(
                replicas=args.replicas,
                tiers=(Tier("gpu", gpu_gb, math.inf), *LOWER_TIERS[lower]),
                kv_bytes_per_token=KV_BYTES_PER_TOKEN[args.model],
                retention=RETENTION[retention],
                router=router,
                max_inflight=args.max_inflight,
                prefill_tokens_per_second=args.prefill_tps,
                sticky_idle=sticky_idle,
            )
            row = {
                "day": args.day,
                "model": args.model,
                "replicas": args.replicas,
                "gpu_gb": gpu_gb,
                "lower": lower,
                "retention": retention,
                "router": router,
                "sticky_idle": sticky_idle if router == "sticky" else None,
                **simulate(sessions, config).summary(),
                "provenance": made_by,
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(
                f"gpu {gpu_gb:>4g} {lower:16} {retention:9} {router:13} "
                f"hit {row['hit_share_of_reusable']:.1%}  spills {row['spills']}"
            )


if __name__ == "__main__":
    main()
