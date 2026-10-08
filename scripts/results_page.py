"""Write results/README.md: aggregate tables behind the README's claims, built from the retained run summaries.

    python scripts/results_page.py          # reads data/ (git-ignored), writes results/README.md

Only aggregate numbers leave data/: no prompts, outputs, traces, or keys.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

DATA = Path("data")
OUT = Path("results/README.md")

# (heading, what it backs, study directories)
STUDIES = [
    ("Engine baselines", "Which engine per tier", ["engine_bench", "engine_bench_r2"]),
    ("Cache-aware routing (simulated pool)", "Does cache-aware routing pay", ["routing_study_v2"]),
    ("Simulator calibration against the real GPU engine", "the routing study's simulator", ["gpu_sim_validation"]),
    (
        "Reactive autoscaling",
        "Does reactive autoscaling absorb bursts",
        ["autoscale", "autoscale_fixed1", "autoscale_fixed4"],
    ),
    ("Failure handling", "Failure handling", ["failover"]),
    ("Routing a GPU + CPU pool", "Routing a GPU + CPU pool", ["hetero_reps", "hetero_reps2"]),
    ("Hosted overflow", "Hosted overflow", ["hosted"]),
    ("Real agent traffic (Copilot replay)", "Real agent traffic", ["copilot"]),
    ("Larger model on the GPU tier", "Larger model", ["copilot-4b", "copilot-4b-s32"]),
    ("Precise vs approximate prefix index", "Precise vs approximate prefix index", ["precise"]),
]


def pct(x: float | None) -> str:
    return "" if x is None else f"{x:.1%}"


def num(x: float | None, digits: int = 1) -> str:
    return "" if x is None else f"{x:.{digits}f}"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(r) + " |" for r in rows),
        "",
    ]


def run_rows(study: str) -> list[list[str]]:
    rows = []
    for path in sorted(glob.glob(str(DATA / study / "**" / "summary.json"), recursive=True)):
        s = json.loads(Path(path).read_text())
        if "output_tokens_per_second" not in s:
            continue
        run = str(Path(path).parent.relative_to(DATA))
        ttft = ((s.get("latency") or {}).get("ttft_seconds") or {}).get("p95")
        swap = (s.get("host_swap_pages") or {}).get("swapouts")
        rows.append(
            [
                f"`{run}`",
                str(s.get("requests", "")),
                str(s.get("errors", "")),
                num(s["output_tokens_per_second"]),
                num(s.get("wall_seconds"), 0),
                num(ttft, 2),
                pct(s.get("slo_attainment")),
                "" if swap is None else str(swap),
            ]
        )
    return rows


def platform_section() -> list[str]:
    lines = [
        "## Platform studies",
        "",
        "One row per retained run (`summary.json`). Swap-outs are host pages written to swap during the run; runs",
        "that paged heavily are noisy, which is why the README claims no difference under ~15%.",
        "",
    ]
    header = ["Run", "Requests", "Errors", "Output tok/s", "Wall (s)", "TTFT p95 (s)", "SLO", "Swap-outs"]
    for title, backs, dirs in STUDIES:
        rows = [r for d in dirs for r in run_rows(d)]
        if rows:
            lines += [f"### {title}", "", f'Backs the README row "{backs}".', "", *table(header, rows)]
    bfcl = []
    for path in sorted(glob.glob(str(DATA / "bfcl" / "*" / "summary.json"))):
        s = json.loads(Path(path).read_text())
        o = s["overall"]
        low, high = o["wilson_95"]
        bfcl.append(
            [f"`{Path(path).parent.name}`", pct(o["accuracy"]), f"{low:.1%}-{high:.1%}", f"{o['correct']}/{o['total']}"]
        )
    if bfcl:
        lines += [
            "### Tool-call quality (BFCL subset)",
            "",
            'Backs "BFCL tool-call accuracy 81.0-81.5% on every engine and batch size".',
            "",
            *table(["Engine and concurrency", "Accuracy", "Wilson 95% CI", "Correct"], bfcl),
        ]
    return lines


def phase6_section() -> list[str]:
    lines = [
        "## Agent-session KV memory (Phase 6)",
        "",
        "How much KV cache to keep for agent sessions, for how long, in which memory tier, and how to route each",
        "call back to it. Data: all seven days of the GitHub Copilot coding-agent traces (Azure, June 1-7, 2026; 301k",
        "sessions). The trace's cached-token counts reflect one unnamed provider's cache, so they are an observed",
        "baseline, not ground truth.",
        "",
    ]
    m1 = DATA / "kvmem" / "m1-week.json"
    if m1.exists():
        week = json.loads(m1.read_text())
        rows = []
        for day, s in week.items():
            c = {k: v["share"] for k, v in s["waste_by_cause"].items()}
            rows.append(
                [
                    day,
                    f"{s['sessions']:,}",
                    pct(s["waste_share_of_later_prompt"]),
                    pct(c["expiry"]),
                    pct(c["placement"]),
                    pct(c["placement_or_edit"]),
                    pct(c["other"]),
                    pct(c["compaction"]),
                    pct(c["model_switch"]),
                ]
            )
        lines += [
            "### Reusable prompt tokens the provider recomputed (`memtrace report retention --by-day`)",
            "",
            "Waste: tokens of a later call's prompt that repeat the previous call's prompt but were not served from",
            "cache. Placement (gap < 10 s, prompt only grew) is an upper bound: a growing prompt can still have",
            "rewritten earlier content, which token counts cannot reveal.",
            "",
            *table(
                [
                    "Day",
                    "Sessions",
                    "Waste",
                    "Expiry (gap >= 5 min)",
                    "Placement",
                    "Placement or edit",
                    "Gap 10 s-5 min",
                    "Compaction",
                    "Model switch",
                ],
                rows,
            ),
        ]
        bins = list(next(iter(week.values()))["hit_by_gap"])
        rows = [[day] + [pct(s["hit_by_gap"][b]["hit_ratio_tokens"]) for b in bins] for day, s in week.items()]
        lines += [
            "### Provider cache-hit ratio by gap since the session's previous call",
            "",
            *table(["Day", *bins], rows),
        ]
        rows = []
        for day, s in week.items():
            lt = s["lifetimes"]
            rows.append(
                [day, pct(lt["5min"]["observed_cached_share_of_reusable"])]
                + [pct(lt[k]["retained_share_of_reusable"]) for k in ("5min", "1h", "24h")]
                + [f"{lt[k]['working_set_gb']['Qwen3-4B']['mean'] / 1000:,.1f}" for k in ("5min", "1h", "24h")]
            )
        lines += [
            "### What a cache lifetime would keep with perfect placement",
            "",
            "Share of the reusable prefix served; working set is the mean KV held, in TB of bf16 KV at Qwen3-4B size",
            "(about 15 GB for a 100k-token context), measured within each trace day.",
            "",
            *table(["Day", "Observed", "5 min", "1 h", "24 h", "TB @ 5 min", "TB @ 1 h", "TB @ 24 h"], rows),
        ]
    for day, replicas in (("2026-06-03", 64), ("2026-06-06", 16)):
        path = DATA / "kvmem" / f"m2-{day}.jsonl"
        if not path.exists():
            continue
        runs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

        def get(**k: Any) -> dict[str, Any]:
            return next(r for r in runs if all(r[a] == b for a, b in k.items()))

        routers = ["least-loaded", "session-key", "approximate", "precise"]
        rows = [
            [
                f"{g:g} GB",
                *(
                    pct(get(gpu_gb=g, lower="none", retention="lru", router=rt)["hit_share_of_reusable"])
                    for rt in routers
                ),
            ]
            for g in (16, 32, 64)
        ]
        tiers = ["none", "ram256", "ram1024", "ram1024+ssd4000"]
        retentions = ["lru", "ttl-5min", "ttl-1h", "session"]
        tier_rows = [
            [
                f"{g:g} GB",
                t,
                *(
                    pct(get(gpu_gb=g, lower=t, retention=rt, router="approximate")["hit_share_of_reusable"])
                    for rt in retentions
                ),
            ]
            for g in (16, 64)
            for t in tiers
        ]
        lines += [
            f"### Simulated pool, {day} ({replicas} replicas, Qwen3-4B bf16 KV) (`memtrace report kv-sim`)",
            "",
            "Share of the reusable prefix served. GPU budget is per replica, for idle prefixes. Assumed bandwidths:",
            "RAM 50 GB/s, SSD 7 GB/s; prefill 10k tokens/s.",
            "",
            "Placement, GPU tier only, LRU:",
            "",
            *table(["GPU budget", *routers], rows),
            "Memory tiers and retention, approximate routing (`ram256` = 256 GB host RAM per replica):",
            "",
            *table(["GPU budget", "Lower tiers", *retentions], tier_rows),
        ]
    pilot = DATA / "m3" / "pilot" / "validate.json"
    if pilot.exists():
        v = json.loads(pilot.read_text())
        order = ["<10s", "10-60s", "1-5min", "5-10min", "10-60min", ">1h"]
        rows = [
            [b, str(r["calls"]), pct(r["engine_hit_share"]), pct(r["sim_hit_share"]), f"{r['diff_points']:+.1f}"]
            for b in order
            if (r := v["by_gap"].get(b))
        ]
        lines += [
            "### Simulator vs a real engine (`scripts/studies/m3_pilot.sh`, `memtrace report kv-validate`)",
            "",
            f"One `vllm-metal` replica (Qwen3-0.6B) with KV fixed at {v['kv_tokens']:,} tokens; 120 Copilot sessions",
            "replayed append-only with their real gaps (capped at 10 min). Predictions were fixed before the run.",
            "One run, one replica.",
            "",
            *table(["Gap", "Calls", "Engine", "Simulator", "Diff (points)"], rows),
        ]
    return lines


def main() -> None:
    lines = [
        "# Results",
        "",
        "Aggregate numbers behind the claims in the [README](../README.md), generated by `scripts/results_page.py`",
        "from the run summaries kept locally under `data/` (not in the repository). Nothing here is raw traffic: no",
        "prompts, outputs, or trace records. All runs are on one 16 GB Apple Silicon Mac.",
        "",
        *phase6_section(),
        *platform_section(),
    ]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines).rstrip() + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
