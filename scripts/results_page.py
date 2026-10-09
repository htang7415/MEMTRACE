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


def commit(made_by: dict[str, Any] | None) -> str:
    if not made_by or not made_by.get("git_commit"):
        return "commit not recorded"
    dirty = " (uncommitted changes)" if made_by.get("git_dirty") else ""
    return f"commit `{made_by['git_commit'][:7]}`{dirty}"


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def pick(runs: list[dict[str, Any]], **k: Any) -> dict[str, Any]:
    return next(r for r in runs if all(r.get(a) == b for a, b in k.items()))


def phase6_section() -> list[str]:
    lines = [
        "## Agent-session KV memory (Phase 6)",
        "",
        "How much KV cache to keep for agent sessions, for how long, in which memory tier, and how to route each call",
        "back to it. Data: all seven days of the GitHub Copilot coding-agent traces (Azure, June 1-7, 2026; 301k",
        "sessions), which record per LLM call the prompt, cached, and completion tokens and the timing, but no text.",
        "",
        "Terms used in every table below:",
        "",
        "- **Reusable prefix** of a later call: `min(prompt, previous prompt)` tokens, when the model is unchanged and the",
        "  prompt did not shrink by 10% or more. Shares in the lifetime, simulator, and pilot tables are shares of it.",
        "- **Waste**: overlap with the previous prompt that the provider did not serve from cache.",
        "- **Short-gap miss** (gap < 10 s): the cache should still exist, so the miss is a routing (placement) miss, an",
        "  eviction under memory pressure, or an edit of earlier content; token counts cannot tell these apart.",
        "- The provider's cached counts reflect one unnamed provider's policy, and that provider also caches prefixes",
        "  shared across sessions (see *first-call cached*), which this analysis credits to the session's own history.",
        "",
    ]
    m1 = DATA / "kvmem" / "m1-week.json"
    if m1.exists():
        doc = json.loads(m1.read_text())
        week = doc["days"]
        rows = []
        for day, s in week.items():
            c = {k: v["share"] for k, v in s["waste_by_cause"].items()}
            rows.append(
                [
                    day,
                    f"{s['sessions']:,}",
                    pct(s["waste_share_of_later_prompt"]),
                    pct(c["expiry"]),
                    pct(c["short_gap_grew"]),
                    pct(c["short_gap_shrank"]),
                    pct(c["mid_gap"]),
                    pct(c["compaction"]),
                    pct(c["model_switch"]),
                    pct(s["first_call_cached_share"]),
                ]
            )
        lines += [
            f"### What the provider recomputed (`memtrace report retention --by-day`, {commit(doc.get('provenance'))})",
            "",
            "Waste as a share of later calls' prompt tokens, and how the waste splits by cause.",
            "",
            *table(
                [
                    "Day",
                    "Sessions",
                    "Waste",
                    "Expiry (gap >= 5 min)",
                    "Short gap, prompt grew",
                    "Short gap, prompt shrank",
                    "Gap 10 s-5 min",
                    "Compaction",
                    "Model switch",
                    "First-call cached",
                ],
                rows,
            ),
        ]
        bins = list(next(iter(week.values()))["hit_by_gap"])
        rows = [[day] + [pct(s["hit_by_gap"][b]["hit_ratio_tokens"]) for b in bins] for day, s in week.items()]
        lines += [
            "### Provider cache-hit ratio by gap since the session's previous call",
            "",
            "Cached tokens / prompt tokens, summed over the later calls in each gap bin.",
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
            "### What a cache lifetime would keep, with every call routed to its cache",
            "",
            "Share of the reusable prefix served (*Observed*: what the provider served). Working set: mean KV held, in TB",
            "of bf16 KV at Qwen3-4B size (about 15 GB for a 100k-token context), within each trace day. These are upper",
            "bounds: the reusable prefix assumes append-only prompts.",
            "",
            *table(["Day", "Observed", "5 min", "1 h", "24 h", "TB @ 5 min", "TB @ 1 h", "TB @ 24 h"], rows),
        ]
    routers = ["least-loaded", "session-key", "sticky", "kv-aware"]
    retentions = ["lru", "ttl-5min", "ttl-1h", "session"]
    tiers = ["none", "ram256", "ram1024", "ram1024+ssd4000"]
    simulator_terms = [
        "Simulated with `memtrace report kv-sim`: each replica caches one prefix per session; GPU budget is per replica",
        "for idle prefixes; overflow is demoted to RAM, then SSD (assumed 50 and 7 GB/s; recompute 10k tokens/s).",
        "Routers are idealized, not llm-d's scorers: *session-key* hashes the session; *sticky* returns to the replica",
        "it last used (the idealized form of llm-d's approximate index); *kv-aware* knows where the KV is, instantly",
        "(the idealized precise index). A full replica (8 calls in flight) spills to the least-loaded one.",
        "Retention: *lru*; *ttl-5min* / *ttl-1h* drop KV idle that long; *session* demotes the KV of sessions whose turn",
        "ended (no further tool call) first.",
        "",
    ]
    lines += ["### Simulated replica pool", "", *simulator_terms]
    for day, replicas in (("2026-06-03", 64), ("2026-06-06", 16)):
        runs = jsonl(DATA / "kvmem" / f"m2-{day}.jsonl")
        if not runs:
            continue
        rows = [
            [
                f"{g} GB",
                *(
                    pct(pick(runs, gpu_gb=g, lower="none", retention="lru", router=rt)["hit_share_of_reusable"])
                    for rt in routers
                ),
            ]
            for g in (16, 32, 64)
        ]
        tier_rows = [
            [
                f"{g} GB",
                t,
                *(
                    pct(pick(runs, gpu_gb=g, lower=t, retention=rt, router="sticky")["hit_share_of_reusable"])
                    for rt in retentions
                ),
            ]
            for g in (16, 64)
            for t in tiers
        ]
        lines += [
            f"#### {day}: {replicas} replicas, Qwen3-4B bf16 KV ({commit(runs[0].get('provenance'))})",
            "",
            "Share of the reusable prefix served, by router (GPU tier only, LRU):",
            "",
            *table(["GPU budget per replica", *routers], rows),
            "By memory tier and retention (sticky router; `ram256` = 256 GB host RAM per replica):",
            "",
            *table(["GPU budget", "Lower tiers", *retentions], tier_rows),
        ]
    weekday = jsonl(DATA / "kvmem" / "m2-2026-06-03.jsonl")
    if weekday:
        candidates = [r for r in weekday if r["router"] == "sticky" and r["gpu_gb"] == 16]
        rows = [
            [
                pct(r["hit_share_of_reusable"]),
                f"{sum(v for k, v in r['gb_hours'].items() if k != 'gpu') / 1000:,.0f}",
                r["lower"],
                r["retention"],
            ]
            for r in sorted(frontier(candidates), key=lambda r: r["hit_share_of_reusable"])
        ]
        lines += [
            "#### Reuse against memory-time (2026-06-03, 16 GB GPU per replica, sticky router)",
            "",
            "Configurations no other one beats on both reuse and RAM/SSD memory-time (GPU memory-time is nearly the same).",
            "",
            *table(["Reusable prefix served", "RAM + SSD (TB-hours)", "Lower tiers", "Retention"], rows),
        ]
    sticky = jsonl(DATA / "kvmem" / "m2-2026-06-03-sticky.jsonl")
    if sticky and weekday:
        rows = []
        for lower in ("none", "ram256"):
            base = pick(weekday, gpu_gb=16, lower=lower, retention="lru", router="sticky")["hit_share_of_reusable"]
            aware = pick(weekday, gpu_gb=16, lower=lower, retention="lru", router="kv-aware")["hit_share_of_reusable"]
            cells = [
                pct(pick(sticky, gpu_gb=16, lower=lower, sticky_idle=t)["hit_share_of_reusable"])
                for t in (10, 30, 60, 300)
            ]
            rows.append([lower, pct(base), *cells, pct(aware)])
        lines += [
            "#### Can sticky routing approximate kv-aware by forgetting idle sessions? (2026-06-03, 16 GB GPU, LRU)",
            "",
            *table(["Lower tiers", "sticky", "forget after 10 s", "30 s", "60 s", "300 s", "kv-aware"], rows),
        ]
    big = jsonl(DATA / "kvmem" / "m2-2026-06-06-32b.jsonl")
    fast = jsonl(DATA / "kvmem" / "m2-2026-06-06-fastprefill.jsonl")
    base6 = jsonl(DATA / "kvmem" / "m2-2026-06-06.jsonl")
    if big and fast and base6:
        rows = []
        for g in (16, 64):
            for lower in ("none", "ram256"):
                key = {"gpu_gb": g, "lower": lower, "retention": "lru", "router": "kv-aware"}
                rows.append(
                    [
                        f"{g} GB",
                        lower,
                        pct(pick(base6, **key)["hit_share_of_reusable"]),
                        pct(pick(big, **key)["hit_share_of_reusable"]),
                        pct(pick(fast, **key)["hit_share_of_reusable"]),
                    ]
                )
        lines += [
            "#### Sensitivity (2026-06-06, kv-aware router, LRU)",
            "",
            *table(["GPU budget", "Lower tiers", "Qwen3-4B KV", "Qwen3-32B KV", "Recompute 100k tokens/s"], rows),
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
            f"### Simulator vs a real engine (`scripts/studies/m3_pilot.sh`, {commit(v.get('provenance'))})",
            "",
            f"One `vllm-metal` replica (Qwen3-0.6B), KV fixed at {v['kv_tokens']:,} tokens, LRU; 120 Copilot sessions",
            "replayed with append-only prompts at their real gaps (capped at 10 min); predictions fixed before the run.",
            "This validates the simulator for one replica's GPU-tier eviction only, not routing, memory tiers, lifetimes,",
            "or the session policy. One run; the host swapped heavily during it, which changes timing but not which",
            "prefixes are cached (the simulator replays the measured timing).",
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
        "prompts, outputs, or trace records. All runs are on one Apple Silicon Mac. Reports:",
        "[results dashboard](https://htang7415.github.io/MEMTRACE/) (source: `../dashboard/`) and",
        "[memory-risk benchmark](https://htang7415.github.io/MEMTRACE/results/benchmark/memtrace_results.html) (source: `benchmark/`).",
        "",
        *phase6_section(),
        *platform_section(),
    ]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines).rstrip() + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
