"""Write results/README.md: every published experiment's cells with 95% CIs, the analyses, and the platform drills.

    python scripts/results_page.py      # after `memtrace report dashboard`; reads dashboard/public/data and data/

Tables come from the dashboard's exported snapshot, so this page and the dashboard always show the same numbers.
The drills (failover, autoscaling) and the simulator calibration are per-run tables from the retained summaries in
data/. Only aggregate numbers: no prompts, outputs, traces, or keys.
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path
from typing import Any

from memtrace.harness.results import metric_bounds

EXPORT = Path("dashboard/public/data")
DATA = Path("data")
OUT = Path("results/README.md")

# Topic -> experiments in display order (as the dashboard groups them), and the metrics shown for each.
SERVING = ["throughput", "ttft_p95_ms", "slo_attainment", "errors"]
TOPICS: list[tuple[str, list[tuple[str, list[str]]]]] = [
    (
        "Serving",
        [
            ("e7-engines-sharegpt", ["output_tokens_per_s", "ttft_p95_ms", "tpot_p50_ms"]),
            ("e7b-engines-mooncake", ["output_tokens_per_s", "ttft_p95_ms", "tpot_p50_ms"]),
            ("e1-engines-gpu", ["output_tokens_per_s", "ttft_p95_ms", "tpot_p50_ms"]),
            ("e1-engines-cpu", ["output_tokens_per_s", "ttft_p95_ms"]),
            ("e2-prefix-caching", ["ttft_p50_ms", "prefix_cache_hit_ratio"]),
            ("e6-gemini-caching", ["usd_per_1k_requests", "cached_token_ratio", "ttft_p50_ms"]),
            ("e8-routing-kind-sims", ["output_tokens_per_s", "prefix_cache_hit_ratio", "ttft_p95_ms"]),
            ("e9-hetero-pool", ["output_tokens_per_s", "ttft_p95_ms", "slo_attainment", "errors"]),
            ("e9b-hetero-policies-agent", ["output_tokens_per_s", "ttft_p95_ms", "errors"]),
            ("e9c-hetero-policies-mooncake", ["output_tokens_per_s", "ttft_p95_ms", "errors"]),
            ("e3-llmd-sim", ["goodput_rps", "ttft_p95_ms", "prefix_cache_hit_ratio"]),
            ("e3-llmd-metal", ["goodput_rps", "ttft_p95_ms"]),
            ("e11-copilot-replay", ["duration_s", "ttft_p95_ms", "errors"]),
            ("e11b-copilot-replay-4b", ["duration_s", "ttft_p95_ms", "errors"]),
            ("e11c-copilot-precise-index", ["duration_s", "ttft_p95_ms", "errors"]),
            ("e10-hosted-overflow", ["requests_per_s", "ttft_p95_ms", "errors"]),
            ("e4-hybrid-gateway", ["slo_attainment", "goodput_rps", "ttft_p99_ms"]),
            ("e4b-slo-overflow", ["slo_attainment", "goodput_rps", "ttft_p99_ms"]),
            ("e5-gemini", ["accuracy", "latency_p50_ms", "usd_per_correct"]),
            ("e5-qwen3-4b", ["accuracy", "latency_p50_ms"]),
        ],
    ),
    (
        "Agent context",
        [
            ("c1-context-policies", ["accuracy", "cost_usd_per_task", "cached_share"]),
            ("k6-copilot-context-policies", ["recomputed_tokens_per_request", "token_hit_rate"]),
            ("k7-llmd-copilot-context-policies", ["recomputed_tokens_per_request"]),
            ("k8-vllm-metal-copilot-context-policies", ["recomputed_tokens_per_request"]),
            ("k9-gateway-context-qwen3-8b", ["recomputed_tokens_per_request", "slo_attainment"]),
            ("k9b-gateway-mask-min-growth-qwen3-8b", ["recomputed_tokens_per_request", "slo_attainment"]),
            ("c2a-gateway-context", ["accuracy", "prompt_tokens_per_task", "cost_usd_per_task"]),
            ("c2b-gateway-context", ["accuracy", "prompt_tokens_per_task", "cost_usd_per_task", "cached_share"]),
        ],
    ),
    (
        "KV memory",
        [
            ("k10a-copilot-reuse-routing", ["hit_share_of_reusable", "token_hit_rate", "overflow_rate"]),
            ("k10b-copilot-reuse-tiers", ["hit_share_of_reusable", "cpu_loaded_share", "ssd_loaded_share"]),
            ("k10c-copilot-reuse-retention", ["hit_share_of_reusable", "token_hit_rate"]),
            ("k11-engine-kv-check", ["prefix_cache_hit_ratio", "duration_s"]),
        ],
    ),
]
GAP = ("<10s", "10-60s", "1-5min", "5-10min", "10-60min", ">1h")


def fmt(metric: str, value: float) -> str:
    if metric_bounds(metric)[1] == 1.0:  # a proportion
        return f"{100 * value:.1f}%"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.3g}" if abs(value) < 10 else f"{value:.1f}"


def param_text(value: Any) -> str:
    """A cell parameter as text; nested settings (a model or gateway config) as key=value pairs."""
    if isinstance(value, dict):
        return ", ".join(param_text(v) if isinstance(v, dict) else f"{k}={v}" for k, v in value.items())
    return str(value)


def cell_order(cells: list[dict[str, Any]], params: list[str]) -> list[dict[str, Any]]:
    """Cells sorted by their parameters: numbers ascending, other values in order of first appearance."""
    numeric = {
        p: all(isinstance(c["params"][p], (int, float)) and not isinstance(c["params"][p], bool) for c in cells)
        for p in params
    }
    seen = {p: list(dict.fromkeys(param_text(c["params"][p]) for c in cells)) for p in params}

    def key(c: dict[str, Any]) -> tuple[float, ...]:
        return tuple(
            float(c["params"][p]) if numeric[p] else float(seen[p].index(param_text(c["params"][p]))) for p in params
        )

    return sorted(cells, key=key)


def interval(metric: str, ci: dict[str, Any]) -> str:
    """Mean [95% CI] from 3+ repeats; mean (min–max) from 2, whose t-interval (t = 12.7) says little."""
    if ci["n"] >= 3:
        return f"{fmt(metric, ci['mean'])} [{fmt(metric, ci['ci_low'])}, {fmt(metric, ci['ci_high'])}]"
    if ci["n"] == 2:
        half = ci["std"] / math.sqrt(2)  # two values sit at mean ± std/sqrt(2)
        return f"{fmt(metric, ci['mean'])} ({fmt(metric, ci['mean'] - half)}–{fmt(metric, ci['mean'] + half)})"
    return fmt(metric, ci["mean"])


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(r) + " |" for r in rows),
        "",
    ]


def experiment(name: str, metrics: list[str], result: dict[str, Any], entry: dict[str, Any]) -> list[str]:
    cells = result["cells"]
    params = list(cells[0]["params"]) if cells else []
    shown = [m for m in metrics if any(m in c["metrics"] for c in cells)]
    rows = []
    for c in cell_order(cells, params):
        row = [param_text(c["params"][p]) for p in params] or [c["cell_id"]]
        for m in shown:
            ci = c["metrics"].get(m)
            row.append("" if ci is None else interval(m, ci))
        row.append(str(c["n_ok"]))
        rows.append(row)
    commit = (
        "imported (recorded before the harness)" if entry["git_commit"] == "unknown" else f"`{entry['git_commit'][:7]}`"
    )
    header = [p.split(".", 1)[-1] for p in params] or ["cell"]
    return [
        f"### {name}",
        "",
        result["description"].strip(),
        "",
        f"Run `{entry['run_id']}`, {commit}. n = repeats: mean [95% CI] for n ≥ 3, mean (range) for n = 2, the single"
        " value for n = 1.",
        "",
        *table([*header, *shown, "n"], rows),
    ]


def analyses_section(a: dict[str, Any]) -> list[str]:
    lines = ["## Analyses", ""]
    k = a.get("retention")
    if k:
        lines += [
            "### Provider-cache retention over the Copilot week (`memtrace report retention --by-day`)",
            "",
            f"{k['sessions']:,} sessions; commit `{k['git_commit'][:7]}`. The provider recomputed "
            f"{100 * k['waste'][0]:.1f}–{100 * k['waste'][1]:.1f}% of later calls' reusable prompt tokens.",
            "",
            *table(
                ["Gap since previous call", "Hit ratio, weekdays", "Hit ratio, weekend"],
                [
                    [b, f"{100 * w:.1f}%", f"{100 * e:.1f}%"]
                    for b, w, e in zip(k["bins"], k["hit_weekday"], k["hit_weekend"])
                ],
            ),
            *table(
                ["Cause of recomputed reusable tokens", "Share (weekdays)"],
                [[c, f"{100 * s:.1f}%"] for c, s in zip(k["causes"], k["cause_share"])],
            ),
            *table(
                ["Cache lifetime", "Reusable prefix served", "Mean KV held (TB, Qwen3-4B)"],
                [["provider (observed)", f"{100 * k['observed']:.1f}%", ""]]
                + [
                    [life, f"{100 * r:.1f}%", f"{tb:.0f}"]
                    for life, r, tb in zip(("5 min", "1 h", "24 h"), k["retained"], k["working_tb"])
                ],
            ),
        ]
    s = a.get("simulator_check")
    if s:
        lines += [
            "### KV-cache simulator vs a real engine (K11, `memtrace report kv-validate`)",
            "",
            f"One vllm-metal replica with {s['kv_tokens']:,} tokens of KV cache; commit `{s['git_commit'][:7]}`.",
            "",
            *table(
                ["Gap", "Calls", "Engine", "Simulator", "Difference (pts)"],
                [
                    [
                        b["bin"],
                        str(b["calls"]),
                        f"{100 * b['engine_hit_share']:.1f}%",
                        f"{100 * b['sim_hit_share']:.1f}%",
                        f"{b['diff_points']:+.1f}",
                    ]
                    for b in s["bins"]
                ],
            ),
        ]
    if a.get("bfcl_gate"):
        lines += [
            "### Tool-call quality across engines (BFCL simple + multiple, `memtrace evaluate bfcl`)",
            "",
            *table(
                ["Engine and concurrency", "Accuracy", "Wilson 95% CI", "Correct"],
                [
                    [
                        f"`{b['config']}`",
                        f"{100 * b['accuracy']:.1f}%",
                        f"{100 * b['wilson_95'][0]:.1f}–{100 * b['wilson_95'][1]:.1f}%",
                        f"{b['correct']}/{b['total']}",
                    ]
                    for b in a["bfcl_gate"]
                ],
            ),
        ]
    return lines


def drill_rows(study: str) -> list[list[str]]:
    rows = []
    for path in sorted(glob.glob(str(DATA / study / "**" / "summary.json"), recursive=True)):
        s = json.loads(Path(path).read_text())
        if "output_tokens_per_second" not in s:
            continue
        ttft = ((s.get("latency") or {}).get("ttft_seconds") or {}).get("p95")
        rows.append(
            [
                f"`{Path(path).parent.relative_to(DATA)}`",
                str(s.get("requests", "")),
                str(s.get("errors", "")),
                f"{s['output_tokens_per_second']:.1f}",
                f"{s.get('wall_seconds', 0):.0f}",
                "" if ttft is None else f"{ttft:.2f}",
                "" if s.get("slo_attainment") is None else f"{100 * s['slo_attainment']:.1f}%",
            ]
        )
    return rows


def drills_section() -> list[str]:
    header = ["Run", "Requests", "Errors", "Output tok/s", "Wall (s)", "TTFT p95 (s)", "SLO"]
    lines = [
        "## Platform drills and calibration",
        "",
        "Per-run tables from the retained run summaries (recorded with the engine bench).",
        "",
    ]
    for title, dirs in (
        (
            "Simulated replicas calibrated against the real GPU engine (backs E8's simulator settings)",
            ["gpu_sim_validation"],
        ),
        ("Reactive autoscaling (`scripts/drills.sh burst`)", ["autoscale", "autoscale_fixed1", "autoscale_fixed4"]),
        ("Failure handling (`scripts/drills.sh failover`)", ["failover"]),
    ):
        rows = [r for d in dirs for r in drill_rows(d)]
        if rows:
            lines += [f"### {title}", "", *table(header, rows)]
    recovery = [p.read_text().splitlines()[0] for p in sorted((DATA / "failover").glob("recovery-*.txt"))]
    if recovery:
        lines += ["Recovery: " + "; ".join(recovery) + ".", ""]
    return lines


def main() -> None:
    index = {e["name"]: e for e in json.loads((EXPORT / "index.json").read_text())["experiments"]}
    lines = [
        "# Results",
        "",
        "Every published experiment's cells, generated by `scripts/results_page.py` from the dashboard's exported",
        "snapshot (`memtrace report dashboard`), so this page and the [dashboard](https://htang7415.github.io/MEMTRACE/)",
        "show the same numbers. All runs are on one Apple Silicon Mac. Specs: `../experiments/`. Nothing here is raw",
        "traffic: no prompts, outputs, or trace records. The memory-risk benchmark has its own",
        "[report](https://htang7415.github.io/MEMTRACE/results/benchmark/memtrace_results.html) (source: `benchmark/`).",
        "",
    ]
    for topic, experiments in TOPICS:
        lines += [f"## {topic}", ""]
        for name, metrics in experiments:
            if name in index:
                result = json.loads((EXPORT / index[name]["file"]).read_text())
                lines += experiment(name, metrics, result, index[name])
    lines += analyses_section(json.loads((EXPORT / "analyses.json").read_text()))
    lines += drills_section()
    OUT.write_text("\n".join(lines).rstrip() + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
