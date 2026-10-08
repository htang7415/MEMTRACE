"""Benchmark one OpenAI-compatible engine on a public-trace workload at several concurrency levels."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memtrace.backends.models.openai_runner import OpenAICompatibleActorModel
from memtrace.serving.metrics import summarize_requests
from memtrace.serving import workloads
from memtrace.serving.bench import (
    PowerSampler,
    run_sessions,
    host_swap_pages,
    pool_delta,
    run_requests,
    scrape_pool,
    scrape_prefix_cache,
)

_WORKLOAD_FILES = {
    "sharegpt": "sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json",
    "mooncake-toolagent": "mooncake/toolagent_trace.jsonl",
    "mooncake-conversation": "mooncake/conversation_trace.jsonl",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, help="Label for the engine under test, e.g. vllm-metal")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, e.g. http://localhost:8200/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--workload", choices=sorted([*_WORKLOAD_FILES, "agent-sessions", "copilot-agent"]), required=True
    )
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--concurrency", default="1,2,4,8", help="Comma-separated concurrency levels")
    parser.add_argument("--block-tokens", type=int, default=32, help="Words per Mooncake hash block")
    parser.add_argument("--max-blocks", type=int, default=100, help="Keep only the first N Mooncake blocks")
    parser.add_argument("--sessions", type=int, default=32, help="agent-sessions: concurrent sessions")
    parser.add_argument("--turns", type=int, default=8, help="agent-sessions: turns per session")
    parser.add_argument(
        "--copilot-shards",
        default="copilot_agent/date=2026-06-06/shard-0000.jsonl.gz",
        help="copilot-agent: comma-separated trace shards under --data-dir",
    )
    parser.add_argument("--token-scale", type=float, default=1 / 40, help="copilot-agent: token length scale")
    parser.add_argument("--max-calls", type=int, default=40, help="copilot-agent: first N LLM calls per session")
    parser.add_argument("--gap-scale", type=float, default=0.1, help="copilot-agent: inter-call gap compression")
    parser.add_argument("--max-gap", type=float, default=30.0, help="copilot-agent: cap on compressed gaps (s)")
    parser.add_argument("--window-seconds", type=float, default=300.0, help="copilot-agent: session start window")
    parser.add_argument(
        "--reuse",
        choices=("observed", "full"),
        default="observed",
        help="copilot-agent: reuse the provider's observed cached prefix, or the whole previous prompt (append-only)",
    )
    parser.add_argument("--slo-ttft", type=float, default=2.0, help="TTFT SLO in seconds")
    parser.add_argument("--slo-tpot", type=float, default=0.1, help="TPOT SLO in seconds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--idle-seconds", type=float, default=5.0, help="Idle power baseline before each level")
    parser.add_argument("--data-dir", type=Path, default=Path("data/public"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/engine_bench"))
    parser.add_argument("--no-reset-prefix-cache", action="store_true", help="Keep the prefix cache between levels")
    parser.add_argument("--ignore-eos", action="store_true", help="Ask the engine to generate exactly max_tokens")
    parser.add_argument("--k8s-context", help="With --k8s-pool: kubectl context of the cluster")
    parser.add_argument("--k8s-pool", help="Label selector of InferencePool pods to read per-pod metrics from")
    args = parser.parse_args(argv)

    path = args.data_dir / _WORKLOAD_FILES.get(args.workload, "")
    sessions: list[workloads.AgentSession] = []
    requests: list[workloads.Request] = []
    if args.workload == "copilot-agent":
        sessions = workloads.copilot_sessions(
            [args.data_dir / shard for shard in args.copilot_shards.split(",")],
            num_sessions=args.sessions,
            seed=args.seed,
            token_scale=args.token_scale,
            max_calls=args.max_calls,
            gap_scale=args.gap_scale,
            max_gap_seconds=args.max_gap,
            window_seconds=args.window_seconds,
            output_range=(4, args.max_tokens),
            reuse=args.reuse,
        )
    elif args.workload == "agent-sessions":
        requests = workloads.agent_sessions(
            num_sessions=args.sessions, turns=args.turns, max_tokens=args.max_tokens, seed=args.seed
        )
    elif args.workload == "sharegpt":
        requests = workloads.sharegpt(path, num_requests=args.num_requests, max_tokens=args.max_tokens, seed=args.seed)
    else:
        requests = workloads.mooncake(
            path,
            num_requests=args.num_requests,
            max_tokens=args.max_tokens,
            block_tokens=args.block_tokens,
            max_blocks=args.max_blocks,
        )

    model = OpenAICompatibleActorModel(
        args.model, base_url=args.base_url, max_attempts=1, extra_body={"ignore_eos": True} if args.ignore_eos else None
    )
    run_dir = args.out_dir / args.engine / args.workload
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "manifest.json",
        {
            "engine": args.engine,
            "base_url": args.base_url,
            "model": args.model,
            "workload": args.workload,
            "engine_version": _engine_version(args.base_url),
            "dataset": _dataset_description(args),
            "num_requests": len(requests) or sum(len(session.calls) for session in sessions),
            "truncated_requests": sum(1 for request in requests if request.truncated_blocks),
            "replay": _replay_description(args, sessions) if sessions else None,
            "settings": {k: str(v) for k, v in vars(args).items()},
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    levels = [0] if sessions else [int(level) for level in args.concurrency.split(",")]
    for concurrency in levels:
        cache_reset = False if args.no_reset_prefix_cache else _reset_prefix_cache(args.base_url)
        for warmup in ("Warm up. Reply with one word.", "Second warm-up. Reply with one word."):
            model.generate(warmup, max_tokens=8)
        with PowerSampler() as idle:
            time.sleep(args.idle_seconds)
        cache_before = scrape_prefix_cache(args.base_url)
        pool_before = scrape_pool(args.k8s_context, args.k8s_pool) if args.k8s_pool else None
        swap_before = host_swap_pages()
        with PowerSampler() as power:
            if sessions:  # open loop: concurrency is whatever the replay produces
                records, wall, concurrency = run_sessions(model, sessions)
            else:
                records, wall = run_requests(model, requests, concurrency)
        cache_after = scrape_prefix_cache(args.base_url)
        pool_after = scrape_pool(args.k8s_context, args.k8s_pool) if args.k8s_pool else None
        swap_after = host_swap_pages()

        summary = summarize_requests(
            records,
            wall_seconds=wall,
            concurrency=concurrency,
            slo_ttft_seconds=args.slo_ttft,
            slo_tpot_seconds=args.slo_tpot,
        )
        summary["prefix_cache"] = _prefix_cache_summary(records, cache_before, cache_after, cache_reset)
        if swap_before and swap_after:
            # Paging during the run, not swap size, is what distorts latency on a memory-tight host.
            summary["host_swap_pages"] = {key: swap_after[key] - swap_before[key] for key in swap_after}
        if pool_before is not None and pool_after is not None:
            summary["pool"] = pool_delta(pool_before, pool_after)
        summary["power"] = power_summary = power.summary()
        summary["idle_power"] = idle_summary = idle.summary()
        if power_summary and idle_summary and summary["output_tokens"]:
            added_watts = power_summary["mean_sys_w"] - idle_summary["mean_sys_w"]
            summary["energy_joules_per_output_token"] = added_watts * wall / summary["output_tokens"]

        if sessions:
            summary["open_loop"] = {"sessions": len(sessions), "calls": len(records), "peak_in_flight": concurrency}
        level_dir = run_dir / ("open" if sessions else f"c{concurrency}")
        level_dir.mkdir(exist_ok=True)
        (level_dir / "requests.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
        (level_dir / "power.jsonl").write_text("".join(json.dumps(sample) + "\n" for sample in power.samples))
        _write_json(level_dir / "summary.json", summary)
        latency = summary["latency"]
        print(
            f"{args.engine} {args.workload} {'open-loop peak' if sessions else 'c'}={concurrency}: "
            f"{summary['output_tokens_per_second']:.1f} out tok/s, "
            f"TTFT p95 {_p(latency, 'ttft_seconds')}, TPOT p95 {_p(latency, 'tpot_seconds')}, "
            f"SLO {summary['slo_attainment']:.0%}, errors {summary['errors']}"
        )


def _dataset_description(args: argparse.Namespace) -> dict[str, Any]:
    if args.workload == "agent-sessions":
        return {"synthetic": "agent-sessions", "sessions": args.sessions, "turns": args.turns}
    if args.workload == "copilot-agent":
        return {
            "name": "GitHubCopilotCodingAgentDataset2026",
            "license": "CC-BY-4.0",
            "shards": args.copilot_shards.split(","),
        }
    return _dataset_entry(args.data_dir, _WORKLOAD_FILES[args.workload])


def _replay_description(args: argparse.Namespace, sessions: list[workloads.AgentSession]) -> dict[str, Any]:
    calls = [call for session in sessions for call in session.calls]
    prompt = sum(call.prompt_target for call in calls)
    return {
        "token_scale": args.token_scale,
        "gap_scale": args.gap_scale,
        "max_gap_seconds": args.max_gap,
        "capped_gaps": sum(1 for call in calls if call.gap_before >= args.max_gap),
        "window_seconds": args.window_seconds,
        "reuse": args.reuse,
        "max_calls_per_session": args.max_calls,
        "sessions": len(sessions),
        "calls": len(calls),
        "synthesized_cached_share": sum(call.cached_target for call in calls) / prompt if prompt else None,
        "prompt_capped_calls": sum(1 for call in calls if call.prompt_target >= 3500),
    }


def _engine_version(base_url: str) -> str | None:
    """vLLM's GET /version, so each run records the exact engine build. None if the server has none."""
    url = re.sub(r"/v1/?$", "", base_url.rstrip("/")) + "/version"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return str(json.loads(response.read()).get("version")) or None
    except (OSError, ValueError):
        return None


def _reset_prefix_cache(base_url: str) -> bool:
    """POST /reset_prefix_cache (vLLM, requires VLLM_SERVER_DEV_MODE=1). False if unsupported."""
    url = re.sub(r"/v1/?$", "", base_url.rstrip("/")) + "/reset_prefix_cache"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="POST"), timeout=10) as response:
            return bool(200 <= response.status < 300)
    except OSError:
        return False


def _prefix_cache_summary(
    records: list[dict[str, Any]],
    before: dict[str, float] | None,
    after: dict[str, float] | None,
    reset: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {"reset_before_level": reset, "engine_hit_rate": None, "usage_hit_rate": None}
    if before and after:
        queries = after["vllm:prefix_cache_queries_total"] - before["vllm:prefix_cache_queries_total"]
        hits = after["vllm:prefix_cache_hits_total"] - before["vllm:prefix_cache_hits_total"]
        result["engine_hit_rate"] = hits / queries if queries else None
    reported = [r for r in records if r.get("ok") and r.get("cached_prompt_tokens") is not None]
    prompt_tokens = sum(r.get("prompt_tokens") or 0 for r in reported)
    if prompt_tokens:
        result["usage_hit_rate"] = sum(r["cached_prompt_tokens"] for r in reported) / prompt_tokens
    return result


def _dataset_entry(data_dir: Path, rel: str) -> dict[str, Any]:
    manifest_path = data_dir / "MANIFEST.json"
    name, _, file_rel = rel.partition("/")
    if not manifest_path.exists():
        return {"path": rel}
    entry = json.loads(manifest_path.read_text()).get(name, {})
    return {"path": rel, "license": entry.get("license"), **entry.get("files", {}).get(file_rel, {})}


def _p(latency: dict[str, Any], field: str) -> str:
    stats = latency.get(field)
    return f"{stats['p95']:.3f}s" if stats else "n/a"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
