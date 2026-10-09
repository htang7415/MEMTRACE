"""Serving telemetry summary from trace `inference_calls` (OpenAI-compatible backend)."""

from __future__ import annotations

from typing import Any

import numpy as np

_LATENCY_FIELDS = ("ttft_seconds", "tpot_seconds", "e2e_seconds")


def summarize_serving(
    traces: list[list[dict[str, Any]]],
    *,
    wall_seconds: float,
    concurrency: int,
) -> dict[str, Any] | None:
    """Return latency percentiles, throughput, and prefix-cache hit rate, or None without telemetry."""
    calls = [call for trace in traces for turn in trace for call in (turn.get("inference_calls") or [])]
    if not calls:
        return None

    prompt_tokens = sum(call.get("prompt_tokens") or 0 for call in calls)
    output_tokens = sum(call.get("output_tokens") or 0 for call in calls)
    cache_reported = [call for call in calls if call.get("cached_prompt_tokens") is not None]
    cached_tokens = sum(call["cached_prompt_tokens"] for call in cache_reported)
    cache_prompt_tokens = sum(call.get("prompt_tokens") or 0 for call in cache_reported)

    return {
        "concurrency": concurrency,
        "episodes": len(traces),
        "calls": len(calls),
        "wall_seconds": wall_seconds,
        "episodes_per_second": len(traces) / wall_seconds if wall_seconds else None,
        "output_tokens_per_second": output_tokens / wall_seconds if wall_seconds else None,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "prefix_cache_hit_rate": cached_tokens / cache_prompt_tokens if cache_prompt_tokens else None,
        "retries": sum(call.get("retries") or 0 for call in calls),
        "latency": _latency_percentiles(calls),
        "latency_by_role": {
            role: _latency_percentiles([call for call in calls if call.get("role") == role])
            for role in sorted({call.get("role") for call in calls if call.get("role")})
        },
    }


def _latency_percentiles(calls: list[dict[str, Any]]) -> dict[str, dict[str, float] | None]:
    result: dict[str, dict[str, float] | None] = {}
    for field in _LATENCY_FIELDS:
        values = [call[field] for call in calls if call.get(field) is not None]
        if not values:
            result[field] = None
            continue
        p50, p95, p99 = np.percentile(values, [50, 95, 99])
        result[field] = {"p50": float(p50), "p95": float(p95), "p99": float(p99), "mean": float(np.mean(values))}
    return result
