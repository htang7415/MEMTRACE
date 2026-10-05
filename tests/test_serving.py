import pytest

from memtrace.commands import run_pilot as run_pilot_module
from memtrace.evaluation.serving import summarize_serving


def _call(role, ttft, e2e, prompt_tokens=100, cached=None, output_tokens=10):
    return {
        "role": role,
        "ttft_seconds": ttft,
        "tpot_seconds": 0.05,
        "e2e_seconds": e2e,
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached,
        "output_tokens": output_tokens,
        "retries": 0,
    }


def test_summarize_serving_reports_throughput_percentiles_and_cache_hits() -> None:
    traces = [
        [{"inference_calls": [_call("writer", 1.0, 2.0, cached=50), _call("planner", 3.0, 4.0, cached=0)]}],
        [{"inference_calls": [_call("writer", 2.0, 3.0)]}, {"inference_calls": None}],
    ]

    summary = summarize_serving(traces, wall_seconds=10.0, concurrency=4)

    assert summary["episodes"] == 2
    assert summary["calls"] == 3
    assert summary["episodes_per_second"] == 0.2
    assert summary["output_tokens_per_second"] == 3.0
    # only calls that report cached tokens count toward the hit rate: 50 / (100 + 100)
    assert summary["prefix_cache_hit_rate"] == 0.25
    assert summary["latency"]["ttft_seconds"]["p50"] == 2.0
    assert summary["latency_by_role"]["writer"]["e2e_seconds"]["mean"] == 2.5
    assert summary["latency_by_role"]["planner"]["ttft_seconds"]["p99"] == 3.0


def test_summarize_serving_returns_none_without_telemetry() -> None:
    assert summarize_serving([[{"inference_calls": None}]], wall_seconds=1.0, concurrency=1) is None


def test_run_pilot_rejects_concurrency_without_openai_backend(monkeypatch) -> None:
    monkeypatch.setattr(run_pilot_module, "MEMORY_WRITER_BACKEND", "mlx")
    monkeypatch.setattr(run_pilot_module, "PLANNER_BACKEND", "mlx")

    with pytest.raises(SystemExit):
        run_pilot_module.main(["--concurrency", "4", "--dry-run"])
