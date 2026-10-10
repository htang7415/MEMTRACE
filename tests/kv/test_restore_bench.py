from __future__ import annotations

import random
from typing import Any

import pytest

from memtrace.kv import restore_bench


def test_measure_sequence_and_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []
    loaded = iter([100.0, 100.0 + 4096])  # the connector counter before and after the restore

    def fake_complete(url: str, model: str, prompt: list[int], max_tokens: int, timeout_s: float) -> tuple[float, str]:
        calls.append((len(prompt), max_tokens))
        n = len(calls)
        return {1: (2.0, "abc"), 2: (0.1, "abc")}.get(n, (0.5, "abc") if max_tokens == 32 else (0.3, "x"))

    def fake_counters(url: str) -> dict[str, Any]:
        return {"external_prefix_cache_hits_total": next(loaded)}

    monkeypatch.setattr(restore_bench, "complete", fake_complete)
    monkeypatch.setattr(restore_bench, "scrape_vllm_counters", fake_counters)
    monkeypatch.setattr(restore_bench, "drain_offload", lambda url, timeout_s: 1.5)
    m = restore_bench.measure("http://x", "m", random.Random(0), prefix_tokens=4096, flush_tokens=5000,
                              filler_tokens=2048, max_tokens=32, timeout_s=1.0)  # fmt: skip
    assert calls == [
        (4096, 32),
        (4096, 32),
        (2048, 1),
        (2048, 1),
        (2048, 1),
        (4096, 32),
    ]  # cold, hit, 3 fillers, restore
    assert m["ttft_cold_ms"] == 2000 and m["ttft_gpu_hit_ms"] == 100 and m["ttft_restore_ms"] == 500
    assert m["restore_speedup"] == pytest.approx(4.0)
    assert m["restored_share"] == pytest.approx(1.0)
    assert m["output_same_as_gpu_hit_rate"] == 1.0 and m["output_same_as_cold_rate"] == 1.0
    assert m["offload_drain_s"] == 1.5


def test_committed_spec_loads() -> None:
    spec = restore_bench.load_spec(restore_bench.Path("experiments/s1_restore_bench.yaml"))
    assert spec["schema_version"] == restore_bench.SPEC_SCHEMA
    for model in spec["models"].values():
        cap = int(model["target"]["max_model_len"])
        assert cap < model["gpu_blocks"] * 16  # vLLM needs one request of max_model_len to fit the fixed cache
        assert min(model.get("max_prefix_tokens", 10**9), max(spec["prefix_tokens"])) + spec["max_tokens"] <= cap


def test_measure_without_restore_is_cold_and_gpu_hit_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_complete(url: str, model: str, prompt: list[int], max_tokens: int, timeout_s: float) -> tuple[float, str]:
        calls.append(len(prompt))
        return (1.0, "t")

    monkeypatch.setattr(restore_bench, "complete", fake_complete)
    m = restore_bench.measure("http://x", "m", random.Random(0), prefix_tokens=512, flush_tokens=10_000,
                              filler_tokens=2048, max_tokens=8, timeout_s=1.0, restore=False)  # fmt: skip
    assert calls == [512, 512] and set(m) == {"ttft_cold_ms", "ttft_gpu_hit_ms"}


def test_each_tier_repetition_gets_its_own_engine_and_store() -> None:
    spec = {"repeats": 2, "prefix_tokens": [1024, 4096], "models": {"a": {}, "b": {"max_prefix_tokens": 1024}}}
    runs = [(name, kind, items) for name, _, kind, items in restore_bench.engine_lifetimes(spec)]
    assert runs[0] == ("a", "stock", [(1024, 0), (1024, 1), (4096, 0), (4096, 1)])
    assert [items for name, kind, items in runs if name == "a" and kind == "tier"] == [
        [(1024, 0)],
        [(1024, 1)],
        [(4096, 0)],
        [(4096, 1)],
    ]
    assert [(kind, items) for name, kind, items in runs if name == "b"] == [
        ("stock", [(1024, 0), (1024, 1)]),
        ("tier", [(1024, 0)]),
        ("tier", [(1024, 1)]),
    ]
