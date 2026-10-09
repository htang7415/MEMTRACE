"""Targets run by the shell tooling: engines.sh engines and llm-d on kind."""

import pytest

from memtrace.harness import kind as kind_mod
from memtrace.harness.kind import Engine, KindLlmd, pool_delta, sum_counters


def test_pool_delta_reports_per_pod_requests_hits_and_imbalance() -> None:
    def counters(requests, queries, hits):
        return {
            "vllm:request_success_total": requests,
            "vllm:prefix_cache_queries_total": queries,
            "vllm:prefix_cache_hits_total": hits,
        }

    before = {"pod-a": counters(10, 100, 10), "pod-b": counters(0, 0, 0)}
    after = {"pod-a": counters(40, 500, 210), "pod-b": counters(10, 100, 0), "pod-new": counters(10, 0, 0)}

    delta = pool_delta(before, after)

    assert delta["pods"]["pod-a"] == {"requests": 30, "prefix_hit_rate": 0.5}
    assert delta["pods"]["pod-new"] == {"requests": 10, "prefix_hit_rate": None}
    assert delta["pool_prefix_hit_rate"] == 200 / 500
    assert delta["load_imbalance"] == 30 / (50 / 3)


def test_counters_accept_names_with_and_without_total_suffix() -> None:
    text = (
        "# HELP vllm:prefix_cache_hits_total hits\n"
        'vllm:prefix_cache_hits_total{engine="0"} 5\n'
        'vllm:prefix_cache_hits{model_name="m"} 7\n'
        "vllm:prefix_cache_queries 20\n"
        'vllm:prefix_cache_hits_created{engine="0"} 1.7e9\n'
    )

    totals = sum_counters(text, ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"))

    assert totals == {"vllm:prefix_cache_hits_total": 12.0, "vllm:prefix_cache_queries_total": 20.0}


def test_pool_delta_reports_hosted_spend_only_for_pods_that_spend() -> None:
    def counters(requests, spend=0.0):
        return {
            "vllm:request_success_total": requests,
            "vllm:prefix_cache_queries_total": 0,
            "vllm:prefix_cache_hits_total": 0,
            "memtrace_hosted_spend_usd": spend,
        }

    delta = pool_delta(
        {"gpu": counters(0), "gemini": counters(0, 0.5)}, {"gpu": counters(9), "gemini": counters(3, 0.52)}
    )

    assert "spend_usd" not in delta["pods"]["gpu"]
    assert delta["pods"]["gemini"]["spend_usd"] == pytest.approx(0.02)


def test_kind_target_builds_the_pool_once_and_applies_the_policy_every_trial(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        kind_mod, "_bash", lambda script, env=None: calls.append(script.removeprefix("scripts/stack.sh "))
    )
    monkeypatch.setattr(kind_mod, "scrape_pool", lambda *a: {})
    monkeypatch.setattr(KindLlmd, "prepared", None)

    with KindLlmd({"pool": "hetero", "policy": "combined", "hosted": 2}):
        pass
    with KindLlmd({"pool": "hetero", "policy": "capacity", "hosted": 2}):
        pass
    with KindLlmd({"pool": "sims", "replicas": 4, "policy": "prefix"}):
        pass

    setup = ["wait_gpu_free", "hetero", "capacity_epp", "hosted 2"]
    assert calls == [
        *setup,
        "policy combined",
        "reset_caches",
        "policy capacity",
        "reset_caches",  # same pool: not rebuilt
        "sims 4",
        "policy prefix",  # simulators restart with the policy; no cache reset needed
    ]


def test_kind_target_reports_the_pool_delta_and_its_request_model(monkeypatch) -> None:
    snapshots = iter(
        [
            {
                "pod": {
                    "vllm:request_success_total": 0.0,
                    "vllm:prefix_cache_queries_total": 0.0,
                    "vllm:prefix_cache_hits_total": 0.0,
                }
            },
            {
                "pod": {
                    "vllm:request_success_total": 4.0,
                    "vllm:prefix_cache_queries_total": 10.0,
                    "vllm:prefix_cache_hits_total": 5.0,
                }
            },
        ]
    )
    monkeypatch.setattr(kind_mod, "_bash", lambda script, env=None: None)
    monkeypatch.setattr(kind_mod, "scrape_pool", lambda *a: next(snapshots))
    target = KindLlmd({"pool": "hetero", "policy": "combined", "overlay": "qwen3-4b"})
    with target:
        collected = target.collect()
    assert collected["pool_prefix_hit_rate"] == 0.5 and collected["pods"]["pod"]["requests"] == 4.0
    assert target.request_options() == {"extra_body": {"model": "Qwen/Qwen3-4B"}}
    assert target.env == {"OVERLAY": "qwen3-4b", "MODEL_4B": "1"}


def test_engine_target_starts_and_stops_through_engines_sh(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(kind_mod, "_bash", lambda script, env=None: calls.append((script, env)))
    with Engine({"engine": "mlx-lm"}) as engine:
        assert engine.base_urls == ["http://127.0.0.1:8300"]
    assert calls == [
        ("source scripts/engines.sh && start_engine mlx-lm", {"MODEL": "Qwen/Qwen3-0.6B"}),
        ("source scripts/engines.sh && stop_engine mlx-lm", {"MODEL": "Qwen/Qwen3-0.6B"}),
    ]
    with pytest.raises(ValueError):
        Engine({"engine": "tgi"})
