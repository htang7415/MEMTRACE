"""The block simulator on Copilot-style sessions in trace time (the M1/M2/M3 analyses' setting).

One-token blocks keep the expected numbers exact."""

from __future__ import annotations

import math

import pytest

from memtrace.datasets.loaders.copilot import TraceCall, TraceSession
from memtrace.kv.retention import later_calls
from memtrace.kv.sessions import copilot_session
from memtrace.kv.sim import SimParams, simulate


def call(start: float, prompt: int, *, completion: int = 0, model: str = "A", turn_end: bool = False) -> TraceCall:
    return TraceCall(start, start + 1, model, prompt, 0, completion, turn_end)


def run(sessions: list[TraceSession], record: dict | None = None, **overrides: object) -> dict[str, float]:
    base: dict[str, object] = {
        "replicas": 1,
        "concurrency": 0,
        "capacity_tokens": 10**9,
        "routing": "kv_aware",
        "warmup_s": 0.0,
        "block_tokens": 1,
        "arrivals": "trace",
    }
    base.update(overrides)
    built = [copilot_session(s, 1) for s in sessions]
    return simulate(built, SimParams(**base), seed=0, record=record)  # type: ignore[arg-type]


def reused(m: dict[str, float]) -> float:
    """Prompt tokens served from any tier."""
    total = m["prompt_tokens_per_request"] * m["requests_measured"]
    return total * (m["token_hit_rate"] + m["cpu_loaded_share"] + m["ssd_loaded_share"])


def test_copilot_session_shares_the_reusable_prefix_and_starts_over_after_a_compaction_or_model_switch() -> None:
    s = copilot_session(
        TraceSession(
            "a", (call(0, 100), call(3, 150), call(5, 120, model="B"), call(7, 200, model="B"), call(9, 50, model="B"))
        ),
        block_tokens=10,
    )
    blocks = [r.blocks.tolist() for r in s.requests]
    assert blocks[1][:10] == blocks[0] and len(blocks[1]) == 15  # 100 shared, 50 new
    assert not set(blocks[2]) & set(blocks[1])  # model switch
    assert blocks[3][:12] == blocks[2]
    assert not set(blocks[4]) & set(blocks[3])  # 50 <= 0.9 x 200: compaction
    assert s.start == 0 and [r.t for r in s.requests] == [0, 3, 5, 7, 9]
    assert s.requests[0].next_t == 3 and math.isinf(s.requests[-1].next_t)


@pytest.mark.parametrize("lifetime", [None, 300.0, 30.0, 2.5])
def test_unlimited_single_replica_with_a_lifetime_matches_m1_retained(lifetime: float | None) -> None:
    sessions = [
        TraceSession("a", (call(0, 100), call(3, 150), call(40, 200), call(400, 210), call(402, 100))),
        TraceSession("b", (call(1, 500), call(2.5, 600, model="B"), call(5, 650, model="B"))),
    ]
    m = run(sessions, ttl_s=lifetime)
    retained = sum(item.retained(math.inf if lifetime is None else lifetime) for item in later_calls(sessions))
    reusable = sum(item.retained(math.inf) for item in later_calls(sessions))
    assert reused(m) == pytest.approx(retained)
    assert m["hit_share_of_reusable"] == pytest.approx(retained / reusable)


def test_turn_eviction_drops_kv_of_sessions_waiting_for_their_user_first() -> None:
    sessions = [
        TraceSession("b", (call(0, 100), call(3, 100))),  # tool loop: back 2 s after its KV is released
        TraceSession("a", (call(0.5, 100, turn_end=True),)),  # turn ended: waits for its user
        TraceSession("c", (call(2, 100),)),  # needs room for 100 while b and a sit in the cache
    ]
    # LRU evicts b (released first), so b's return misses; turn eviction evicts a instead.
    assert reused(run(sessions, capacity_tokens=200)) == 0
    assert reused(run(sessions, capacity_tokens=200, eviction="turn")) == 100


def test_evicted_kv_is_demoted_down_the_tiers_and_loaded_back() -> None:
    sessions = [
        TraceSession("a", (call(0, 100), call(10, 100))),
        TraceSession("b", (call(2, 100),)),  # demotes a to the CPU tier
        TraceSession("c", (call(4, 100),)),  # demotes b to the CPU tier and a to the SSD tier
    ]
    gpu_only = run(sessions, capacity_tokens=100)
    cpu = run(sessions, capacity_tokens=100, cpu_capacity_tokens=200)
    ssd = run(sessions, capacity_tokens=100, cpu_capacity_tokens=100, ssd_capacity_tokens=100)
    assert reused(gpu_only) == 0
    assert cpu["cpu_loaded_share"] * 400 == pytest.approx(100) and cpu["ssd_loaded_share"] == 0
    assert ssd["ssd_loaded_share"] * 400 == pytest.approx(100) and ssd["cpu_loaded_share"] == 0
    assert ssd["prefill_cost_tokens_per_request"] * 4 == pytest.approx(300 + 0.5 * 100)


def test_routers_and_spill() -> None:
    sessions = [
        TraceSession("a", (call(0, 100), call(2, 100), call(4, 100))),
        TraceSession("z", (call(1.5, 10),)),  # busy on replica 0 while a's second call arrives
    ]
    # Least-loaded sends a's second call to idle replica 1 (a miss); its third goes back to replica 0.
    assert reused(run(sessions, replicas=2, routing="least_loaded")) == 100
    # kv_aware follows the KV: both later calls hit.
    assert reused(run(sessions, replicas=2, routing="kv_aware")) == 200
    # With one call per replica, the busy preferred replica spills a's second call.
    assert reused(run(sessions, replicas=2, routing="kv_aware", max_inflight=1)) == 100
    # Sticky stays with the replica of the previous call.
    assert reused(run(sessions, replicas=2, routing="sticky")) == 200


def test_memory_time_is_blocks_held_over_time() -> None:
    sessions = [TraceSession("a", (call(0, 100, completion=20), call(10, 100)))]
    m = run(sessions, ttl_s=5.0)
    # 120 blocks while the first call runs (0..1), its 100 prompt blocks until they expire at 6, then the second
    # call's 100 while it runs (10..11), which ends the replay.
    assert m["gpu_token_hours"] == pytest.approx((120 + 500 + 100) / 3600)


def test_zero_and_negative_duration_calls_end_at_their_start() -> None:
    instant = TraceCall(0, 0, "A", 100, 0, 0)
    assert reused(run([TraceSession("a", (instant, TraceCall(5, 6, "A", 120, 0, 0)))])) == 100
    backwards = TraceSession("a", (TraceCall(10, 4, "A", 100, 0, 0), TraceCall(12, 13, "A", 120, 0, 0)))
    assert reused(run([backwards], ttl_s=3.0)) == 100


def test_overlapping_calls_share_blocks_until_both_end() -> None:
    session = TraceSession(
        "a",
        (TraceCall(0, 5, "A", 100, 0, 0), TraceCall(3, 6, "A", 100, 0, 0), TraceCall(20, 21, "A", 100, 0, 0)),
    )
    m = run([session], ttl_s=2.0)
    # 100 shared blocks held 0..6, then until they expire at 8; the third call's 100 from 20 to 21.
    assert m["gpu_token_hours"] == pytest.approx((600 + 200 + 100) / 3600)
    assert reused(m) == 100


def test_per_request_hits_are_recorded_on_request() -> None:
    hits: dict = {}
    run([TraceSession("a", (call(0, 100), call(3, 150), call(400, 200)))], record=hits, ttl_s=300.0)
    assert hits == {(0, 0): 0, (0, 1): 100, (0, 2): 0}  # the third call comes after the lifetime


def test_lower_tiers_expire_with_the_lifetime_too() -> None:
    sessions = [
        TraceSession("a", (call(0, 100), call(30, 100))),
        TraceSession("b", (call(2, 100),)),  # demotes a to the CPU tier
    ]
    assert reused(run(sessions, capacity_tokens=100, cpu_capacity_tokens=100)) == 100
    assert reused(run(sessions, capacity_tokens=100, cpu_capacity_tokens=100, ttl_s=20.0)) == 0
