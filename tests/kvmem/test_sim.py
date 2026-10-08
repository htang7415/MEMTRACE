from __future__ import annotations

import math

import pytest

from memtrace.kvmem.retention import later_calls
from memtrace.kvmem.sim.engine import SimConfig, simulate
from memtrace.kvmem.sim.policies import RETENTION, Retention
from memtrace.kvmem.sim.tiers import Tier
from memtrace.kvmem.traces import TraceCall, TraceSession

# One token = 1 GB keeps capacities readable: a tier of 1000 GB holds 1000 tokens.
GB = 10**9
GPU = Tier("gpu", 1e9, math.inf)


def call(start: float, prompt: int, *, completion: int = 0, model: str = "A", turn_end: bool = False) -> TraceCall:
    return TraceCall(start, start + 1, model, prompt, 0, completion, turn_end)


def config(**overrides: object) -> SimConfig:
    base: dict[str, object] = {
        "replicas": 1,
        "tiers": (GPU,),
        "kv_bytes_per_token": GB,
        "retention": RETENTION["lru"],
        "router": "precise",
        "max_inflight": 100,
        "prefill_tokens_per_second": 1.0,
    }
    base.update(overrides)
    return SimConfig(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize("lifetime", [math.inf, 300.0, 30.0, 2.5])
def test_unlimited_single_replica_matches_m1_retained(lifetime: float) -> None:
    sessions = [
        TraceSession("a", (call(0, 100), call(3, 150), call(40, 200), call(400, 210), call(402, 100))),
        TraceSession("b", (call(1, 500), call(2.5, 600, model="B"), call(5, 650, model="B"))),
    ]
    result = simulate(sessions, config(retention=Retention("t", lifetime)))
    expected = sum(item.retained(lifetime) for item in later_calls(sessions))
    assert result.hit_tokens["gpu"] == expected
    assert result.reusable_tokens == sum(item.retained(math.inf) for item in later_calls(sessions))


def test_lru_evicts_oldest_and_turn_aware_protects_in_turn_kv() -> None:
    # GPU holds 150 tokens: one session's 100-token context at a time, plus change.
    small_gpu = (Tier("gpu", 150.0, math.inf),)
    sessions = [
        TraceSession("a", (call(0, 100, turn_end=True), call(10, 100))),  # turn ends; user returns at 10
        TraceSession("b", (call(2, 100), call(4, 100))),  # tool loop: b's KV is needed 1 s after it is written
    ]
    # LRU: b's write at 3 evicts a; b's write at 5 evicts nothing (b replaced itself). a misses, b hits.
    lru = simulate(sessions, config(tiers=small_gpu))
    assert lru.hit_tokens["gpu"] == 100
    # When the idle session wrote last, LRU evicts the wrong one.
    flipped = [
        TraceSession("b", (call(0, 100), call(4, 100))),  # in-turn, written at 1
        TraceSession("a", (call(1.5, 100, turn_end=True), call(10, 100))),  # turn ends, written at 2.5
    ]
    # LRU evicts b (oldest) at 2.5, so b misses at 4; b's rewrite at 5 then evicts a, which misses at 10.
    assert simulate(flipped, config(tiers=small_gpu)).hit_tokens["gpu"] == 0
    # Turn-aware evicts a (idle) instead, keeping b for its tool call: b hits at 4, a misses at 10.
    aware = simulate(flipped, config(tiers=small_gpu, retention=RETENTION["session"]))
    assert aware.hit_tokens["gpu"] == 100
    assert aware.prefill_seconds_saved == pytest.approx(100.0)


def test_evicted_kv_is_demoted_and_loaded_back_if_faster_than_recompute() -> None:
    tiers = (Tier("gpu", 100.0, math.inf), Tier("ram", 100.0, 50.0))  # 50 GB/s = 50 tokens/s here
    sessions = [
        TraceSession("a", (call(0, 100), call(10, 100))),
        TraceSession("b", (call(2, 100),)),  # its write at 3 demotes a to RAM
    ]
    # Loading 100 tokens takes 2 s; recomputing at 100 tokens/s takes 1 s, so the hit is not used.
    slow = simulate(sessions, config(tiers=tiers, prefill_tokens_per_second=100.0))
    assert slow.hit_tokens == {"gpu": 0, "ram": 0}
    # At 10 tokens/s recompute takes 10 s: load (2 s) wins and saves 8 s.
    fast = simulate(sessions, config(tiers=tiers, prefill_tokens_per_second=10.0))
    assert fast.hit_tokens == {"gpu": 0, "ram": 100}
    assert fast.load_seconds == pytest.approx(2.0)
    assert fast.prefill_seconds_saved == pytest.approx(8.0)


def test_routers_and_spill() -> None:
    session = [TraceSession("a", (call(0, 100), call(2, 100), call(4, 100)))]
    other = [TraceSession("z", (call(1.5, 10, completion=0),))]  # occupies a replica while a's 2nd call starts
    sessions = session + other
    # Least-loaded sends a's calls wherever is idle; z is busy on replica 0 at t=2, so a's 2nd call goes to 1.
    assert simulate(sessions, config(replicas=2, router="least-loaded")).hit_tokens["gpu"] == 100
    # Precise follows the KV: both later calls hit.
    assert simulate(sessions, config(replicas=2, router="precise")).hit_tokens["gpu"] == 200
    # With room for one call per replica, the busy preferred replica spills the call.
    spilled = simulate(sessions, config(replicas=2, router="precise", max_inflight=1))
    assert spilled.spills == 1
    assert spilled.hit_tokens["gpu"] == 100


def test_memory_time_is_tokens_held_over_time() -> None:
    sessions = [TraceSession("a", (call(0, 100, completion=20), call(10, 100)))]
    result = simulate(sessions, config(retention=Retention("t", 5.0)))
    # 120 tokens held from 1 (end of call 1) until expiry at 6: 600 token-seconds at 1 GB per token.
    # The second call's write at 11 is the end of the window, so it adds nothing.
    assert result.gb_hours["gpu"] == pytest.approx(600 / 3600)


def test_zero_duration_calls_are_handled() -> None:
    instant = TraceCall(0, 0, "A", 100, 0, 0)
    session = TraceSession("a", (instant, TraceCall(5, 6, "A", 120, 0, 0)))
    assert simulate([session], config()).hit_tokens["gpu"] == 100


def test_a_call_ending_before_it_starts_ends_at_its_start() -> None:
    session = TraceSession("a", (TraceCall(10, 4, "A", 100, 0, 0), TraceCall(12, 13, "A", 120, 0, 0)))
    assert simulate([session], config(retention=Retention("t", 3.0))).hit_tokens["gpu"] == 100


def test_turn_awareness_does_not_reorder_lower_tiers() -> None:
    # GPU and RAM each hold one 100-token context.
    tiers = (Tier("gpu", 100.0, math.inf), Tier("ram", 100.0, 1e12))
    sessions = [
        TraceSession("c", (call(0, 100),)),  # in-turn, written at 1; demoted to RAM by b's write at 3
        TraceSession("b", (call(2, 100),)),  # in-turn, written at 3
        TraceSession("a", (call(4, 100, turn_end=True), call(10, 100))),  # turn ends; written at 5, back at 10
    ]
    # At 5 the GPU demotes a (its turn ended) to RAM, which then holds c (written at 1) and a (at 5). RAM is LRU,
    # so it drops c, the older one, and a's return at 10 is served from RAM. Preferring to drop idle-turn KV in RAM
    # would drop a instead.
    aware = simulate(sessions, config(tiers=tiers, prefill_tokens_per_second=1e-3, retention=RETENTION["session"]))
    assert aware.hit_tokens == {"gpu": 0, "ram": 100}


def test_approximate_router_forgets_sessions_idle_past_sticky_idle() -> None:
    # a's replica 0 is busy with z when a returns after 10 s idle. Sticky: a waits in line on replica 0 (capacity
    # allows), and hits; with sticky_idle 5 the router forgets a and places it on idle replica 1, missing.
    sessions = [
        TraceSession("a", (call(0, 100), call(12, 100))),
        TraceSession("z", (TraceCall(11, 20, "A", 10, 0, 0),)),  # busy on replica 0 from 11 to 20
    ]
    sticky = simulate(sessions, config(replicas=2, router="approximate"))
    forgetful = simulate(sessions, config(replicas=2, router="approximate", sticky_idle=5.0))
    assert sticky.hit_tokens["gpu"] == 100
    assert forgetful.hit_tokens["gpu"] == 0


def test_overlapping_calls_replace_the_sessions_entry() -> None:
    # a's second call starts before its first ends; both write on the one replica. The second write replaces the
    # first entry: 100 tokens held 5..8 (lifetime 2 after the second write at 6), not 200 during 6..7.
    session = TraceSession(
        "a",
        (TraceCall(0, 5, "A", 100, 0, 0), TraceCall(3, 6, "A", 100, 0, 0), TraceCall(20, 21, "A", 100, 0, 0)),
    )
    result = simulate([session], config(retention=Retention("t", 2.0)))
    assert result.gb_hours["gpu"] == pytest.approx(300 / 3600)


def test_per_call_hits_are_recorded_on_request() -> None:
    sessions = [TraceSession("a", (call(0, 100), call(3, 150), call(400, 200)))]
    result = simulate(sessions, config(retention=Retention("t", 300.0)), record_calls=True)
    assert result.call_hits == {(0, 1): 100}  # the third call comes after the lifetime
    assert simulate(sessions, config()).call_hits == {}
