from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import Any

import pytest

from memtrace.kv.gateway_replay import ChatCall
from memtrace.kv.resume import (
    SPEC_SCHEMA,
    CallRecord,
    ResumeSession,
    arm_params,
    common_prefix,
    load_spec,
    parse_session,
    replay,
    resume_metrics,
)


def _call(end: str, dur_ms: float, n_msgs: int) -> dict[str, Any]:
    return {
        "timestamp": f"2026-06-01T{end}.000000000Z",
        "duration_ms": dur_ms,
        "tokens": {"prompt": 100 * n_msgs, "completion": 20},
        "message_metadata": [{"sequenceId": i, "type": "Msg", "role": "user", "token_len": 100} for i in range(n_msgs)],
    }


def test_parse_session_records_each_calls_pause_after_the_previous_one_ends() -> None:
    # Calls end at 10:00:10, 10:00:20 and 10:05:20 and last 2 s, 3 s and 1 s.
    row = {
        "session_id": "s",
        "turns": [
            {"llm_calls": [_call("10:00:10", 2_000, 1), _call("10:00:20", 3_000, 2), _call("10:05:20", 1_000, 3)]}
        ],
    }
    s = parse_session(row)
    assert s.gaps == (None, 7.0, 299.0)  # 10:00:17 - 10:00:10, then 10:05:19 - 10:00:20
    assert [c.t for c in s.calls] == [0.0, 9.0, 311.0]
    assert [len(c.messages) for c in s.calls] == [1, 2, 3]


def test_common_prefix() -> None:
    assert common_prefix([1, 2, 3, 4], [1, 2, 9]) == 2
    assert common_prefix([1, 2], [1, 2, 3]) == 2
    assert common_prefix([], [1]) == 0


def _session(sid: str, gaps: list[float | None]) -> ResumeSession:
    calls = tuple(ChatCall(t=float(i), dur=0.0, messages=(), out_tokens=1) for i in range(len(gaps)))
    return ResumeSession(id=sid, calls=calls, gaps=tuple(gaps))


def test_replay_is_closed_loop_per_session_and_never_repeats_a_session() -> None:
    sessions = [_session(f"s{i}", [None, 0.2, 0.2]) for i in range(4)]
    in_flight: dict[str, int] = {}
    lock = threading.Lock()
    seen_prev: list[tuple[str, int, list[int] | None]] = []

    def send(s: ResumeSession, k: int, prev: list[int] | None) -> tuple[CallRecord, list[int] | None]:
        with lock:
            assert in_flight.get(s.id, 0) == 0, "a session's calls must not overlap"
            in_flight[s.id] = 1
            seen_prev.append((s.id, k, prev))
        time.sleep(0.05)
        with lock:
            in_flight[s.id] = 0
        return CallRecord(session=s.id, call=k, gap_s=None, sent_s=0.0, status="ok"), [k]

    records = replay(
        sessions, send=send, concurrency=2, horizon_s=5.0, stagger_s=0.0, idle_cap_s=60.0, time_scale=1.0, seed=3
    )
    by_session: dict[str, list[CallRecord]] = {}
    for r in records:
        by_session.setdefault(r.session, []).append(r)
    assert set(by_session) == {"s0", "s1", "s2", "s3"}  # every session ran once
    for calls in by_session.values():
        ks = [r.call for r in calls]
        assert ks == sorted(ks) and len(set(ks)) == len(ks)
        assert calls[0].gap_s is None  # a session's first replayed call has no pause (and no reusable prefix)
        assert all(r.gap_s == pytest.approx(0.2) for r in calls[1:])
    for sid, k, prev in seen_prev:
        first = min(r.call for r in by_session[sid])
        assert prev == (None if k == first else [k - 1])


def test_replay_rejects_more_slots_than_sessions() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        replay([_session("s", [None, 1.0])], send=lambda *a: (None, None), concurrency=2, horizon_s=1.0,  # type: ignore[arg-type,return-value]
               stagger_s=0.0, idle_cap_s=1.0, time_scale=1.0, seed=0)  # fmt: skip


def _rec(gap: float | None, reusable: int, cached: int, ttft: float = 0.1, status: str = "ok") -> CallRecord:
    return CallRecord(session="s", call=1, gap_s=gap, sent_s=100.0, status=status, ttft_s=ttft if status == "ok" else None,
                      cached_tokens=cached, reusable_tokens=reusable)  # fmt: skip


def test_resume_metrics() -> None:
    records = [
        _rec(None, 0, 0),  # first call: no reusable prefix
        _rec(2.0, 1000, 1000),  # quick tool call: fully served
        _rec(120.0, 1000, 0, ttft=2.0),  # resumed after 2 min: recomputed
        _rec(900.0, 2000, 1000, ttft=1.0),  # resumed after 15 min: half served
        _rec(400.0, 500, 0, status="error"),
        CallRecord(session="s", call=0, gap_s=3.0, sent_s=1.0, status="ok", reusable_tokens=999),  # warmup
    ]
    m = resume_metrics(records, warmup_s=10.0, resume_gap_s=60.0, failed_ttft_s=30.0)
    assert m["calls"] == 5 and m["errors"] == 1 and m["resumed_calls"] == 2
    assert m["resumed_recompute_share"] == pytest.approx(2000 / 3000)
    assert m["all_recompute_share"] == pytest.approx(2000 / 4000)
    assert m["resumed_recomputed_tokens_per_call"] == pytest.approx(1000)
    assert m["gap_lt10s_served_share"] == 1.0 and m["gap_1_5min_served_share"] == 0.0
    assert m["gap_10_60min_served_share"] == pytest.approx(0.5)
    assert m["resumed_ttft_p50_ms"] == pytest.approx(1500)
    assert m["resumed_ttft_all_p95_ms"] > 20_000  # the failed resumed call waited the 30 s timeout


def test_committed_specs_load() -> None:
    specs = [p for p in Path("experiments").glob("r*_resume_*.yaml") if not p.name.startswith("._")]
    assert specs
    for path in specs:
        spec = load_spec(path)
        assert spec["schema_version"] == SPEC_SCHEMA
        assert spec["replay"]["concurrency"] <= spec["sessions"]


def test_arm_params_turn_a_kv_tier_into_vllm_metal_offload_flags(tmp_path: Path) -> None:
    target = {"model": "m", "extra_args": ["--block-size", "16"]}
    assert arm_params(target, None, tmp_path) == target
    assert arm_params(target, {"target": {"model": "n"}}, tmp_path)["model"] == "n"
    params = arm_params(target, {"kv_tier": {"host_pool_gib": 0.5, "max_size_gib": 20}}, tmp_path / "store")
    args = params["extra_args"]
    assert args[:2] == ["--block-size", "16"] and args[2:4] == ["--kv-offloading-size", "0.5"]
    config = json.loads(args[args.index("--kv-transfer-config") + 1])
    tier = config["kv_connector_extra_config"]["secondary_tiers"][0]
    assert tier == {"type": "fs", "root_dir": str(tmp_path / "store"), "max_size_gib": 20}
    assert target["extra_args"] == ["--block-size", "16"]  # the shared target is not mutated
    pool_only = arm_params(target, {"kv_tier": {"host_pool_gib": 1}}, tmp_path)["extra_args"]
    assert pool_only == ["--block-size", "16", "--kv-offloading-size", "1"]
