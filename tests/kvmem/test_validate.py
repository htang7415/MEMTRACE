from __future__ import annotations

import pytest

from memtrace.kvmem.validate import compare, sessions_from_run


def record(session: int, call: int, start: float, prompt: int, cached: int, ok: bool = True) -> dict[str, object]:
    return {
        "session": session,
        "call": call,
        "started_at": start,
        "e2e_seconds": 1.0,
        "prompt_tokens": prompt,
        "cached_prompt_tokens": cached,
        "output_tokens": 10,
        "ok": ok,
    }


def test_engine_and_sim_hits_are_compared_per_gap_bin() -> None:
    records = [
        record(0, 1, 3.0, 200, 96),  # gap 2 s: 96 of 100 reusable cached (whole blocks)
        record(0, 0, 0.0, 100, 0),
        record(0, 2, 100.0, 300, 0),  # gap 96 s: engine evicted it
        record(1, 0, 0.0, 50, 0, ok=False),  # failed calls are left out
    ]
    sessions, cached = sessions_from_run(records)
    assert [len(s.calls) for s in sessions] == [3]
    # Unlimited capacity: the simulator keeps everything; hits rounded down to 16-token blocks.
    result = compare(sessions, cached, kv_tokens=10**9)
    assert result["by_gap"]["<10s"] == {
        "calls": 1,
        "engine_hit_share": pytest.approx(0.96),
        "sim_hit_share": pytest.approx(96 / 100),
        "diff_points": pytest.approx(0.0),
    }
    assert result["by_gap"]["1-5min"]["sim_hit_share"] == pytest.approx(192 / 200)
    assert result["by_gap"]["1-5min"]["engine_hit_share"] == 0.0
    assert result["within_tolerance"] is False
    # With 100 tokens of capacity no context (110 or 210 tokens) fits, so the simulator predicts no hits.
    assert compare(sessions, cached, kv_tokens=100)["by_gap"]["1-5min"]["sim_hit_share"] == 0.0


def test_provenance_names_the_commit() -> None:
    from memtrace.kvmem.provenance import provenance

    made_by = provenance()
    assert set(made_by) == {"git_commit", "git_dirty", "generated_utc"}
