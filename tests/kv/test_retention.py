from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from memtrace.kv.retention import analyze, gap_bin, later_calls, working_set
from memtrace.kv.traces import TraceCall, TraceSession, read_sessions, trace_time


def call(start: float, prompt: int, cached: int, *, model: str = "A", duration: float = 1.0) -> TraceCall:
    return TraceCall(start, start + duration, model, prompt, cached, 10)


def test_trace_time_reads_nanosecond_utc_stamps() -> None:
    assert trace_time("1970-01-01T00:00:01.500000000Z") == pytest.approx(1.5)
    assert trace_time("1970-01-01T00:01:00Z") == 60.0


def test_read_sessions_orders_calls_and_skips_missing_counts(tmp_path: Path) -> None:
    def raw(stamp: str, prompt: object) -> dict[str, object]:
        tokens = {"prompt": prompt, "cached": 0, "completion": 5}
        return {"timestamp": stamp, "duration_ms": 500.0, "model": "Model A", "tokens": tokens}

    record = {
        "session_id": "s1",
        "turns": [
            {"llm_calls": [raw("1970-01-01T00:00:10.000Z", 200), raw("1970-01-01T00:00:12.000Z", None)]},
            {"llm_calls": [raw("1970-01-01T00:00:05.000Z", 100)]},
        ],
    }
    path = tmp_path / "shard-0000.jsonl.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(json.dumps(record) + "\n")
    (session,) = read_sessions([path])
    assert session.session_id == "s1"
    assert [c.prompt for c in session.calls] == [100, 200]
    assert session.calls[0].start == pytest.approx(4.5)
    assert session.calls[0].end == pytest.approx(5.0)


def test_gap_bins_are_half_open() -> None:
    assert gap_bin(9.99) == "<10s"
    assert gap_bin(10.0) == "10-60s"
    assert gap_bin(299.0) == "1-5min"
    assert gap_bin(300.0) == "5-10min"
    assert gap_bin(3600.0) == ">1h"


def test_each_miss_is_attributed_to_one_cause_in_precedence_order() -> None:
    # Each later call's previous call ends at t + 1; gaps are measured from there.
    calls = [
        call(0, 1000, 0),
        call(3, 1100, 1000),  # gap 2 s, full hit: no waste
        call(6, 1200, 200, model="B"),  # different model: model_switch, waste 1100 - 200
        call(9, 1300, 1000, model="B"),  # gap 2 s, grew: short_gap_grew, waste 1200 - 1000
        call(11, 800, 0, model="B"),  # shrank 38%: compaction, waste 800
        call(612, 900, 0, model="B"),  # gap 600 s: expiry, waste 800
        call(643, 950, 0, model="B"),  # gap 30 s: mid_gap, waste 900
        call(646, 940, 40, model="B"),  # gap 2 s, shrank 1%: short_gap_shrank, waste 900
        call(649, 860, 0, model="B"),  # shrank 8.5%, under the threshold: short_gap_shrank, waste 860
        call(652, 774, 0, model="B"),  # shrank exactly 10%: compaction, waste 774
    ]
    items = list(later_calls([TraceSession("s", tuple(calls))]))
    assert [(i.cause, i.waste) for i in items if i.waste] == [
        ("model_switch", 900),
        ("short_gap_grew", 200),
        ("compaction", 800),
        ("expiry", 800),
        ("mid_gap", 900),
        ("short_gap_shrank", 900),
        ("short_gap_shrank", 860),
        ("compaction", 774),
    ]


def test_analyze_totals_and_shares() -> None:
    sessions = [
        TraceSession("a", (call(0, 1000, 0), call(3, 1000, 0))),  # short gap, near-total miss: waste 1000
        TraceSession("b", (call(0, 1000, 0), call(401, 1000, 1000))),  # gap 400 s, full hit: no waste
        TraceSession("c", (call(0, 500, 0),)),  # no later call
    ]
    summary = analyze(sessions)
    assert summary["sessions"] == 3
    assert summary["sessions_with_later_calls"] == 2
    assert summary["later_calls"] == 2
    assert summary["waste_tokens"] == 1000
    assert summary["waste_share_of_later_prompt"] == pytest.approx(0.5)
    by_cause = summary["waste_by_cause"]
    assert isinstance(by_cause, dict)
    assert by_cause["short_gap_grew"] == {"calls": 1, "tokens": 1000, "share": 1.0}
    assert summary["short_gap_grew_near_total_miss_share_of_waste"] == pytest.approx(1.0)
    hits = summary["hit_by_gap"]
    assert isinstance(hits, dict)
    assert hits["<10s"]["hit_ratio_tokens"] == 0.0
    assert hits["5-10min"]["hit_ratio_tokens"] == 1.0
    assert hits[">1h"]["hit_ratio_tokens"] is None
    assert summary["model_switch_calls"] == 0


def test_retained_needs_same_model_no_compaction_and_a_gap_within_the_lifetime() -> None:
    (item,) = later_calls([TraceSession("s", (call(0, 1000, 0), call(61, 1200, 0)))])  # gap 60 s
    assert item.retained(61) == 1000
    assert item.retained(60) == 0
    (switched,) = later_calls([TraceSession("s", (call(0, 1000, 0), call(3, 1200, 0, model="B")))])
    assert switched.retained(300) == 0
    (compacted,) = later_calls([TraceSession("s", (call(0, 1000, 0), call(3, 900, 0)))])
    assert compacted.retained(300) == 0


def test_working_set_holds_each_call_until_the_next_call_or_the_lifetime() -> None:
    # Prompt + completion: 110 then 210. Window is 0..6 s (first start to last end).
    session = TraceSession("s", (call(0, 100, 0), call(5, 200, 0)))
    # lifetime 2: 110 held 0..3 (call end 1 + 2), 210 held 5..6 (clipped at the window end).
    assert working_set([session], 2) == (pytest.approx((110 * 3 + 210 * 1) / 6), 210)
    # lifetime 100: 110 held until the next call starts at 5.
    assert working_set([session], 100) == (pytest.approx((110 * 5 + 210 * 1) / 6), 210)
    # Two overlapping sessions add up.
    other = TraceSession("t", (call(0, 300, 0, duration=6),))
    assert working_set([session, other], 2)[1] == 210 + 310  # at t = 5


def test_working_set_is_measured_over_the_sessions_day() -> None:
    day = trace_time("2026-06-06T00:00:00Z")
    # One call held 10 s at noon of its day, plus an old call from the previous day that must not count.
    old = call(day - 7200, 999, 0)
    session = TraceSession("s", (old, call(day + 43200, 90, 0, duration=10)), "2026-06-06")
    mean, peak = working_set([session], 0)
    assert peak == 100
    assert mean == pytest.approx(100 * 10 / 86400)


def test_peak_counts_caches_written_before_the_window() -> None:
    day = trace_time("2026-06-06T00:00:00Z")
    # Written at 23:59:50 the day before, held until it expires 20 s later, inside the day.
    session = TraceSession("s", (call(day - 11, 90, 0, duration=1),), "2026-06-06")
    mean, peak = working_set([session], 20)
    assert peak == 100
    assert mean == pytest.approx(100 * 10 / 86400)


def test_analyze_handles_too_few_later_calls() -> None:
    assert analyze([TraceSession("a", (call(0, 100, 0),))])["gap_seconds"] is None
    one = analyze([TraceSession("a", (call(0, 100, 0), call(3, 120, 100)))])
    assert one["gap_seconds"] == {"median": 2.0, "p90": 2.0, "p99": 2.0}


def test_first_call_cached_share_shows_prefixes_shared_across_sessions() -> None:
    sessions = [TraceSession("a", (call(0, 100, 40), call(3, 120, 100))), TraceSession("b", (call(0, 300, 0),))]
    assert analyze(sessions)["first_call_cached_share"] == pytest.approx(40 / 400)


def test_lifetime_shares_use_the_simulator_denominator() -> None:
    # Call 2 is a model switch: its overlap counts as waste but not as reusable for lifetimes.
    sessions = [TraceSession("a", (call(0, 100, 0), call(3, 120, 100), call(6, 150, 0, model="B")))]
    lifetime = analyze(sessions)["lifetimes"]["5min"]
    assert lifetime["retained_share_of_reusable"] == 1.0
    assert lifetime["observed_cached_share_of_reusable"] == 1.0
