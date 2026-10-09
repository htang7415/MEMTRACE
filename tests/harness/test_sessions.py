"""Agent replays: sessions issue their calls in order with recorded gaps."""

import gzip
import json

from memtrace.harness import workloads
from memtrace.harness.loadgen import RequestRecord, RequestSpec, SessionSpec, run_sessions, summarize
from memtrace.harness.pickers import PICKERS
from memtrace.kv.validate import sessions_from_run
from memtrace.serving.client import CompletionResult


def _spec(rid: str, session: str) -> RequestSpec:
    return RequestSpec(rid, session, session, ({"role": "user", "content": rid},))


def test_run_sessions_keeps_calls_ordered_within_a_session() -> None:
    sent = []

    def send(url, messages, **kw):
        sent.append(messages[0]["content"])
        return CompletionResult(
            status="ok",
            text="ok",
            ttft_s=0.01,
            e2e_s=0.02,
            prompt_tokens=5,
            completion_tokens=2,
            cached_tokens=1,
            error=None,
        )

    sessions = [
        SessionSpec("s0", 0.0, ((_spec("s0-c0", "s0"), 0.0), (_spec("s0-c1", "s0"), 0.5))),
        SessionSpec("s1", 0.0, ((_spec("s1-c0", "s1"), 0.0),)),
    ]
    slept = []
    records, _, peak = run_sessions(
        sessions,
        base_urls=["http://x"],
        picker=PICKERS["round_robin"](1),
        timeout_s=5,
        max_tokens=8,
        send=send,
        sleep=slept.append,
    )
    assert [r.request_id for r in records] == ["s0-c0", "s0-c1", "s1-c0"] and all(r.status == "ok" for r in records)
    assert sent.index("s0-c0") < sent.index("s0-c1")
    assert 0.5 in slept and 1 <= peak <= 2


def test_copilot_agent_workload_replays_sessions(tmp_path, monkeypatch) -> None:
    def call(end, prompt, cached):
        tokens = {"prompt": prompt, "cached": cached, "completion": 40}
        return {"timestamp": f"2026-06-06T00:00:{end:02d}.000000000Z", "duration_ms": 500, "tokens": tokens}

    path = tmp_path / "shard.jsonl.gz"
    with gzip.open(path, "wt") as fh:
        fh.write(json.dumps({"session_id": "a", "turns": [{"llm_calls": [call(1, 400, 0), call(3, 600, 400)]}]}) + "\n")
    monkeypatch.setattr("memtrace.datasets.loaders.copilot.archive_path", lambda day, root=None: path)
    w = workloads.make_workload("copilot_agent", {"sessions": 4, "token_scale": 0.1, "reuse": "full"}, seed=0)
    assert w.sessions is not None and len(w.sessions) == 1
    (session,) = w.sessions
    assert [spec.request_id for spec, _ in session.calls] == ["s0-c0", "s0-c1"]
    first, second = (spec.messages[0]["content"].split(" ") for spec, _ in session.calls)
    assert second[: len(first)] == first  # append-only
    assert len(w.specs) == 2


def test_tpot_slo_counts_toward_attainment() -> None:
    def rec(e2e, tokens):
        return RequestRecord("r", "s", 0, 0.0, "ok", 0.1, e2e, 10, 0, tokens, False, None)

    records = [rec(1.1, 11), rec(3.1, 11)]  # 0.1 s and 0.3 s per output token
    assert summarize(records, duration_s=1, ttft_slo_s=2, e2e_slo_s=60)["slo_attainment"] == 1.0
    assert summarize(records, duration_s=1, ttft_slo_s=2, e2e_slo_s=60, tpot_slo_s=0.15)["slo_attainment"] == 0.5


def test_m3_reads_harness_request_records() -> None:
    def row(rid, start, prompt, cached):
        return {
            "request_id": rid,
            "session_id": rid.split("-")[0],
            "scheduled_s": start,
            "status": "ok",
            "e2e_s": 1.0,
            "prompt_tokens": prompt,
            "cached_tokens": cached,
            "completion_tokens": 10,
        }

    sessions, cached = sessions_from_run([row("s0-c1", 3.0, 200, 96), row("s0-c0", 0.0, 100, 0)])
    assert [c.prompt for c in sessions[0].calls] == [100, 200] and cached[(0, 1)] == 96
