import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from memtrace.commands import run_engine_bench
from memtrace.serving.metrics import summarize_requests
from memtrace.serving import workloads


def test_mooncake_shared_hash_ids_render_shared_prefix(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps({"timestamp": 0, "input_length": 1024, "output_length": 512, "hash_ids": [0, 1]})
        + "\n"
        + json.dumps({"timestamp": 1, "input_length": 1024, "output_length": 5000, "hash_ids": [0, 2]})
        + "\n"
    )

    first, second = workloads.mooncake(trace, num_requests=5, max_tokens=64, block_tokens=8)

    first_blocks, second_blocks = first.prompt.split(" "), second.prompt.split(" ")
    assert len(first_blocks) == len(second_blocks) == 16
    assert first_blocks[:8] == second_blocks[:8]
    assert first_blocks[8:] != second_blocks[8:]
    # output length scales with the block size (512 * 8 / 512 = 8) and is capped by max_tokens
    assert first.max_tokens == 8
    assert second.max_tokens == 64


def test_sharegpt_samples_first_human_turn_deterministically(tmp_path) -> None:
    data = [{"conversations": [{"from": "human", "value": f"q{i}"}, {"from": "gpt", "value": "a"}]} for i in range(10)]
    data.append({"conversations": [{"from": "gpt", "value": "skip"}, {"from": "human", "value": "x"}]})
    data.append({"conversations": [{"from": "human", "value": "only one turn"}]})
    path = tmp_path / "sharegpt.json"
    path.write_text(json.dumps(data))

    sample = workloads.sharegpt(path, num_requests=4, max_tokens=32, seed=1)

    assert sample == workloads.sharegpt(path, num_requests=4, max_tokens=32, seed=1)
    assert all(request.prompt.startswith("q") and request.max_tokens == 32 for request in sample)


def test_summarize_requests_counts_goodput_under_slo() -> None:
    records = [
        {"ok": True, "ttft_seconds": 0.5, "tpot_seconds": 0.05, "e2e_seconds": 1.0, "output_tokens": 10},
        {"ok": True, "ttft_seconds": 3.0, "tpot_seconds": 0.05, "e2e_seconds": 4.0, "output_tokens": 10},
        {"ok": True, "ttft_seconds": 0.5, "tpot_seconds": 0.20, "e2e_seconds": 2.0, "output_tokens": 10},
        {"ok": True, "ttft_seconds": 0.5, "tpot_seconds": None, "e2e_seconds": 0.5, "output_tokens": 1},
        {"ok": False, "error": "boom"},
    ]

    summary = summarize_requests(records, wall_seconds=2.0, concurrency=2, slo_ttft_seconds=1.0, slo_tpot_seconds=0.1)

    assert summary["errors"] == 1
    assert summary["output_tokens_per_second"] == 15.5
    assert summary["slo_attainment"] == pytest.approx(2 / 5)
    assert summary["goodput_requests_per_second"] == 1.0
    assert summary["latency"]["ttft_seconds"]["p50"] == 0.5


class _FakeEngine(BaseHTTPRequestHandler):
    hits = 0.0
    queries = 0.0
    resets = 0

    def do_GET(self) -> None:
        if self.path == "/version":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"version": "0.31.0"}')
            return
        body = (
            f'vllm:prefix_cache_queries_total{{engine="0"}} {_FakeEngine.queries}\n'
            f'vllm:prefix_cache_hits_total{{engine="0"}} {_FakeEngine.hits}\n'
        ).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path == "/reset_prefix_cache":
            _FakeEngine.resets += 1
            self.send_response(200)
            self.end_headers()
            return
        self.rfile.read(int(self.headers["Content-Length"]))
        _FakeEngine.queries += 100
        _FakeEngine.hits += 25
        events = [
            {"choices": [{"delta": {"content": "hi"}}]},
            {"choices": [{"delta": {"content": " there"}}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 100, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 25}},
            },
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args) -> None:
        pass


def test_engine_bench_end_to_end_against_fake_engine(tmp_path, monkeypatch) -> None:
    trace_dir = tmp_path / "public" / "mooncake"
    trace_dir.mkdir(parents=True)
    (trace_dir / "toolagent_trace.jsonl").write_text(
        "".join(
            json.dumps({"timestamp": i, "input_length": 512, "output_length": 512, "hash_ids": [0, i]}) + "\n"
            for i in range(6)
        )
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeEngine)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(run_engine_bench, "PowerSampler", _NoPower)
    try:
        run_engine_bench.main(
            [
                "--engine",
                "fake",
                "--base-url",
                f"http://127.0.0.1:{server.server_port}/v1",
                "--model",
                "m",
                "--workload",
                "mooncake-toolagent",
                "--num-requests",
                "4",
                "--concurrency",
                "1,2",
                "--data-dir",
                str(tmp_path / "public"),
                "--out-dir",
                str(tmp_path / "out"),
                "--idle-seconds",
                "0",
            ]
        )
    finally:
        server.shutdown()

    run_dir = tmp_path / "out" / "fake" / "mooncake-toolagent"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["num_requests"] == 4
    assert manifest["engine_version"] == "0.31.0"
    summary = json.loads((run_dir / "c2" / "summary.json").read_text())
    assert summary["requests"] == 4 and summary["errors"] == 0
    assert summary["output_tokens"] == 8
    assert summary["prefix_cache"] == {"reset_before_level": True, "engine_hit_rate": 0.25, "usage_hit_rate": 0.25}
    assert len((run_dir / "c1" / "requests.jsonl").read_text().splitlines()) == 4
    assert _FakeEngine.resets == 2


class _NoPower:
    samples: list = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def summary(self):
        return None


def test_engine_report_renders_one_row_per_level(tmp_path) -> None:
    from memtrace.serving.report import engine_rows, render

    for engine, level in (("b-engine", 2), ("a-engine", 1), ("a-engine", 8)):
        level_dir = tmp_path / engine / "sharegpt" / f"c{level}"
        level_dir.mkdir(parents=True)
        summary = summarize_requests(
            [{"ok": True, "ttft_seconds": 0.2, "tpot_seconds": 0.01, "e2e_seconds": 1.0, "output_tokens": 10}],
            wall_seconds=1.0,
            concurrency=level,
            slo_ttft_seconds=1.0,
            slo_tpot_seconds=0.1,
        )
        (level_dir / "summary.json").write_text(json.dumps(summary))

    rows = engine_rows(tmp_path)

    assert [(row[0], row[2]) for row in rows] == [("a-engine", "1"), ("a-engine", "8"), ("b-engine", "2")]
    assert rows[0][4] == "0.200 / 0.200" and rows[0][6] == "100%" and rows[0][8] == "n/a"
    assert render(rows).count("\n") == 5


def test_engine_parity_compares_outputs_by_index(tmp_path) -> None:
    from memtrace.serving.parity import parity

    ref, cand = tmp_path / "ref.jsonl", tmp_path / "cand.jsonl"
    ref.write_text(
        "".join(
            json.dumps(r) + "\n"
            for r in [{"ok": True, "index": 0, "output_text": "abcd"}, {"ok": True, "index": 1, "output_text": "same"}]
        )
    )
    cand.write_text(
        "".join(
            json.dumps(r) + "\n"
            for r in [{"ok": True, "index": 1, "output_text": "same"}, {"ok": True, "index": 0, "output_text": "abXY"}]
        )
    )

    result = parity(ref, cand)

    assert result["compared"] == 2
    assert result["exact_match_rate"] == 0.5
    assert result["first_divergence_chars"] == 2
    assert result["mean_similarity"] == pytest.approx((0.5 + 1.0) / 2)


def test_mooncake_truncates_long_prompts_to_leading_blocks(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps({"timestamp": 0, "input_length": 0, "output_length": 64, "hash_ids": [0, 1, 2, 3]}) + "\n"
    )

    (full,) = workloads.mooncake(trace, num_requests=1, max_tokens=64, block_tokens=4)
    (cut,) = workloads.mooncake(trace, num_requests=1, max_tokens=64, block_tokens=4, max_blocks=2)

    assert cut.truncated_blocks == 2 and full.truncated_blocks == 0
    assert full.prompt.startswith(cut.prompt) and len(cut.prompt.split(" ")) == 8


def test_engine_parity_normalizes_inline_and_separate_thinking(tmp_path) -> None:
    from memtrace.serving.parity import parity

    inline, separate = tmp_path / "inline.jsonl", tmp_path / "separate.jsonl"
    inline.write_text(
        json.dumps({"index": 0, "ok": True, "output_text": "<think>\nplan it\n</think>\n\nanswer"}) + "\n"
    )
    separate.write_text(
        json.dumps({"index": 0, "ok": True, "reasoning_text": "plan it\n", "output_text": "answer"}) + "\n"
    )

    assert parity(inline, separate)["exact_match_rate"] == 1.0


def test_power_summary_drops_implausible_samples() -> None:
    from memtrace.serving.bench import PowerSampler

    sampler = PowerSampler(max_plausible_w=100.0)
    base = {"cpu_w": 0.0, "gpu_w": 5.0, "ane_w": 0.0, "ram_bytes": 1e9, "swap_bytes": 0}
    sampler.samples = [{**base, "sys_w": w} for w in (20.0, 30.0, 40.0, 5000.0)]

    summary = sampler.summary()

    assert summary["samples"] == 3 and summary["dropped_implausible_samples"] == 1
    assert summary["mean_sys_w"] == 30.0 and summary["max_sys_w"] == 40.0


def test_pool_delta_reports_per_pod_requests_hits_and_imbalance() -> None:
    from memtrace.serving.bench import pool_delta

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
    from memtrace.serving.bench import _sum_counters

    text = (
        "# HELP vllm:prefix_cache_hits_total hits\n"
        'vllm:prefix_cache_hits_total{engine="0"} 5\n'
        'vllm:prefix_cache_hits{model_name="m"} 7\n'
        "vllm:prefix_cache_queries 20\n"
        'vllm:prefix_cache_hits_created{engine="0"} 1.7e9\n'
    )

    totals = _sum_counters(text, ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"))

    assert totals == {"vllm:prefix_cache_hits_total": 12.0, "vllm:prefix_cache_queries_total": 20.0}


def test_agent_sessions_resend_growing_history_with_unique_prefixes() -> None:
    requests = workloads.agent_sessions(num_sessions=3, turns=2, max_tokens=16, prefix_words=20, turn_words=5)

    assert requests == workloads.agent_sessions(num_sessions=3, turns=2, max_tokens=16, prefix_words=20, turn_words=5)
    assert len(requests) == 6 and all(r.max_tokens == 16 for r in requests)
    first_turn, second_turn = requests[:3], requests[3:]
    assert all(len(r.prompt.split(" ")) == 25 for r in first_turn)
    assert all(len(r.prompt.split(" ")) == 30 for r in second_turn)
    # every second-turn prompt extends exactly one first-turn prompt (same session prefix + history)
    for r in second_turn:
        assert sum(r.prompt.startswith(f.prompt) for f in first_turn) == 1
    assert len({r.prompt.split(" ")[:20].__str__() for r in first_turn}) == 3


def test_pool_delta_reports_hosted_spend_only_for_pods_that_spend() -> None:
    from memtrace.serving.bench import pool_delta

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


def _copilot_trace(tmp_path):
    import gzip

    def call(end, duration_ms, prompt, cached, completion):
        return {
            "message_id": f"m{end}",
            "timestamp": f"2026-06-06T00:00:{end:02d}.000000000Z",
            "duration_ms": duration_ms,
            "tokens": {"prompt": prompt, "cached": cached, "completion": completion},
        }

    sessions = [
        # ends at t=2 (started t=0), then a 3 s gap, ends at t=7 (started t=5)
        {"session_id": "a", "turns": [{"llm_calls": [call(2, 2000, 400, 0, 80), call(7, 2000, 800, 400, 40)]}]},
        {"session_id": "b", "turns": [{"llm_calls": [call(12, 1000, 200, 0, 8), call(14, 1000, 240, 200, 8)]}]},
        {"session_id": "c", "turns": [{"llm_calls": [call(20, 1000, 100, 0, 4)]}]},  # single call: skipped
        {"session_id": "d", "turns": [{"llm_calls": [call(30, 1000, 100, None, 4), call(31, 500, 120, 100, 4)]}]},
    ]
    path = tmp_path / "shard.jsonl.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("".join(json.dumps(s) + "\n" for s in sessions))
    return path


def test_copilot_sessions_reproduce_cache_structure_and_timing(tmp_path) -> None:
    sessions = workloads.copilot_sessions(
        [_copilot_trace(tmp_path)], num_sessions=10, token_scale=0.1, gap_scale=0.5, window_seconds=10
    )

    assert [s.session_id for s in sessions] == ["a", "b"]  # c: one call; d: one call left after dropping null tokens
    a = sessions[0]
    first, second = a.calls
    assert (first.prompt_target, first.cached_target) == (40, 0)
    assert (second.prompt_target, second.cached_target) == (80, 40)
    assert second.prompt.split(" ")[:40] == first.prompt.split(" ")  # cached part is the previous prompt
    assert second.prompt.split(" ")[40:] != first.prompt.split(" ")
    assert first.gap_before == 0.0 and second.gap_before == pytest.approx(1.5)  # (5 - 2) s x 0.5
    assert (first.max_tokens, second.max_tokens) == (8, 4)  # 80 x 0.1; 40 x 0.1 floored at 4
    assert a.start_offset == 0.0 and sessions[1].start_offset == pytest.approx(10.0)  # starts 0 s and 11 s -> window


def test_run_sessions_keeps_calls_ordered_within_a_session() -> None:
    from memtrace.serving.bench import run_sessions
    from memtrace.serving.workloads import AgentCall, AgentSession

    class _Model:
        def __init__(self):
            self.prompts = []
            self.last_call = {"ttft_seconds": 0.1, "tpot_seconds": 0.01, "e2e_seconds": 0.2, "output_tokens": 2}

        def generate(self, prompt, max_tokens=None):
            self.prompts.append(prompt)
            return "ok"

    sessions = [
        AgentSession("s0", 0.0, (AgentCall("a1", 4, 0.0, 2, 0), AgentCall("a2", 4, 0.0, 2, 1))),
        AgentSession("s1", 0.0, (AgentCall("b1", 4, 0.0, 2, 0),)),
    ]
    model = _Model()
    records, _, peak = run_sessions(model, sessions, sleep=lambda _: None)

    assert len(records) == 3 and all(r["ok"] for r in records)
    assert model.prompts.index("a1") < model.prompts.index("a2")
    assert sorted(r["index"] for r in records) == [0, 1, 2]
    assert {(r["session"], r["call"]) for r in records} == {(0, 0), (0, 1), (1, 0)}
    assert 1 <= peak <= 2


def test_copilot_gaps_are_capped(tmp_path) -> None:
    sessions = workloads.copilot_sessions(
        [_copilot_trace(tmp_path)], num_sessions=10, token_scale=0.1, gap_scale=1.0, max_gap_seconds=2.0
    )

    assert sessions[0].calls[1].gap_before == 2.0  # 3 s gap at scale 1.0, capped at 2 s
