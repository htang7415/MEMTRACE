"""Prompt generators for the serving workloads (public traces rendered to text with their prefix structure)."""

import json

import pytest

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


def test_mooncake_truncates_long_prompts_to_leading_blocks(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps({"timestamp": 0, "input_length": 0, "output_length": 64, "hash_ids": [0, 1, 2, 3]}) + "\n"
    )

    (full,) = workloads.mooncake(trace, num_requests=1, max_tokens=64, block_tokens=4)
    (cut,) = workloads.mooncake(trace, num_requests=1, max_tokens=64, block_tokens=4, max_blocks=2)

    assert cut.truncated_blocks == 2 and full.truncated_blocks == 0
    assert full.prompt.startswith(cut.prompt) and len(cut.prompt.split(" ")) == 8


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


def test_copilot_gaps_are_capped(tmp_path) -> None:
    sessions = workloads.copilot_sessions(
        [_copilot_trace(tmp_path)], num_sessions=10, token_scale=0.1, gap_scale=1.0, max_gap_seconds=2.0
    )

    assert sessions[0].calls[1].gap_before == 2.0  # 3 s gap at scale 1.0, capped at 2 s


def test_copilot_full_reuse_extends_the_previous_prompt(tmp_path) -> None:
    import gzip

    def call(end, prompt, cached):
        tokens = {"prompt": prompt, "cached": cached, "completion": 40}
        return {"timestamp": f"2026-06-06T00:00:{end:02d}.000000000Z", "duration_ms": 500, "tokens": tokens}

    # The provider cached only 100 of 400 reusable tokens on call 2; call 3 is a compaction (600 -> 300).
    record = {"session_id": "e", "turns": [{"llm_calls": [call(1, 400, 0), call(3, 600, 100), call(5, 300, 300)]}]}
    path = tmp_path / "e.jsonl.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(json.dumps(record) + "\n")

    observed = workloads.copilot_sessions([path], num_sessions=1, token_scale=0.1)[0].calls
    full = workloads.copilot_sessions([path], num_sessions=1, token_scale=0.1, reuse="full")[0].calls
    assert [c.cached_target for c in observed] == [0, 10, 30]
    assert [c.cached_target for c in full] == [0, 40, 0]
    assert full[1].prompt.split(" ")[:40] == full[0].prompt.split(" ")
