"""Public-trace workloads for engine and routing benchmarks.

Each workload is a list of single-turn requests (prompt text + output-token cap). The
prompt text is what reaches the engine; prefix structure is preserved so prefix caching
behaves the way it would under the original traffic.
"""

from __future__ import annotations

import gzip
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Mooncake traces hash prompts in 512-token blocks. Engines here serve small models with
# short context windows, so each block is scaled down to `block_tokens` words. Identical
# hash IDs always render to identical text, so shared prefixes stay shared.
MOONCAKE_BLOCK_TOKENS = 512

_WORDS = (
    "order customer return refund policy account shipping address payment invoice item price "
    "store product size color delivery status update cancel exchange warranty receipt support "
    "ticket agent tool call result search database record user request response check confirm"
).split()


@dataclass(frozen=True)
class Request:
    prompt: str
    max_tokens: int
    truncated_blocks: int = 0


def sharegpt(
    path: Path, *, num_requests: int, max_tokens: int, seed: int = 0, max_prompt_chars: int = 4000
) -> list[Request]:
    """First human turn of randomly sampled ShareGPT conversations (the `vllm bench serve` convention)."""
    conversations = json.loads(path.read_text())
    prompts = [
        turns[0]["value"]
        for turns in (item.get("conversations") or [] for item in conversations)
        if len(turns) >= 2
        and turns[0].get("from") == "human"
        and 0 < len(turns[0].get("value", "")) <= max_prompt_chars
    ]
    rng = random.Random(seed)
    return [Request(prompt, max_tokens) for prompt in rng.sample(prompts, num_requests)]


def mooncake(
    path: Path, *, num_requests: int, max_tokens: int, block_tokens: int = 32, max_blocks: int = 100
) -> list[Request]:
    """First `num_requests` records of a Mooncake trace, in trace order, with scaled prefix blocks.

    Prompts longer than `max_blocks` keep only their leading blocks, so they still fit the
    engine's context window and still share whatever prefix they shared in the trace.
    """
    requests: list[Request] = []
    with path.open() as handle:
        for line in handle:
            if len(requests) == num_requests:
                break
            record = json.loads(line)
            hash_ids = record["hash_ids"][:max_blocks]
            prompt = " ".join(_block_text(hash_id, block_tokens) for hash_id in hash_ids)
            output_cap = max(1, min(max_tokens, round(record["output_length"] * block_tokens / MOONCAKE_BLOCK_TOKENS)))
            requests.append(Request(prompt, output_cap, len(record["hash_ids"]) - len(hash_ids)))
    return requests


def _block_text(hash_id: int, block_tokens: int) -> str:
    rng = random.Random(hash_id)
    return " ".join(rng.choice(_WORDS) for _ in range(block_tokens))


def agent_sessions(
    *,
    num_sessions: int,
    turns: int,
    max_tokens: int,
    prefix_words: int = 600,
    turn_words: int = 150,
    seed: int = 0,
) -> list[Request]:
    """Synthetic multi-turn agent traffic: each session has its own long prefix (system prompt
    and tool definitions) and resends its whole growing history every turn, the append-only
    prompt pattern of agent loops. Requests interleave sessions turn by turn, so a replica must
    hold many sessions' prefixes at once for prefix caching to pay off."""
    rng = random.Random(seed)
    sessions = []
    for session in range(num_sessions):
        # Integer seeds, not hash(): hashing tuples with strings is randomized per process.
        base = (seed * 1_000_003 + session) * 10_007
        prefix = _block_text(base, prefix_words)
        chunks = [_block_text(base + 1 + turn, turn_words) for turn in range(turns)]
        sessions.append((prefix, chunks))
    requests = []
    for turn in range(turns):
        order = list(range(num_sessions))
        rng.shuffle(order)
        for session in order:
            prefix, chunks = sessions[session]
            requests.append(Request(" ".join([prefix, *chunks[: turn + 1]]), max_tokens))
    return requests


@dataclass(frozen=True)
class AgentCall:
    prompt: str
    max_tokens: int
    gap_before: float  # seconds to wait after the previous call in the session finishes
    prompt_target: int  # scaled prompt length in words (~tokens)
    cached_target: int  # leading words reused from the session's previous prompt


@dataclass(frozen=True)
class AgentSession:
    session_id: str
    start_offset: float  # seconds after the replay starts
    calls: tuple[AgentCall, ...]


def _trace_time(stamp: str) -> float:
    base, _, frac = stamp.rstrip("Z").partition(".")
    return datetime.fromisoformat(base + "+00:00").timestamp() + float("0." + (frac or "0"))


def copilot_sessions(
    paths: list[Path],
    *,
    num_sessions: int,
    seed: int = 0,
    token_scale: float = 1 / 40,
    max_prompt_tokens: int = 3500,
    max_calls: int = 40,
    gap_scale: float = 0.1,
    max_gap_seconds: float = 30.0,
    window_seconds: float = 300.0,
    output_range: tuple[int, int] = (4, 128),
) -> list[AgentSession]:
    """Replayable agent sessions from the GitHub Copilot coding-agent traces (Azure, 2026).

    The traces carry token counts but no text. Each call's prompt is synthesized so that its first
    `cached` tokens repeat the session's previous prompt and the rest is new, reproducing the real
    prefix-cache structure; all lengths are scaled by `token_scale` to fit a small model's context.
    Call `timestamp`s mark completion (consecutive calls never overlap under that reading; 32% would if
    they marked the start), so a call starts at `timestamp - duration_ms` and the gap before it is
    measured from the previous call's completion. Gaps are compressed by `gap_scale` and capped at
    `max_gap_seconds` (the long tail is a user idle between turns, up to 46 minutes after compression,
    which would stretch a replay without adding load), and session start
    times are compressed into `window_seconds` in their real order. Calls without token counts are skipped.
    """
    raw = []
    for path in paths:
        with gzip.open(path, "rt") as handle:
            for line in handle:
                session = json.loads(line)
                calls = sorted(
                    (
                        c
                        for turn in session["turns"]
                        for c in turn["llm_calls"]
                        if all(isinstance(c["tokens"].get(k), int) for k in ("prompt", "cached", "completion"))
                    ),
                    key=lambda c: c["timestamp"],
                )
                if len(calls) >= 2:
                    raw.append((session["session_id"], calls[:max_calls]))
    raw.sort(key=lambda item: item[0])
    chosen = random.Random(seed).sample(raw, min(num_sessions, len(raw)))
    starts = [_trace_time(calls[0]["timestamp"]) - calls[0]["duration_ms"] / 1000 for _, calls in chosen]
    first, span = min(starts), (max(starts) - min(starts)) or 1.0
    low, high = output_range

    sessions = []
    for (session_id, calls), start in zip(chosen, starts):
        rng = random.Random(f"{seed}:{session_id}")
        context: list[str] = []
        previous_end = None
        built = []
        for call in calls:
            tokens = call["tokens"]
            length = min(max_prompt_tokens, max(1, round(tokens["prompt"] * token_scale)))
            cached = min(length, len(context), round(tokens["cached"] * token_scale))
            words = context[:cached] + [rng.choice(_WORDS) for _ in range(length - cached)]
            end = _trace_time(call["timestamp"])
            begin = end - call["duration_ms"] / 1000
            gap = 0.0 if previous_end is None else min(max_gap_seconds, max(0.0, begin - previous_end) * gap_scale)
            output = min(high, max(low, round(tokens["completion"] * token_scale)))
            built.append(AgentCall(" ".join(words), output, gap, length, cached))
            context, previous_end = words, end
        sessions.append(AgentSession(session_id, (start - first) / span * window_seconds, tuple(built)))
    return sorted(sessions, key=lambda s: s.start_offset)
