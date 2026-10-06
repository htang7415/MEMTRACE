"""Public-trace workloads for engine and routing benchmarks.

Each workload is a list of single-turn requests (prompt text + output-token cap). The
prompt text is what reaches the engine; prefix structure is preserved so prefix caching
behaves the way it would under the original traffic.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
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
    requests = []
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
