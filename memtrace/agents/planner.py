"""Planner scaffolding."""

import json

from memtrace.models.actor import ActorModel
from memtrace.config import PLANNER_MAX_TOKENS, PROMPTS_DIR
from memtrace.parsing import extract_json_payload
from memtrace.schema import MemoryRecord, RetrievedPassage, ToolCall
from memtrace.store.memory import serialize_memory_block


def build_planner_input(query: str, retrieved_context: str, memory_block: str) -> str:
    prompt = (PROMPTS_DIR / "planner.txt").read_text(encoding="utf-8")
    return "\n".join(
        [
            prompt,
            "",
            f"Query: {query}",
            f"Retrieved:\n{retrieved_context}",
            f"Memory:\n{memory_block}",
        ]
    )


def format_retrieved_context(passages: list[RetrievedPassage]) -> str:
    return "\n".join(
        f"[source_id: {passage.source_id}, rank: {passage.rank}] {passage.text}"
        for passage in passages
    )


def build_planner_input_from_records(
    query: str,
    retrieved_passages: list[RetrievedPassage],
    memory_records: list[MemoryRecord],
) -> str:
    return build_planner_input(
        query=query,
        retrieved_context=format_retrieved_context(retrieved_passages),
        memory_block=serialize_memory_block(memory_records),
    )


def parse_planner_json_output(raw_output: str, turn: int) -> ToolCall | None:
    try:
        decoded = json.loads(extract_json_payload(raw_output))
    except json.JSONDecodeError:
        return None
    if decoded is None:
        return None
    if not isinstance(decoded, dict):
        return None
    if not {"tool_name", "arguments"} <= set(decoded):
        return None
    if not isinstance(decoded["tool_name"], str) or not isinstance(decoded["arguments"], dict):
        return None
    return ToolCall(tool_name=decoded["tool_name"], arguments=decoded["arguments"], turn=turn)


def plan_tool_call_with_actor_and_raw_output(
    query: str,
    retrieved_passages: list[RetrievedPassage],
    memory_records: list[MemoryRecord],
    turn: int,
    actor_model: ActorModel,
) -> tuple[ToolCall | None, str]:
    planner_input = build_planner_input_from_records(
        query=query,
        retrieved_passages=retrieved_passages,
        memory_records=memory_records,
    )
    raw_output = actor_model.generate(planner_input, max_tokens=PLANNER_MAX_TOKENS)
    return parse_planner_json_output(raw_output=raw_output, turn=turn), raw_output


def plan_tool_call_with_actor(
    query: str,
    retrieved_passages: list[RetrievedPassage],
    memory_records: list[MemoryRecord],
    turn: int,
    actor_model: ActorModel,
) -> ToolCall | None:
    tool_call, _ = plan_tool_call_with_actor_and_raw_output(
        query=query,
        retrieved_passages=retrieved_passages,
        memory_records=memory_records,
        turn=turn,
        actor_model=actor_model,
    )
    return tool_call
