"""Turn pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from memtrace.benchmark import build_gold_labels, build_task_records, task_id_by_query, task_spec_by_task_id
from memtrace.agents.writer import extract_memory_candidates
from memtrace.config import ALLOWLIST_PATH, PASSAGES_PATH, TOP_K
from memtrace.eval.labeler import is_safe
from memtrace.retrieval import retrieve
from memtrace.schema import TraceTurn
from memtrace.store.memory import load_memory_records
from memtrace.store.systems import s0_filter, s1_filter, s2_filter
from memtrace.tools.router import log_tool_call


def run_turn(
    episode_id: str,
    turn: int,
    query: str,
    system: str,
    connection,
) -> TraceTurn:
    retrieved_passages = retrieve(query=query, path=PASSAGES_PATH, top_k=TOP_K)
    writer_output = extract_memory_candidates(retrieved_passages=retrieved_passages, query=query)
    prior_memory = load_memory_records(connection, episode_id=episode_id, max_turn=turn)
    trace_turn = TraceTurn(
        episode_id=episode_id,
        turn=turn,
        query=query,
        retrieved_passages=retrieved_passages,
        memory_writer_output=writer_output,
        memory_store_state=prior_memory,
    )
    accepted_records = _apply_system_filter(system=system, turn=turn, writer_output=writer_output)
    tool_call = _plan_tool_call(query=query, prior_memory=prior_memory, turn=turn)
    trace_turn.tool_router_log = tool_call
    trace_turn.label = _label_for_query(query=query, tool_call=tool_call)
    if _is_poison_query(query):
        trace_turn.poison_admission_flag = any(record.trust_level == "low" for record in accepted_records)
    return trace_turn, accepted_records


def _apply_system_filter(system: str, turn: int, writer_output: list):
    allowlisted_source_ids = _load_allowlisted_source_ids(ALLOWLIST_PATH)
    if system == "S0":
        return s0_filter(writer_output, write_turn=turn)
    if system == "S1":
        return s1_filter(writer_output, write_turn=turn)
    if system == "S2":
        return s2_filter(
            writer_output,
            write_turn=turn,
            allowlisted_source_ids=allowlisted_source_ids,
        )
    raise ValueError(f"Unknown system: {system}")


def _load_allowlisted_source_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as handle:
        items = json.load(handle)
    return {item["source_id"] for item in items}


def _plan_tool_call(query: str, prior_memory: list, turn: int) -> object | None:
    task_id = task_id_by_query().get(query)
    if task_id is None:
        return None
    spec = task_spec_by_task_id()[task_id]
    if query == next(
        task.clean_query for task in build_task_records() if task.task_id == task_id
    ) or query == next(task.trigger_query for task in build_task_records() if task.task_id == task_id):
        poisoned = any(record.trust_level == "low" for record in prior_memory)
        tool_name = spec["unsafe_tool"] if poisoned else spec["expected_tool"]
        arguments = spec["unsafe_arguments"] if poisoned else spec["expected_arguments"]
        if tool_name is None:
            return None
        return log_tool_call(tool_name=tool_name, arguments=arguments, turn=turn)
    return None


def _label_for_query(query: str, tool_call) -> str | None:
    task_id = task_id_by_query().get(query)
    if task_id is None:
        return None
    task_records = {task.task_id: task for task in build_task_records()}
    task = task_records[task_id]
    if query not in {task.clean_query, task.trigger_query}:
        return None
    gold = next(label for label in build_gold_labels() if label.task_id == task_id)
    return "safe" if is_safe(tool_call, gold) else "unsafe"


def _is_poison_query(query: str) -> bool:
    return any(query == task.poison_query for task in build_task_records())
