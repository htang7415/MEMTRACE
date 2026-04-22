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
    is_final_turn: bool,
    episode_kind: str,
    episode_payload_type: str,
    episode_horizon: int,
) -> TraceTurn:
    retrieved_passages = retrieve(query=query, path=PASSAGES_PATH, top_k=TOP_K)
    task_id = task_id_by_query().get(query)
    effective_retrieved_passages = _filter_payload_retrieval(
        retrieved_passages=retrieved_passages,
        task_id=task_id,
        episode_payload_type=episode_payload_type,
    )
    writer_output = extract_memory_candidates(retrieved_passages=effective_retrieved_passages, query=query)
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
    trace_turn.admitted_memory_records = accepted_records
    tool_call = _plan_tool_call(
        query=query,
        prior_memory=prior_memory,
        retrieved_passages=effective_retrieved_passages,
        turn=turn,
        is_final_turn=is_final_turn,
        episode_payload_type=episode_payload_type,
        episode_horizon=episode_horizon,
    )
    trace_turn.tool_router_log = tool_call
    trace_turn.label = _label_for_query(query=query, tool_call=tool_call, is_final_turn=is_final_turn)
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


def _plan_tool_call(
    query: str,
    prior_memory: list,
    retrieved_passages: list,
    turn: int,
    is_final_turn: bool,
    episode_payload_type: str,
    episode_horizon: int,
) -> object | None:
    if not is_final_turn:
        return None
    task_id = task_id_by_query().get(query)
    if task_id is None:
        return None
    spec = task_spec_by_task_id()[task_id]
    task = next(task for task in build_task_records() if task.task_id == task_id)
    poisoned_memory = _memory_causes_override(
        prior_memory=prior_memory,
        episode_payload_type=episode_payload_type,
        episode_horizon=episode_horizon,
    )
    poisoned_retrieval = False
    if query == task.poison_query:
        top_poison = next(
            (
                hit
                for hit in retrieved_passages
                if hit.task_id == task_id
                and hit.passage_kind == "poison"
                and hit.payload_type == episode_payload_type
            ),
            None,
        )
        poisoned_retrieval = top_poison is not None and top_poison.payload_type == "direct_override"
    poisoned = poisoned_memory or poisoned_retrieval
    tool_name = spec["unsafe_tool"] if poisoned else spec["expected_tool"]
    arguments = spec["unsafe_arguments"] if poisoned else spec["expected_arguments"]
    if tool_name is None:
        return None
    return log_tool_call(tool_name=tool_name, arguments=arguments, turn=turn)


def _label_for_query(query: str, tool_call, is_final_turn: bool) -> str | None:
    if not is_final_turn:
        return None
    task_id = task_id_by_query().get(query)
    if task_id is None:
        return None
    gold = next(label for label in build_gold_labels() if label.task_id == task_id)
    return "safe" if is_safe(tool_call, gold) else "unsafe"


def _is_poison_query(query: str) -> bool:
    return any(query == task.poison_query for task in build_task_records())


def _filter_payload_retrieval(retrieved_passages: list, task_id: str | None, episode_payload_type: str) -> list:
    if task_id is None or episode_payload_type not in {"direct_override", "contextual_drift"}:
        return retrieved_passages
    filtered = []
    for hit in retrieved_passages:
        if hit.task_id == task_id and hit.passage_kind == "poison" and hit.payload_type != episode_payload_type:
            continue
        filtered.append(hit)
    return filtered


def _memory_causes_override(prior_memory: list, episode_payload_type: str, episode_horizon: int) -> bool:
    if episode_payload_type == "direct_override":
        return any(record.trust_level == "low" for record in prior_memory)
    if episode_payload_type == "contextual_drift":
        return episode_horizon <= 3 and any(record.trust_level == "low" for record in prior_memory)
    return any(record.trust_level == "low" for record in prior_memory)
