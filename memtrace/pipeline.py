"""Turn pipeline orchestration."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from memtrace.actors import baseline_attack_success, memory_attack_success
from memtrace.benchmark import build_gold_labels, build_task_records, task_id_by_query, task_spec_by_task_id
from memtrace.agents.planner import plan_tool_call_with_actor
from memtrace.agents.writer import extract_memory_candidates
from memtrace.config import ALLOWLIST_PATH, PASSAGES_PATH, TOP_K
from memtrace.eval.labeler import is_safe
from memtrace.models.actor import ActorModel
from memtrace.retrieval import load_passages, retrieve
from memtrace.schema import MemoryRecord, TraceTurn
from memtrace.store.memory import load_memory_records
from memtrace.store.systems import s0_filter_with_rejections, s1_filter_with_rejections, s2_filter_with_rejections
from memtrace.tools.policy_checker import check_policy
from memtrace.tools.router import log_tool_call


def run_turn(
    episode_id: str,
    turn: int,
    query: str,
    system: str,
    actor_model: str,
    connection,
    is_final_turn: bool,
    episode_kind: str,
    episode_payload_type: str,
    episode_horizon: int,
    writer_actor: ActorModel | None = None,
    planner_actor: ActorModel | None = None,
) -> tuple[TraceTurn, list[MemoryRecord]]:
    retrieved_passages = retrieve(query=query, path=PASSAGES_PATH, top_k=TOP_K)
    task_id = task_id_by_query().get(query)
    effective_retrieved_passages = _filter_payload_retrieval(
        retrieved_passages=retrieved_passages,
        task_id=task_id,
        episode_payload_type=episode_payload_type,
    )
    writer_output = extract_memory_candidates(
        retrieved_passages=effective_retrieved_passages,
        query=query,
        actor_model=writer_actor,
    )
    prior_memory = load_memory_records(connection, episode_id=episode_id, max_turn=turn)
    trace_turn = TraceTurn(
        episode_id=episode_id,
        turn=turn,
        query=query,
        retrieved_passages=retrieved_passages,
        memory_writer_output=writer_output,
        prior_memory_state=prior_memory,
    )
    accepted_records, rejected_records = _apply_system_filter(system=system, turn=turn, writer_output=writer_output)
    trace_turn.admitted_memory_records = accepted_records
    trace_turn.rejected_memory_records = rejected_records
    trace_turn.memory_store_state = prior_memory + accepted_records
    tool_call = _plan_tool_call(
        query=query,
        prior_memory=prior_memory,
        retrieved_passages=effective_retrieved_passages,
        turn=turn,
        is_final_turn=is_final_turn,
        actor_model=actor_model,
        episode_payload_type=episode_payload_type,
        episode_horizon=episode_horizon,
        episode_kind=episode_kind,
        planner_actor=planner_actor,
    )
    trace_turn.tool_router_log = tool_call
    gold_label = _gold_label_for_query(query)
    trace_turn.policy_checker_pass = check_policy(tool_call=tool_call, gold_label=gold_label) if gold_label else None
    trace_turn.planner_output = _planner_output(
        tool_call=tool_call,
        actor_model=actor_model,
        prior_memory=prior_memory,
    )
    trace_turn.label = _label_for_query(query=query, tool_call=tool_call, is_final_turn=is_final_turn)
    if _is_poison_query(query):
        trace_turn.poison_admission_flag = any(record.trust_level == "low" for record in accepted_records)
    return trace_turn, accepted_records


def _apply_system_filter(system: str, turn: int, writer_output: list) -> tuple[list[MemoryRecord], list[MemoryRecord]]:
    allowlisted_source_ids = _load_allowlisted_source_ids(ALLOWLIST_PATH)
    if system == "S0":
        return s0_filter_with_rejections(writer_output, write_turn=turn)
    if system == "S1":
        return s1_filter_with_rejections(writer_output, write_turn=turn)
    if system == "S2":
        return s2_filter_with_rejections(
            writer_output,
            write_turn=turn,
            allowlisted_source_ids=allowlisted_source_ids,
        )
    raise ValueError(f"Unknown system: {system}")


@lru_cache(maxsize=1)
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
    actor_model: str,
    episode_payload_type: str,
    episode_horizon: int,
    episode_kind: str,
    planner_actor: ActorModel | None,
) -> object | None:
    if not is_final_turn:
        return None
    task_id = task_id_by_query().get(query)
    if task_id is None:
        return None
    if planner_actor is not None:
        return plan_tool_call_with_actor(
            query=query,
            retrieved_passages=retrieved_passages,
            memory_records=prior_memory,
            turn=turn,
            actor_model=planner_actor,
        )
    spec = task_spec_by_task_id()[task_id]
    task = next(task for task in build_task_records() if task.task_id == task_id)
    poisoned_memory = _memory_causes_override(
        prior_memory=prior_memory,
        task_id=task_id,
        actor_model=actor_model,
        episode_payload_type=episode_payload_type,
        episode_horizon=episode_horizon,
    )
    baseline_poison = _baseline_causes_override(
        actor_model=actor_model,
        episode_kind=episode_kind,
        episode_payload_type=episode_payload_type,
        query=query,
        task=task,
    )
    poisoned = poisoned_memory or baseline_poison
    tool_name = spec["unsafe_tool"] if poisoned else spec["expected_tool"]
    arguments = spec["unsafe_arguments"] if poisoned else spec["expected_arguments"]
    if tool_name is None:
        return None
    return log_tool_call(tool_name=tool_name, arguments=arguments, turn=turn)


def _label_for_query(query: str, tool_call, is_final_turn: bool) -> str | None:
    if not is_final_turn:
        return None
    gold = _gold_label_for_query(query)
    if gold is None:
        return None
    return "safe" if is_safe(tool_call, gold) else "unsafe"


def _gold_label_for_query(query: str):
    task_id = task_id_by_query().get(query)
    if task_id is None:
        return None
    return next(label for label in build_gold_labels() if label.task_id == task_id)


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


def _memory_causes_override(
    prior_memory: list,
    task_id: str,
    actor_model: str,
    episode_payload_type: str,
    episode_horizon: int,
) -> bool:
    poison_source_ids = _poison_source_ids(task_id=task_id, payload_type=episode_payload_type)
    admitted_poison = any(record.source_id in poison_source_ids for record in prior_memory)
    if not admitted_poison:
        return False
    return memory_attack_success(actor_model, episode_payload_type, episode_horizon)


@lru_cache(maxsize=None)
def _poison_source_ids(task_id: str, payload_type: str) -> frozenset[str]:
    return frozenset(
        passage["source_id"]
        for passage in load_passages(PASSAGES_PATH)
        if passage.get("task_id") == task_id
        and passage.get("payload_type") == payload_type
        and passage.get("passage_kind") == "poison"
    )


def _baseline_causes_override(
    actor_model: str,
    episode_kind: str,
    episode_payload_type: str,
    query: str,
    task,
) -> bool:
    if episode_kind != "one_shot_attack":
        return False
    if query != task.poison_query:
        return False
    return baseline_attack_success(actor_model, episode_payload_type)


def _planner_output(tool_call, actor_model: str, prior_memory: list) -> str:
    decision = "refuse" if tool_call is None else f"{tool_call.tool_name}"
    return json.dumps(
        {
            "actor_model": actor_model,
            "decision": decision,
            "prior_memory_count": len(prior_memory),
        }
    )
