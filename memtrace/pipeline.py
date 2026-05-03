"""Turn pipeline orchestration."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from memtrace.benchmark import build_task_records, gold_label_for_query, task_id_by_query
from memtrace.agents.planner import plan_tool_call_with_actor_and_raw_output
from memtrace.agents.responder import format_response
from memtrace.agents.writer import extract_memory_candidates, generate_memory_candidates_with_actor
from memtrace.config import ALLOWLIST_PATH, PASSAGES_PATH, TOP_K
from memtrace.eval.labeler import classify_outcome
from memtrace.models.actor import ActorModel
from memtrace.retrieval import retrieve
from memtrace.schema import MemoryRecord, TraceTurn
from memtrace.store.memory import load_memory_records
from memtrace.store.systems import s0_filter_with_rejections, s1_filter_with_rejections, s2_filter_with_rejections
from memtrace.tools.policy_checker import check_policy

LOGGER = logging.getLogger(__name__)


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
    calibration_condition: str | None = None,
    oracle_memory_source_id: str | None = None,
    oracle_memory_inserted: bool | None = None,
    disable_current_turn_poison: bool = False,
) -> tuple[TraceTurn, list[MemoryRecord]]:
    retrieved_passages = retrieve(query=query, path=PASSAGES_PATH, top_k=TOP_K)
    task_id = task_id_by_query().get(query)
    effective_retrieved_passages = _filter_payload_retrieval(
        retrieved_passages=retrieved_passages,
        task_id=task_id,
        episode_payload_type=episode_payload_type,
    )
    if disable_current_turn_poison:
        effective_retrieved_passages = [
            passage for passage in effective_retrieved_passages if passage.passage_kind != "poison"
        ]
    raw_writer_output = None
    if writer_actor is not None:
        writer_output, raw_writer_output = generate_memory_candidates_with_actor(
            retrieved_passages=effective_retrieved_passages,
            query=query,
            actor_model=writer_actor,
        )
    else:
        writer_output = extract_memory_candidates(
            retrieved_passages=effective_retrieved_passages,
            query=query,
            actor_model=None,
        )
    prior_memory = load_memory_records(
        connection,
        episode_id=episode_id,
        max_turn=turn,
        task_id=task_id,
    )
    trace_turn = TraceTurn(
        episode_id=episode_id,
        turn=turn,
        query=query,
        retrieved_passages=effective_retrieved_passages,
        raw_memory_writer_output=raw_writer_output,
        memory_writer_output=writer_output,
        prior_memory_state=prior_memory,
        calibration_condition=calibration_condition,
        oracle_memory_inserted=oracle_memory_inserted,
        oracle_memory_retrieved=(
            None
            if oracle_memory_source_id is None or not is_final_turn
            else any(record.source_id == oracle_memory_source_id for record in prior_memory)
        ),
        current_turn_poison_retrieved_at_trigger=(
            None
            if not is_final_turn
            else any(passage.passage_kind == "poison" for passage in effective_retrieved_passages)
        ),
    )
    accepted_records, rejected_records = _apply_system_filter(system=system, turn=turn, writer_output=writer_output)
    trace_turn.admitted_memory_records = accepted_records
    trace_turn.rejected_memory_records = rejected_records
    trace_turn.memory_store_state = prior_memory + accepted_records
    tool_call, raw_planner_output = _plan_tool_call(
        query=query,
        prior_memory=prior_memory,
        retrieved_passages=effective_retrieved_passages,
        turn=turn,
        is_final_turn=is_final_turn,
        planner_actor=planner_actor,
    )
    trace_turn.tool_router_log = tool_call
    trace_turn.raw_planner_output = raw_planner_output
    gold_label = _gold_label_for_query(query)
    if gold_label:
        policy_check_passed = check_policy(tool_call=tool_call, gold_label=gold_label)
        LOGGER.info(
            "policy_checker episode_id=%s turn=%s pass=%s",
            episode_id,
            turn,
            policy_check_passed,
        )
    trace_turn.planner_output = _planner_output(tool_call=tool_call)
    trace_turn.label = _label_for_query(query=query, tool_call=tool_call, is_final_turn=is_final_turn)
    response_text = format_response(tool_call)
    LOGGER.info("responder episode_id=%s turn=%s response=%r", episode_id, turn, response_text)
    if _is_poison_query(query):
        trace_turn.poison_admission_flag = _admitted_malicious_candidate(
            accepted_records=accepted_records,
            retrieved_passages=effective_retrieved_passages,
        )
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
    planner_actor: ActorModel | None,
) -> tuple[object | None, str | None]:
    if not is_final_turn:
        return None, None
    task_id = task_id_by_query().get(query)
    if task_id is None:
        return None, None
    if planner_actor is None:
        raise RuntimeError("planner execution requires an actor backend")
    return plan_tool_call_with_actor_and_raw_output(
        query=query,
        retrieved_passages=retrieved_passages,
        memory_records=prior_memory,
        turn=turn,
        actor_model=planner_actor,
    )


def _label_for_query(query: str, tool_call, is_final_turn: bool) -> str | None:
    if not is_final_turn:
        return None
    gold = _gold_label_for_query(query)
    if gold is None:
        return None
    return classify_outcome(tool_call, gold)


def _gold_label_for_query(query: str):
    return gold_label_for_query(query)


def _is_poison_query(query: str) -> bool:
    return any(query == task.poison_query for task in build_task_records())


def _filter_payload_retrieval(retrieved_passages: list, task_id: str | None, episode_payload_type: str) -> list:
    if task_id is None or episode_payload_type == "clean_control":
        return retrieved_passages
    filtered = []
    for hit in retrieved_passages:
        if hit.task_id == task_id and hit.passage_kind == "poison" and hit.payload_type != episode_payload_type:
            continue
        filtered.append(hit)
    return filtered


def _admitted_malicious_candidate(
    accepted_records: list[MemoryRecord],
    retrieved_passages: list,
) -> bool:
    poison_source_ids = {
        passage.source_id
        for passage in retrieved_passages
        if passage.passage_kind == "poison"
    }
    return any(
        record.source_kind == "retrieval" and record.source_id in poison_source_ids
        for record in accepted_records
    )


def _planner_output(tool_call) -> dict | None:
    if tool_call is None:
        return None
    return tool_call.model_dump() if hasattr(tool_call, "model_dump") else tool_call.dict()
