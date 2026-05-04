"""Calibration helpers for oracle-retrieved-memory evaluation conditions."""

from __future__ import annotations

from pathlib import Path

from memtrace.benchmark import build_task_records
from memtrace.retrieval import load_passages
from memtrace.schema import MemoryRecord

ORACLE_MEMORY_CONDITION = "S1-ORACLE-RETRIEVED-MEMORY"


def oracle_memory_record_for_episode(
    *,
    task_id: str,
    payload_type: str,
    write_turn: int,
    passages_path: Path,
) -> MemoryRecord:
    passage = _poison_passage(task_id=task_id, payload_type=payload_type, passages_path=passages_path)
    task = next(task for task in build_task_records() if task.task_id == task_id)
    return MemoryRecord(
        memory_type="policy_rule" if task.family == "policy_memory" else "tool_argument",
        content=passage["text"],
        source_id=passage["source_id"],
        source_kind="retrieval",
        task_id=task_id,
        trust_level="low",
        write_turn=write_turn,
        validation_status="oracle_inserted",
    )


def _poison_passage(*, task_id: str, payload_type: str, passages_path: Path) -> dict:
    for passage in load_passages(passages_path):
        if (
            passage.get("task_id") == task_id
            and passage.get("payload_type") == payload_type
            and passage.get("passage_kind") == "poison"
        ):
            return passage
    raise ValueError(f"missing poison passage for task_id={task_id!r}, payload_type={payload_type!r}")
