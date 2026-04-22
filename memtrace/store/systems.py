"""Memory system policies."""

from __future__ import annotations

from memtrace.config import MAX_MEMORY_CONTENT_CHARS
from memtrace.constants import MEMORY_TYPES
from memtrace.schema import MemoryCandidate, MemoryRecord


def s0_filter(candidates: list[MemoryCandidate], write_turn: int) -> list[MemoryRecord]:
    del candidates, write_turn
    return []


def s1_filter(candidates: list[MemoryCandidate], write_turn: int) -> list[MemoryRecord]:
    return [
        MemoryRecord(
            memory_type=candidate.memory_type,
            content=candidate.content,
            source_id=candidate.source_id,
            source_kind=candidate.source_kind,
            trust_level=_trust_level(candidate.source_kind, allowlisted=False),
            write_turn=write_turn,
            validation_status="accepted",
        )
        for candidate in candidates
    ]


def s2_filter(
    candidates: list[MemoryCandidate],
    write_turn: int,
    allowlisted_source_ids: set[str],
) -> list[MemoryRecord]:
    records: list[MemoryRecord] = []
    for candidate in candidates:
        trust_level = _trust_level(
            candidate.source_kind,
            allowlisted=candidate.source_id in allowlisted_source_ids,
        )
        if candidate.memory_type not in MEMORY_TYPES:
            validation_status = "rejected_schema"
        elif len(candidate.content) >= MAX_MEMORY_CONTENT_CHARS:
            validation_status = "rejected_length"
        elif trust_level != "high":
            validation_status = "rejected_provenance"
        else:
            validation_status = "accepted"
        if validation_status == "accepted":
            records.append(
                MemoryRecord(
                    memory_type=candidate.memory_type,
                    content=candidate.content,
                    source_id=candidate.source_id,
                    source_kind=candidate.source_kind,
                    trust_level=trust_level,
                    write_turn=write_turn,
                    validation_status=validation_status,
                )
            )
    return records


def _trust_level(source_kind: str, allowlisted: bool) -> str:
    if source_kind in {"system_doc", "admin_doc"}:
        return "high"
    if source_kind == "retrieval" and allowlisted:
        return "high"
    return "low"
