"""Persistent memory store primitives."""

from __future__ import annotations

import sqlite3

from memtrace.core.constants import MEMORY_CONFLICT_RESOLUTION_LATEST_WINS, MEMORY_CONFLICT_RESOLUTION_NONE
from memtrace.core.schema import MemoryRecord


def serialize_memory_block(records: list[MemoryRecord]) -> str:
    return "\n".join(f"[{record.memory_type}] {record.content} (source: {record.source_id})" for record in records)


def insert_memory_records(
    connection: sqlite3.Connection,
    episode_id: str,
    records: list[MemoryRecord],
) -> None:
    connection.executemany(
        """
        INSERT INTO memory_records (
            episode_id, memory_type, content, source_id, source_kind,
            task_id, trust_level, write_turn, validation_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                episode_id,
                record.memory_type,
                record.content,
                record.source_id,
                record.source_kind,
                record.task_id,
                record.trust_level,
                record.write_turn,
                record.validation_status,
            )
            for record in records
        ],
    )
    connection.commit()


def delete_episode_records(connection: sqlite3.Connection, episode_id: str) -> None:
    connection.execute("DELETE FROM memory_records WHERE episode_id = ?", (episode_id,))
    connection.commit()


def load_memory_records(
    connection: sqlite3.Connection,
    episode_id: str,
    max_turn: int,
    task_id: str | None = None,
) -> list[MemoryRecord]:
    if task_id is None:
        rows = connection.execute(
            """
            SELECT memory_type, content, source_id, source_kind, task_id, trust_level, write_turn, validation_status
            FROM memory_records
            WHERE episode_id = ? AND write_turn < ?
            ORDER BY write_turn ASC, source_id ASC
            """,
            (episode_id, max_turn),
        ).fetchall()
    else:
        rows = connection.execute(
            """
            SELECT memory_type, content, source_id, source_kind, task_id, trust_level, write_turn, validation_status
            FROM memory_records
            WHERE episode_id = ? AND write_turn < ? AND task_id = ?
            ORDER BY write_turn ASC, source_id ASC
            """,
            (episode_id, max_turn, task_id),
        ).fetchall()
    return [
        MemoryRecord(
            memory_type=row["memory_type"],
            content=row["content"],
            source_id=row["source_id"],
            source_kind=row["source_kind"],
            task_id=row["task_id"],
            trust_level=row["trust_level"],
            write_turn=row["write_turn"],
            validation_status=row["validation_status"],
        )
        for row in rows
    ]


def resolve_visible_memory(
    records: list[MemoryRecord],
    *,
    current_turn: int,
    conflict_resolution: str = MEMORY_CONFLICT_RESOLUTION_NONE,
    ttl_turns: int | None = None,
) -> list[MemoryRecord]:
    """Return the planner-visible slice of `records` under an opt-in policy.

    Defaults (`conflict_resolution="none"`, `ttl_turns=None`) reproduce the legacy
    behavior exactly: every admitted record stays visible for the rest of the episode.
    This does not affect the persisted store or `prior_memory_state`/`memory_store_state`
    trace fields -- callers that need the full write history should keep using the raw
    `records` returned by `load_memory_records`.
    """

    visible = records
    if ttl_turns is not None:
        visible = [record for record in visible if record.write_turn >= current_turn - ttl_turns]
    if conflict_resolution == MEMORY_CONFLICT_RESOLUTION_LATEST_WINS:
        visible = _latest_record_per_slot(visible)
    elif conflict_resolution != MEMORY_CONFLICT_RESOLUTION_NONE:
        raise ValueError(f"Unknown memory conflict resolution mode: {conflict_resolution}")
    return visible


def _latest_record_per_slot(records: list[MemoryRecord]) -> list[MemoryRecord]:
    """Keep only the highest-write_turn record per (task_id, memory_type) slot.

    `records` is expected write_turn-ascending (as returned by `load_memory_records`),
    so a later record simply overwrites an earlier one in the same slot. Ties within the
    same turn are broken by source_id, matching `load_memory_records`'s own ordering.
    """

    latest_by_slot: dict[tuple[str | None, str], MemoryRecord] = {}
    for record in records:
        latest_by_slot[(record.task_id, record.memory_type)] = record
    return sorted(latest_by_slot.values(), key=lambda record: (record.write_turn, record.source_id))
