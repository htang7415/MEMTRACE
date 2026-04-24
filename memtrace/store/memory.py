"""Persistent memory store primitives."""

from __future__ import annotations

import sqlite3

from memtrace.schema import MemoryRecord


def serialize_memory_block(records: list[MemoryRecord]) -> str:
    return "\n".join(
        f"[{record.memory_type}] {record.content} (source: {record.source_id})"
        for record in records
    )


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
