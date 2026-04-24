"""SQLite store helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_records (
            episode_id TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            content TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            task_id TEXT,
            trust_level TEXT NOT NULL,
            write_turn INTEGER NOT NULL,
            validation_status TEXT NOT NULL
        )
        """
    )
    columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(memory_records)").fetchall()
    }
    if "task_id" not in columns:
        connection.execute("ALTER TABLE memory_records ADD COLUMN task_id TEXT")
    connection.commit()
