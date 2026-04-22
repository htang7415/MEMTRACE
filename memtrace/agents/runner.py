"""Episode runner scaffolding."""

import json
from pathlib import Path

from memtrace.pipeline import run_turn
from memtrace.store.db import connect, init_db
from memtrace.store.memory import delete_episode_records, insert_memory_records


def run_episode(episode_id: str, turns: list[str], system: str, db_path: Path) -> list[dict]:
    connection = connect(db_path)
    init_db(connection)
    delete_episode_records(connection, episode_id)
    trace = []
    for index, query in enumerate(turns, start=1):
        trace_turn, accepted_records = run_turn(
            episode_id=episode_id,
            turn=index,
            query=query,
            system=system,
            connection=connection,
        )
        if accepted_records:
            insert_memory_records(connection, episode_id=episode_id, records=accepted_records)
        if hasattr(trace_turn, "model_dump"):
            trace.append(trace_turn.model_dump())
        else:
            trace.append(trace_turn.dict())
    connection.close()
    return trace


def save_trace(trace_dir: Path, episode_id: str, trace: list[dict]) -> Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{episode_id.replace(':', '__')}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in trace:
            handle.write(json.dumps(row) + "\n")
    return path
