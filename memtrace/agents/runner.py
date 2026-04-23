"""Episode runner scaffolding."""

import json
from functools import lru_cache
from pathlib import Path

from memtrace.config import ACTOR_MODELS, MEMORY_WRITER_BACKEND, PLANNER_BACKEND
from memtrace.models.mlx_runner import load_actor
from memtrace.pipeline import run_turn
from memtrace.store.db import connect, init_db
from memtrace.store.memory import delete_episode_records, insert_memory_records


def run_episode(
    episode_id: str,
    turns: list[str],
    system: str,
    db_path: Path,
    actor_model: str | None = None,
    episode_kind: str | None = None,
    episode_payload_type: str | None = None,
    episode_horizon: int | None = None,
    writer_backend: str | None = None,
    planner_backend: str | None = None,
) -> list[dict]:
    connection = connect(db_path)
    init_db(connection)
    delete_episode_records(connection, episode_id)
    resolved_actor_model = actor_model or ACTOR_MODELS[0]
    resolved_episode_kind = episode_kind or _episode_kind_from_episode_id(episode_id)
    resolved_payload_type = episode_payload_type or _payload_type_from_episode_id(episode_id)
    resolved_horizon = episode_horizon if episode_horizon is not None else _horizon_from_episode_id(episode_id)
    writer_actor = _load_actor_for_backend(resolved_actor_model, writer_backend or MEMORY_WRITER_BACKEND)
    planner_actor = _load_actor_for_backend(resolved_actor_model, planner_backend or PLANNER_BACKEND)
    trace = []
    for index, query in enumerate(turns, start=1):
        trace_turn, accepted_records = run_turn(
            episode_id=episode_id,
            turn=index,
            query=query,
            system=system,
            actor_model=resolved_actor_model,
            connection=connection,
            is_final_turn=index == len(turns),
            episode_kind=resolved_episode_kind,
            episode_payload_type=resolved_payload_type,
            episode_horizon=resolved_horizon,
            writer_actor=writer_actor,
            planner_actor=planner_actor,
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
    path = trace_path(trace_dir, episode_id)
    with path.open("w", encoding="utf-8") as handle:
        for row in trace:
            handle.write(json.dumps(row) + "\n")
    return path


def trace_path(trace_dir: Path, episode_id: str) -> Path:
    return trace_dir / f"{episode_id.replace(':', '__')}.jsonl"


def _episode_kind_from_episode_id(episode_id: str) -> str:
    if ":stateful:" in episode_id:
        return "stateful_attack"
    if ":one-shot:" in episode_id:
        return "one_shot_attack"
    return "clean_control"


def _payload_type_from_episode_id(episode_id: str) -> str:
    if episode_id.endswith(":direct_override"):
        return "direct_override"
    if episode_id.endswith(":contextual_drift"):
        return "contextual_drift"
    return "clean_control"


def _horizon_from_episode_id(episode_id: str) -> int:
    if ":stateful:d1:" in episode_id:
        return 1
    if ":stateful:d3:" in episode_id:
        return 3
    if ":stateful:d7:" in episode_id:
        return 7
    return 1


@lru_cache(maxsize=None)
def _load_actor_for_backend(model_name: str, backend: str):
    return load_actor(model_name, backend=backend)
