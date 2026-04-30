import _bootstrap  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

from memtrace.agents.runner import run_episode, save_trace, trace_path
from memtrace.config import (
    DEFAULT_PLANNER_PROMPT_PATH,
    EPISODES_PATH,
    MEMORY_WRITER_BACKEND,
    PLANNER_BACKEND,
    PLANNER_PROMPT_PATH,
    PROTOCOL_VERSION,
    RUN_SUMMARY_PATH,
    SQLITE_PATH,
    TRACES_DIR,
)
from memtrace.episodes import load_episodes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run official MEMTRACE experiments.")
    parser.parse_args([] if argv is None else argv)

    _require_official_backends()
    episodes = load_episodes(EPISODES_PATH)
    summaries = _load_existing_summary()
    summary_by_episode_id = {item["episode_id"]: item for item in summaries}
    for episode in episodes:
        existing_summary = summary_by_episode_id.get(episode.episode_id)
        reusable_trace = existing_summary and _summary_trace_complete(
            existing_summary,
            expected_turn_count=len(episode.turns),
            expected_actor_model=episode.actor_model,
        )
        if reusable_trace:
            continue
        if existing_summary:
            summaries = [item for item in summaries if item["episode_id"] != episode.episode_id]
        output_trace_path = trace_path(TRACES_DIR, episode.episode_id)
        if output_trace_path.exists():
            output_trace_path.unlink()
        trace = run_episode(
            episode_id=episode.episode_id,
            turns=episode.turns,
            system=episode.system,
            actor_model=episode.actor_model or "",
            db_path=SQLITE_PATH,
            episode_kind=episode.episode_kind,
            episode_payload_type=episode.payload_type,
            episode_horizon=episode.horizon,
        )
        trace_path_for_summary = save_trace(TRACES_DIR, episode.episode_id, trace)
        summaries.append(
            {
                "episode_id": episode.episode_id,
                "system": episode.system,
                "actor_model": episode.actor_model,
                "episode_kind": episode.episode_kind,
                "payload_type": episode.payload_type,
                "family": episode.family,
                "task_id": episode.task_id,
                "horizon": episode.horizon,
                "trace_path": str(trace_path_for_summary),
                "turn_count": len(trace),
                "memory_writer_backend": MEMORY_WRITER_BACKEND,
                "planner_backend": PLANNER_BACKEND,
                "planner_prompt_path": str(PLANNER_PROMPT_PATH),
                "protocol_version": PROTOCOL_VERSION,
            }
        )
        _write_summary(summaries)
    print(f"episodes_run={len(summaries)}")
    print(f"summary_path={RUN_SUMMARY_PATH}")


def _load_existing_summary() -> list[dict]:
    if not RUN_SUMMARY_PATH.exists():
        return []
    with RUN_SUMMARY_PATH.open("r", encoding="utf-8") as handle:
        summaries = json.load(handle)
    return summaries


def _summary_trace_complete(item: dict, expected_turn_count: int, expected_actor_model: str | None) -> bool:
    trace_file = Path(item["trace_path"])
    if not trace_file.exists():
        return False
    return (
        len(_load_trace(trace_file)) == expected_turn_count
        and item.get("memory_writer_backend") == MEMORY_WRITER_BACKEND
        and item.get("planner_backend") == PLANNER_BACKEND
        and _prompt_path_matches(item)
        and item.get("protocol_version") == PROTOCOL_VERSION
        and item.get("actor_model") == expected_actor_model
    )


def _require_official_backends() -> None:
    invalid = [backend for backend in (MEMORY_WRITER_BACKEND, PLANNER_BACKEND) if backend != "mlx"]
    if invalid:
        raise SystemExit(
            "official benchmark runs must use the MLX backends declared in project.md; "
            f"got memory_writer={MEMORY_WRITER_BACKEND}, planner={PLANNER_BACKEND}"
        )


def _prompt_path_matches(item: dict) -> bool:
    recorded = item.get("planner_prompt_path")
    if recorded is None:
        return PLANNER_PROMPT_PATH == DEFAULT_PLANNER_PROMPT_PATH
    return recorded == str(PLANNER_PROMPT_PATH)


def _write_summary(summaries: list[dict]) -> None:
    RUN_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUN_SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)


def _load_trace(path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main(sys.argv[1:])
