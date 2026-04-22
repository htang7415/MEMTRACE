import _bootstrap  # noqa: F401

import json
from pathlib import Path

from memtrace.agents.runner import run_episode, save_trace, trace_path
from memtrace.config import EPISODES_PATH, MEMORY_WRITER_BACKEND, PLANNER_BACKEND, RUN_SUMMARY_PATH, SQLITE_PATH, TRACES_DIR
from memtrace.episodes import load_episodes


def main() -> None:
    episodes = load_episodes(EPISODES_PATH)
    summaries = _load_existing_summary()
    summary_by_episode_id = {item["episode_id"]: item for item in summaries}
    for episode in episodes:
        existing_summary = summary_by_episode_id.get(episode.episode_id)
        if existing_summary and _summary_trace_complete(existing_summary, expected_turn_count=len(episode.turns)):
            continue
        if existing_summary:
            summaries = [item for item in summaries if item["episode_id"] != episode.episode_id]
        output_trace_path = trace_path(TRACES_DIR, episode.episode_id)
        if output_trace_path.exists():
            trace = _load_trace(output_trace_path)
        else:
            trace = []
        if len(trace) == len(episode.turns):
            trace_path_for_summary = output_trace_path
        else:
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


def _summary_trace_complete(item: dict, expected_turn_count: int) -> bool:
    trace_file = Path(item["trace_path"])
    if not trace_file.exists():
        return False
    return len(_load_trace(trace_file)) == expected_turn_count


def _write_summary(summaries: list[dict]) -> None:
    RUN_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUN_SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)


def _load_trace(path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
