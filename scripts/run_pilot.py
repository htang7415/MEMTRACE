import _bootstrap  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

from memtrace.agents.runner import run_episode, save_trace, trace_path
from memtrace.actors import actor_model_slug
from memtrace.benchmark import build_episode_records
from memtrace.config import (
    ACTOR_MODELS,
    DEFAULT_PLANNER_PROMPT_PATH,
    MEMORY_WRITER_BACKEND,
    PLANNER_BACKEND,
    PLANNER_PROMPT_PATH,
    PROTOCOL_VERSION,
    SQLITE_PATH,
)
from memtrace.eval.metrics import aggregate_metrics
from memtrace.eval.scoring import score_run_summary_items
from memtrace.schema import EpisodeRecord


DEFAULT_TASK_IDS = ("budget-limit-rule", "meeting-time")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a small MEMTRACE pilot slice.")
    parser.add_argument("--actor-model", default=ACTOR_MODELS[0])
    parser.add_argument("--episode-id", action="append", dest="episode_ids")
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument("--system", action="append", dest="systems")
    parser.add_argument("--episode-kind", action="append", dest="episode_kinds")
    parser.add_argument("--payload-type", action="append", dest="payload_types")
    parser.add_argument("--horizon", action="append", type=int, dest="horizons")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out-dir", default="data/pilot/latest")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    episode_ids = set(args.episode_ids or ())
    task_ids = set(args.task_ids or DEFAULT_TASK_IDS)
    systems = set(args.systems or ())
    episode_kinds = set(args.episode_kinds or ())
    payload_types = set(args.payload_types or ())
    horizons = set(args.horizons or ())
    out_dir = Path(args.out_dir)
    traces_dir = out_dir / "traces"
    run_summary_path = out_dir / "run_summary.json"
    episode_scores_path = out_dir / "episode_scores.json"
    metrics_path = out_dir / "metrics.json"
    episode_grid = _episode_grid_for_actor(args.actor_model)
    episodes = [
        episode
        for episode in episode_grid
        if episode.actor_model == args.actor_model
        and (not episode_ids or episode.episode_id in episode_ids)
        and episode.task_id in task_ids
        and (not systems or episode.system in systems)
        and (not episode_kinds or episode.episode_kind in episode_kinds)
        and (not payload_types or episode.payload_type in payload_types)
        and (not horizons or episode.horizon in horizons)
    ]
    if args.limit is not None:
        episodes = episodes[: args.limit]
    if args.dry_run:
        print(f"pilot_episodes={len(episodes)}")
        for episode in episodes:
            print(episode.episode_id)
        return
    summaries = _load_existing_summary(run_summary_path)
    summary_by_episode_id = {item["episode_id"]: item for item in summaries}

    for episode in episodes:
        existing_summary = summary_by_episode_id.get(episode.episode_id)
        output_trace_path = trace_path(traces_dir, episode.episode_id)
        if existing_summary and _summary_trace_reusable(existing_summary, output_trace_path, expected_actor_model=episode.actor_model) and not args.force:
            continue
        if existing_summary:
            summaries = [item for item in summaries if item["episode_id"] != episode.episode_id]
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
        trace_path_for_summary = save_trace(traces_dir, episode.episode_id, trace)
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
        _write_json(run_summary_path, summaries)

    episode_scores = score_run_summary_items(summaries)
    metrics = aggregate_metrics(episode_scores)
    _write_json(episode_scores_path, episode_scores)
    _write_json(metrics_path, metrics)
    print(f"pilot_episodes={len(episodes)}")
    print(f"pilot_completed={len(summaries)}")
    print(f"metrics_path={metrics_path}")


def _episode_grid_for_actor(actor_model: str) -> list[EpisodeRecord]:
    episodes = build_episode_records()
    if any(episode.actor_model == actor_model for episode in episodes):
        return episodes

    template_actor = ACTOR_MODELS[0]
    template_slug = actor_model_slug(template_actor)
    target_slug = actor_model_slug(actor_model)
    cloned = []
    for episode in episodes:
        if episode.actor_model != template_actor:
            continue
        data = episode.model_dump() if hasattr(episode, "model_dump") else episode.dict()
        data["actor_model"] = actor_model
        data["episode_id"] = data["episode_id"].replace(f"{template_slug}:", f"{target_slug}:", 1)
        cloned.append(EpisodeRecord(**data))
    return episodes + cloned


def _load_existing_summary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summary_trace_reusable(item: dict, trace_file: Path, *, expected_actor_model: str | None) -> bool:
    return (
        trace_file.exists()
        and item.get("memory_writer_backend") == MEMORY_WRITER_BACKEND
        and item.get("planner_backend") == PLANNER_BACKEND
        and _prompt_path_matches(item)
        and item.get("protocol_version") == PROTOCOL_VERSION
        and item.get("actor_model") == expected_actor_model
    )


def _prompt_path_matches(item: dict) -> bool:
    recorded = item.get("planner_prompt_path")
    if recorded is None:
        return PLANNER_PROMPT_PATH == DEFAULT_PLANNER_PROMPT_PATH
    return recorded == str(PLANNER_PROMPT_PATH)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])
