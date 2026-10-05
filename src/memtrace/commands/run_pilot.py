import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from memtrace.backends.retrieval import retrieve
from memtrace.core.agents.runner import run_episode, save_trace, trace_path
from memtrace.core.actors import actor_model_slug
from memtrace.core.benchmark import build_episode_records
from memtrace.config import (
    ACTOR_MODELS,
    DEFAULT_PLANNER_PROMPT_PATH,
    MEMORY_WRITER_BACKEND,
    PLANNER_BACKEND,
    PASSAGES_PATH,
    PLANNER_PROMPT_PATH,
    PROTOCOL_VERSION,
    SQLITE_PATH,
    TOP_K,
)
from memtrace.evaluation.metrics import aggregate_metrics
from memtrace.evaluation.scoring import score_run_summary_items
from memtrace.evaluation.serving import summarize_serving
from memtrace.core.schema import EpisodeRecord


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
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Episodes in flight at once; values above 1 require the openai backend for writer and planner.",
    )
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.concurrency > 1 and (MEMORY_WRITER_BACKEND, PLANNER_BACKEND) != ("openai", "openai"):
        parser.error("--concurrency above 1 requires memory_writer_backend and planner_backend = 'openai'")

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

    pending = []
    for episode in episodes:
        existing_summary = summary_by_episode_id.get(episode.episode_id)
        output_trace_path = trace_path(traces_dir, episode.episode_id)
        if (
            existing_summary
            and _summary_trace_reusable(existing_summary, output_trace_path, expected_actor_model=episode.actor_model)
            and not args.force
        ):
            continue
        if existing_summary:
            summaries = [item for item in summaries if item["episode_id"] != episode.episode_id]
        if output_trace_path.exists():
            output_trace_path.unlink()
        pending.append(episode)

    if pending and "openai" in (MEMORY_WRITER_BACKEND, PLANNER_BACKEND):
        # Load the retrieval model before timing so serving throughput excludes one-off startup cost.
        retrieve(query=pending[0].turns[0], path=PASSAGES_PATH, top_k=TOP_K)
    executed_traces = []
    started = time.monotonic()
    for episode, trace in _execute_episodes(pending, args.concurrency):
        executed_traces.append(trace)
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
    serving = summarize_serving(
        executed_traces, wall_seconds=time.monotonic() - started, concurrency=args.concurrency
    )

    episode_scores = score_run_summary_items(summaries)
    metrics = aggregate_metrics(episode_scores)
    _write_json(episode_scores_path, episode_scores)
    _write_json(metrics_path, metrics)
    print(f"pilot_episodes={len(episodes)}")
    print(f"pilot_completed={len(summaries)}")
    print(f"metrics_path={metrics_path}")
    if serving is not None:
        serving_path = out_dir / "serving.json"
        _write_json(serving_path, serving)
        print(f"serving_path={serving_path}")


def _run_pilot_episode(episode: EpisodeRecord) -> list[dict]:
    return run_episode(
        episode_id=episode.episode_id,
        turns=episode.turns,
        system=episode.system,
        actor_model=episode.actor_model or "",
        db_path=SQLITE_PATH,
        episode_kind=episode.episode_kind,
        episode_payload_type=episode.payload_type,
        episode_horizon=episode.horizon,
    )


def _execute_episodes(episodes: list[EpisodeRecord], concurrency: int):
    """Yield (episode, trace); sequential in-thread at concurrency 1 so in-process backends are unaffected."""
    if concurrency == 1:
        for episode in episodes:
            yield episode, _run_pilot_episode(episode)
        return
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_run_pilot_episode, episode): episode for episode in episodes}
        for future in as_completed(futures):
            yield futures[future], future.result()


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
