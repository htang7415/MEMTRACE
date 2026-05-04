import _bootstrap  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

import memtrace.agents.runner as runner_module
import memtrace.pipeline as pipeline_module
from memtrace.agents.runner import run_episode, save_trace, trace_path
from memtrace.benchmark import build_episode_records
from memtrace.calibration import ORACLE_MEMORY_CONDITION
from memtrace.config import (
    ACTOR_MODELS,
    MEMORY_WRITER_BACKEND,
    PASSAGES_PATH,
    PLANNER_BACKEND,
    PROTOCOL_VERSION,
    SQLITE_PATH,
)
from memtrace.eval.metrics import aggregate_metrics
from memtrace.eval.scoring import score_run_summary_items


DEFAULT_OUT_DIR = Path("data/calibration/oracle_memory")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the MEMTRACE oracle-retrieved-memory calibration.")
    parser.add_argument("--actor-model", default=ACTOR_MODELS[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--db-path", type=Path, default=SQLITE_PATH)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--writer-backend", default=MEMORY_WRITER_BACKEND)
    parser.add_argument("--planner-backend", default=PLANNER_BACKEND)
    parser.add_argument("--passages-path", type=Path, default=PASSAGES_PATH)
    args = parser.parse_args(argv)
    _set_passages_path(args.passages_path)

    episodes = _calibration_episodes(args.actor_model)
    if args.limit is not None:
        episodes = episodes[: args.limit]
    if args.dry_run:
        print(f"calibration_condition={ORACLE_MEMORY_CONDITION}")
        print(f"calibration_episodes={len(episodes)}")
        for episode in episodes:
            print(_calibration_episode_id(episode.episode_id))
        return

    out_dir = args.out_dir
    traces_dir = out_dir / "traces"
    run_summary_path = out_dir / "run_summary.json"
    episode_scores_path = out_dir / "episode_scores.json"
    metrics_path = out_dir / "metrics.json"
    summaries = _load_existing_summary(run_summary_path)
    summary_by_episode_id = {item["episode_id"]: item for item in summaries}

    for episode in episodes:
        calibration_episode_id = _calibration_episode_id(episode.episode_id)
        output_trace_path = trace_path(traces_dir, calibration_episode_id)
        existing_summary = summary_by_episode_id.get(calibration_episode_id)
        if (
            existing_summary
            and _summary_trace_reusable(
                existing_summary,
                output_trace_path,
                expected_actor_model=args.actor_model,
                writer_backend=args.writer_backend,
                planner_backend=args.planner_backend,
            )
            and not args.force
        ):
            continue
        if existing_summary:
            summaries = [item for item in summaries if item["episode_id"] != calibration_episode_id]
        if output_trace_path.exists():
            output_trace_path.unlink()
        trace = run_episode(
            episode_id=calibration_episode_id,
            turns=episode.turns,
            system="S1",
            actor_model=args.actor_model,
            db_path=args.db_path,
            episode_kind=episode.episode_kind,
            episode_payload_type=episode.payload_type,
            episode_horizon=episode.horizon,
            writer_backend=args.writer_backend,
            planner_backend=args.planner_backend,
            calibration_condition=ORACLE_MEMORY_CONDITION,
        )
        trace_path_for_summary = save_trace(traces_dir, calibration_episode_id, trace)
        summaries.append(
            {
                "episode_id": calibration_episode_id,
                "system": ORACLE_MEMORY_CONDITION,
                "actor_model": args.actor_model,
                "episode_kind": episode.episode_kind,
                "payload_type": episode.payload_type,
                "family": episode.family,
                "task_id": episode.task_id,
                "horizon": episode.horizon,
                "trace_path": str(trace_path_for_summary),
                "turn_count": len(trace),
                "memory_writer_backend": args.writer_backend,
                "planner_backend": args.planner_backend,
                "protocol_version": PROTOCOL_VERSION,
                "calibration_condition": ORACLE_MEMORY_CONDITION,
                "oracle_memory_inserted": True,
            }
        )
        _write_json(run_summary_path, summaries)

    episode_scores = score_run_summary_items(summaries)
    metrics = aggregate_metrics(episode_scores)
    _write_json(episode_scores_path, episode_scores)
    _write_json(metrics_path, metrics)
    print(f"calibration_condition={ORACLE_MEMORY_CONDITION}")
    print(f"calibration_completed={len(summaries)}")
    print(f"metrics_path={metrics_path}")


def _calibration_episodes(actor_model: str):
    return [
        episode
        for episode in build_episode_records()
        if episode.actor_model == actor_model
        and episode.system == "S1"
        and episode.episode_kind == "stateful_attack"
    ]


def _calibration_episode_id(episode_id: str) -> str:
    return episode_id.replace(":S1:", f":{ORACLE_MEMORY_CONDITION}:", 1)


def _load_existing_summary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _set_passages_path(path: Path) -> None:
    runner_module.PASSAGES_PATH = path
    pipeline_module.PASSAGES_PATH = path


def _summary_trace_reusable(
    item: dict,
    trace_file: Path,
    *,
    expected_actor_model: str,
    writer_backend: str,
    planner_backend: str,
) -> bool:
    return (
        trace_file.exists()
        and item.get("actor_model") == expected_actor_model
        and item.get("memory_writer_backend") == writer_backend
        and item.get("planner_backend") == planner_backend
        and item.get("protocol_version") == PROTOCOL_VERSION
        and item.get("calibration_condition") == ORACLE_MEMORY_CONDITION
    )


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])
