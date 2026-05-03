import _bootstrap  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

import memtrace.pipeline as pipeline_module
from memtrace.agents.runner import run_episode, save_trace, trace_path
from memtrace.benchmark import build_stateful_stress_episode_records
from memtrace.config import (
    ACTOR_MODELS,
    MEMORY_WRITER_BACKEND,
    PLANNER_BACKEND,
    PLANNER_PROMPT_PATH,
    PROTOCOL_VERSION,
    SQLITE_PATH,
)
from memtrace.corpus import build_allowlist, build_corpus, save_json, save_jsonl
from memtrace.eval.metrics import aggregate_metrics
from memtrace.eval.scoring import score_run_summary_items
from memtrace.episodes import save_episodes


DEFAULT_OUT_DIR = Path("data/extensions/stateful_stress")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the optional MEMTRACE stateful stress suite.")
    parser.add_argument("--actor-model", default=ACTOR_MODELS[0])
    parser.add_argument("--system", action="append", dest="systems")
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument("--horizon", action="append", type=int, dest="horizons")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--writer-backend", default=MEMORY_WRITER_BACKEND)
    parser.add_argument("--planner-backend", default=PLANNER_BACKEND)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    systems = tuple(args.systems or ("S1", "S2"))
    task_ids = set(args.task_ids or ())
    horizons = set(args.horizons or ())
    episodes = [
        episode
        for episode in build_stateful_stress_episode_records(actor_models=(args.actor_model,), systems=systems)
        if (not task_ids or episode.task_id in task_ids)
        and (not horizons or episode.horizon in horizons)
    ]
    if args.limit is not None:
        episodes = episodes[: args.limit]

    if args.dry_run:
        print(f"stress_episodes={len(episodes)}")
        for episode in episodes:
            print(episode.episode_id)
        return

    assets = _write_assets(args.out_dir, episodes)
    if args.prepare_only:
        print(f"stress_episodes={len(episodes)}")
        print(f"stress_assets_dir={args.out_dir}")
        return

    _use_stress_retrieval_assets(assets["corpus_path"], assets["allowlist_path"])
    summaries = _load_existing_summary(assets["run_summary_path"])
    summary_by_episode_id = {item["episode_id"]: item for item in summaries}

    for episode in episodes:
        output_trace_path = trace_path(assets["traces_dir"], episode.episode_id)
        existing = summary_by_episode_id.get(episode.episode_id)
        if existing and _summary_trace_reusable(existing, output_trace_path, episode, args) and not args.force:
            continue
        if existing:
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
            writer_backend=args.writer_backend,
            planner_backend=args.planner_backend,
        )
        trace_path_for_summary = save_trace(assets["traces_dir"], episode.episode_id, trace)
        summaries.append(_summary_row(episode, trace_path_for_summary, len(trace), args))
        _write_json(assets["run_summary_path"], summaries)

    episode_scores = score_run_summary_items(summaries)
    metrics = aggregate_metrics(episode_scores)
    _write_json(assets["episode_scores_path"], episode_scores)
    _write_json(assets["metrics_path"], metrics)
    print(f"stress_episodes={len(episodes)}")
    print(f"stress_completed={len(summaries)}")
    print(f"metrics_path={assets['metrics_path']}")


def _write_assets(out_dir: Path, episodes: list) -> dict[str, Path]:
    corpus = build_corpus(include_stateful_stress_payloads=True)
    corpus_path = out_dir / "corpus" / "passages.jsonl"
    allowlist_path = out_dir / "corpus" / "allowlist.json"
    episodes_path = out_dir / "episodes.json"
    run_summary_path = out_dir / "run_summary.json"
    traces_dir = out_dir / "traces"
    save_jsonl(corpus_path, corpus)
    save_json(allowlist_path, build_allowlist(corpus))
    save_episodes(episodes_path, episodes)
    return {
        "corpus_path": corpus_path,
        "allowlist_path": allowlist_path,
        "episodes_path": episodes_path,
        "traces_dir": traces_dir,
        "run_summary_path": run_summary_path,
        "episode_scores_path": out_dir / "episode_scores.json",
        "metrics_path": out_dir / "metrics.json",
    }


def _use_stress_retrieval_assets(corpus_path: Path, allowlist_path: Path) -> None:
    pipeline_module.PASSAGES_PATH = corpus_path
    pipeline_module.ALLOWLIST_PATH = allowlist_path
    pipeline_module._load_allowlisted_source_ids.cache_clear()


def _summary_trace_reusable(item: dict, trace_file: Path, episode, args) -> bool:
    if not trace_file.exists():
        return False
    return (
        item.get("actor_model") == episode.actor_model
        and item.get("memory_writer_backend") == args.writer_backend
        and item.get("planner_backend") == args.planner_backend
        and item.get("planner_prompt_path") == str(PLANNER_PROMPT_PATH)
        and item.get("protocol_version") == PROTOCOL_VERSION
    )


def _summary_row(episode, trace_path_for_summary: Path, turn_count: int, args) -> dict:
    return {
        "episode_id": episode.episode_id,
        "system": episode.system,
        "actor_model": episode.actor_model,
        "episode_kind": episode.episode_kind,
        "payload_type": episode.payload_type,
        "family": episode.family,
        "task_id": episode.task_id,
        "horizon": episode.horizon,
        "trace_path": str(trace_path_for_summary),
        "turn_count": turn_count,
        "memory_writer_backend": args.writer_backend,
        "planner_backend": args.planner_backend,
        "planner_prompt_path": str(PLANNER_PROMPT_PATH),
        "protocol_version": PROTOCOL_VERSION,
        "extension": "stateful_stress",
    }


def _load_existing_summary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])
