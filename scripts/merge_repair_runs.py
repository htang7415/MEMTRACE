import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from memtrace.eval.metrics import aggregate_metrics
from memtrace.eval.scoring import score_run_summary_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge pilot repair summaries into a base run summary.")
    parser.add_argument("--base-summary", type=Path, required=True)
    parser.add_argument("--repair-summary", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--actor-model")
    args = parser.parse_args()

    base_items = _load_json(args.base_summary)
    if args.actor_model:
        base_items = [item for item in base_items if item.get("actor_model") == args.actor_model]
    merged_by_id = {item["episode_id"]: item for item in base_items}
    for path in args.repair_summary:
        for item in _load_json(path):
            if args.actor_model and item.get("actor_model") != args.actor_model:
                continue
            merged_by_id[item["episode_id"]] = item

    run_summary = sorted(merged_by_id.values(), key=lambda item: item["episode_id"])
    episode_scores = score_run_summary_items(run_summary)
    metrics = aggregate_metrics(episode_scores)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "run_summary.json", run_summary)
    _write_json(args.out_dir / "episode_scores.json", episode_scores)
    _write_json(args.out_dir / "metrics.json", metrics)
    print(f"merged_episodes={len(run_summary)}")
    print(f"metrics_path={args.out_dir / 'metrics.json'}")


def _load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
