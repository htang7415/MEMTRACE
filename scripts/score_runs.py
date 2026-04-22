import _bootstrap  # noqa: F401

import json

from memtrace.config import EPISODE_SCORES_PATH, METRICS_PATH, RUN_SUMMARY_PATH
from memtrace.eval.metrics import aggregate_metrics
from memtrace.eval.scoring import load_json, score_run_summary_items


def main() -> None:
    run_summary = load_json(RUN_SUMMARY_PATH)
    episode_scores = score_run_summary_items(run_summary)
    metrics = aggregate_metrics(episode_scores)

    EPISODE_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EPISODE_SCORES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(episode_scores, handle, indent=2)
    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"episode_scores={len(episode_scores)}")
    print(f"metrics_path={METRICS_PATH}")


if __name__ == "__main__":
    main()
