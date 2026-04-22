import json

import _bootstrap  # noqa: F401

from memtrace.benchmark import build_episode_records, build_gold_labels, build_task_records
from memtrace.config import EPISODES_PATH, LABELS_PATH, TASKS_PATH
from memtrace.episodes import save_episodes


def main() -> None:
    tasks = build_task_records()
    labels = build_gold_labels()
    episodes = build_episode_records()
    TASKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TASKS_PATH.open("w", encoding="utf-8") as handle:
        json.dump([task.model_dump() if hasattr(task, "model_dump") else task.dict() for task in tasks], handle, indent=2)
    with LABELS_PATH.open("w", encoding="utf-8") as handle:
        json.dump([label.model_dump() if hasattr(label, "model_dump") else label.dict() for label in labels], handle, indent=2)
    save_episodes(EPISODES_PATH, episodes)
    print(f"wrote_tasks={len(tasks)}")
    print(f"wrote_labels={len(labels)}")
    print(f"wrote_episodes={len(episodes)}")


if __name__ == "__main__":
    main()
