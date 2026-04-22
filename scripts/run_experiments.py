import _bootstrap  # noqa: F401

import json

from memtrace.agents.runner import run_episode, save_trace
from memtrace.config import EPISODES_PATH, RUN_SUMMARY_PATH, SQLITE_PATH, TRACES_DIR
from memtrace.episodes import load_episodes


def main() -> None:
    episodes = load_episodes(EPISODES_PATH)
    summaries = []
    for episode in episodes:
        trace = run_episode(
            episode_id=episode.episode_id,
            turns=episode.turns,
            system=episode.system,
            db_path=SQLITE_PATH,
        )
        trace_path = save_trace(TRACES_DIR, episode.episode_id, trace)
        summaries.append(
            {
                "episode_id": episode.episode_id,
                "system": episode.system,
                "episode_kind": episode.episode_kind,
                "payload_type": episode.payload_type,
                "trace_path": str(trace_path),
                "turn_count": len(trace),
            }
        )
    RUN_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUN_SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    print(f"episodes_run={len(summaries)}")
    print(f"summary_path={RUN_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
