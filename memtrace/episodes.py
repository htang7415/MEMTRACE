"""Episode generation helpers."""

from __future__ import annotations

import json
from pathlib import Path

from memtrace.schema import EpisodeRecord


def save_episodes(path: Path, episodes: list[EpisodeRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        serialized = []
        for episode in episodes:
            if hasattr(episode, "model_dump"):
                serialized.append(episode.model_dump())
            else:
                serialized.append(episode.dict())
        json.dump(serialized, handle, indent=2)


def load_episodes(path: Path) -> list[EpisodeRecord]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return [EpisodeRecord(**item) for item in raw]
