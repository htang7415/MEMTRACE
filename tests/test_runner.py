import json
import tempfile
from pathlib import Path

from memtrace.agents.runner import run_episode, save_trace


def test_run_episode_returns_one_trace_row_per_turn() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        trace = run_episode(
            episode_id="ep1",
            turns=["q1", "q2"],
            system="S1",
            db_path=Path(temp_dir) / "memtrace.sqlite3",
        )
        assert len(trace) == 2


def test_save_trace_writes_jsonl() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = save_trace(Path(temp_dir), "ep:1", [{"episode_id": "ep:1", "turn": 1}])
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert json.loads(lines[0])["episode_id"] == "ep:1"
