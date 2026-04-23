import json
import tempfile
from pathlib import Path

import memtrace.pipeline as pipeline_module
from memtrace.corpus import build_corpus, save_jsonl
from memtrace.agents.runner import run_episode, save_trace, trace_path


def test_run_episode_returns_one_trace_row_per_turn(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "passages.jsonl"
        save_jsonl(path, build_corpus())
        monkeypatch.setattr(pipeline_module, "PASSAGES_PATH", path)
        trace = run_episode(
            episode_id="ep1",
            turns=["q1", "q2"],
            system="S1",
            db_path=Path(temp_dir) / "memtrace.sqlite3",
            writer_backend="profile",
            planner_backend="profile",
        )
        assert len(trace) == 2


def test_save_trace_writes_jsonl() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = save_trace(Path(temp_dir), "ep:1", [{"episode_id": "ep:1", "turn": 1}])
        assert path == trace_path(Path(temp_dir), "ep:1")
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert json.loads(lines[0])["episode_id"] == "ep:1"
