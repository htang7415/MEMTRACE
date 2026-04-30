import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import scripts.run_experiments as run_experiments_module
from memtrace.episodes import save_episodes
from memtrace.schema import EpisodeRecord


def test_main_reruns_when_summary_protocol_is_stale(tmp_path, monkeypatch) -> None:
    episode = EpisodeRecord(
        episode_id="ep1",
        task_id="task-a",
        family="policy_memory",
        episode_kind="clean_control",
        payload_type="clean_control",
        system="S0",
        actor_model="model-a",
        horizon=1,
        turns=["q1"],
    )
    episodes_path = tmp_path / "episodes.json"
    save_episodes(episodes_path, [episode])

    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_path = trace_dir / "ep1.jsonl"
    trace_path.write_text(json.dumps({"episode_id": "ep1", "turn": 1, "stale": True}) + "\n", encoding="utf-8")

    run_summary_path = tmp_path / "run_summary.json"
    run_summary_path.write_text(
        json.dumps(
            [
                {
                    "episode_id": "ep1",
                    "system": "S0",
                    "actor_model": "model-a",
                    "episode_kind": "clean_control",
                    "payload_type": "clean_control",
                    "family": "policy_memory",
                    "task_id": "task-a",
                    "horizon": 1,
                    "trace_path": str(trace_path),
                    "turn_count": 1,
                    "memory_writer_backend": "mlx",
                    "planner_backend": "mlx",
                    "planner_prompt_path": "old_prompt.txt",
                    "protocol_version": "project-md-v1",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    calls = []

    def fake_run_episode(**kwargs):
        calls.append(kwargs["episode_id"])
        return [{"episode_id": kwargs["episode_id"], "turn": 1, "fresh": True}]

    monkeypatch.setattr(run_experiments_module, "EPISODES_PATH", episodes_path)
    monkeypatch.setattr(run_experiments_module, "RUN_SUMMARY_PATH", run_summary_path)
    monkeypatch.setattr(run_experiments_module, "TRACES_DIR", trace_dir)
    monkeypatch.setattr(run_experiments_module, "SQLITE_PATH", tmp_path / "memtrace.sqlite3")
    monkeypatch.setattr(run_experiments_module, "PROTOCOL_VERSION", "project-md-v2")
    monkeypatch.setattr(run_experiments_module, "MEMORY_WRITER_BACKEND", "mlx")
    monkeypatch.setattr(run_experiments_module, "PLANNER_BACKEND", "mlx")
    monkeypatch.setattr(run_experiments_module, "PLANNER_PROMPT_PATH", tmp_path / "planner.txt")
    monkeypatch.setattr(run_experiments_module, "run_episode", fake_run_episode)

    run_experiments_module.main([])

    assert calls == ["ep1"]
    trace_rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert trace_rows == [{"episode_id": "ep1", "turn": 1, "fresh": True}]
    summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    assert summary[0]["protocol_version"] == "project-md-v2"
    assert summary[0]["planner_prompt_path"] == str(tmp_path / "planner.txt")


def test_main_reruns_when_trace_exists_without_summary(tmp_path, monkeypatch) -> None:
    episode = EpisodeRecord(
        episode_id="ep2",
        task_id="task-b",
        family="policy_memory",
        episode_kind="clean_control",
        payload_type="clean_control",
        system="S0",
        actor_model="model-b",
        horizon=1,
        turns=["q1"],
    )
    episodes_path = tmp_path / "episodes.json"
    save_episodes(episodes_path, [episode])

    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_path = trace_dir / "ep2.jsonl"
    trace_path.write_text(json.dumps({"episode_id": "ep2", "turn": 1, "stale": True}) + "\n", encoding="utf-8")

    run_summary_path = tmp_path / "run_summary.json"

    calls = []

    def fake_run_episode(**kwargs):
        calls.append(kwargs["episode_id"])
        return [{"episode_id": kwargs["episode_id"], "turn": 1, "fresh": True}]

    monkeypatch.setattr(run_experiments_module, "EPISODES_PATH", episodes_path)
    monkeypatch.setattr(run_experiments_module, "RUN_SUMMARY_PATH", run_summary_path)
    monkeypatch.setattr(run_experiments_module, "TRACES_DIR", trace_dir)
    monkeypatch.setattr(run_experiments_module, "SQLITE_PATH", tmp_path / "memtrace.sqlite3")
    monkeypatch.setattr(run_experiments_module, "MEMORY_WRITER_BACKEND", "mlx")
    monkeypatch.setattr(run_experiments_module, "PLANNER_BACKEND", "mlx")
    monkeypatch.setattr(run_experiments_module, "run_episode", fake_run_episode)

    run_experiments_module.main([])

    assert calls == ["ep2"]
    trace_rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert trace_rows == [{"episode_id": "ep2", "turn": 1, "fresh": True}]


def test_main_help_exits_before_backend_check(monkeypatch, capsys) -> None:
    def fail_backend_check():
        raise AssertionError("backend check should not run for --help")

    monkeypatch.setattr(run_experiments_module, "_require_official_backends", fail_backend_check)

    with pytest.raises(SystemExit) as excinfo:
        run_experiments_module.main(["--help"])

    assert excinfo.value.code == 0
    assert "Run official MEMTRACE experiments." in capsys.readouterr().out
