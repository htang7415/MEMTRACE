import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import scripts.run_pilot as run_pilot_module
from memtrace.schema import EpisodeRecord


def test_run_pilot_writes_subset_outputs(tmp_path, monkeypatch) -> None:
    episodes = [
        EpisodeRecord(
            episode_id="model-a:S0:budget-limit-rule:clean",
            task_id="budget-limit-rule",
            family="policy_memory",
            episode_kind="clean_control",
            payload_type="clean_control",
            system="S0",
            actor_model="model-a",
            horizon=1,
            turns=["q1"],
        ),
        EpisodeRecord(
            episode_id="model-a:S0:meeting-time:clean",
            task_id="meeting-time",
            family="tool_argument_memory",
            episode_kind="clean_control",
            payload_type="clean_control",
            system="S0",
            actor_model="model-a",
            horizon=1,
            turns=["q2"],
        ),
        EpisodeRecord(
            episode_id="model-a:S0:expense-code:clean",
            task_id="expense-code",
            family="tool_argument_memory",
            episode_kind="clean_control",
            payload_type="clean_control",
            system="S0",
            actor_model="model-a",
            horizon=1,
            turns=["q3"],
        ),
    ]
    calls = []

    def fake_run_episode(**kwargs):
        calls.append(kwargs["episode_id"])
        return [
            {
                "episode_id": kwargs["episode_id"],
                "turn": 1,
                "query": "q",
                "label": "safe",
                "poison_admission_flag": None,
                "memory_writer_output": [],
                "admitted_memory_records": [],
                "retrieved_passages": [],
                "tool_router_log": None,
            }
        ]

    monkeypatch.setattr(run_pilot_module, "ACTOR_MODELS", ("model-a",))
    monkeypatch.setattr(run_pilot_module, "SQLITE_PATH", tmp_path / "memtrace.sqlite3")
    monkeypatch.setattr(run_pilot_module, "build_episode_records", lambda: episodes)
    monkeypatch.setattr(run_pilot_module, "run_episode", fake_run_episode)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pilot.py",
            "--actor-model",
            "model-a",
            "--task-id",
            "budget-limit-rule",
            "--system",
            "S0",
            "--episode-kind",
            "clean_control",
            "--out-dir",
            str(tmp_path / "pilot"),
        ],
    )

    run_pilot_module.main()

    assert calls == ["model-a:S0:budget-limit-rule:clean"]
    summary = json.loads((tmp_path / "pilot" / "run_summary.json").read_text(encoding="utf-8"))
    assert len(summary) == 1
    assert (tmp_path / "pilot" / "metrics.json").exists()
