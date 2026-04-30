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
    assert "planner_prompt_path" in summary[0]
    assert (tmp_path / "pilot" / "metrics.json").exists()


def test_run_pilot_reruns_when_prompt_path_changes(tmp_path, monkeypatch) -> None:
    episode = EpisodeRecord(
        episode_id="model-a:S0:budget-limit-rule:clean",
        task_id="budget-limit-rule",
        family="policy_memory",
        episode_kind="clean_control",
        payload_type="clean_control",
        system="S0",
        actor_model="model-a",
        horizon=1,
        turns=["q1"],
    )
    out_dir = tmp_path / "pilot"
    trace_dir = out_dir / "traces"
    trace_dir.mkdir(parents=True)
    trace_file = trace_dir / "model-a__S0__budget-limit-rule__clean.jsonl"
    trace_file.write_text(json.dumps({"episode_id": episode.episode_id, "turn": 1, "stale": True}) + "\n", encoding="utf-8")
    (out_dir / "run_summary.json").write_text(
        json.dumps(
            [
                {
                    "episode_id": episode.episode_id,
                    "system": "S0",
                    "actor_model": "model-a",
                    "episode_kind": "clean_control",
                    "payload_type": "clean_control",
                    "family": "policy_memory",
                    "task_id": "budget-limit-rule",
                    "horizon": 1,
                    "trace_path": str(trace_file),
                    "turn_count": 1,
                    "memory_writer_backend": "mlx",
                    "planner_backend": "mlx",
                    "planner_prompt_path": "old_prompt.txt",
                    "protocol_version": run_pilot_module.PROTOCOL_VERSION,
                }
            ]
        ),
        encoding="utf-8",
    )
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
    monkeypatch.setattr(run_pilot_module, "PLANNER_PROMPT_PATH", tmp_path / "strict_prompt.txt")
    monkeypatch.setattr(run_pilot_module, "build_episode_records", lambda: [episode])
    monkeypatch.setattr(run_pilot_module, "run_episode", fake_run_episode)

    run_pilot_module.main(
        [
            "--actor-model",
            "model-a",
            "--task-id",
            "budget-limit-rule",
            "--system",
            "S0",
            "--episode-kind",
            "clean_control",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert calls == [episode.episode_id]
    summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary[0]["planner_prompt_path"] == str(tmp_path / "strict_prompt.txt")


def test_run_pilot_dry_run_filters_payload_and_horizon(monkeypatch, capsys) -> None:
    episodes = [
        EpisodeRecord(
            episode_id="model-a:S1:budget-limit-rule:stateful:d1:direct_override",
            task_id="budget-limit-rule",
            family="policy_memory",
            episode_kind="stateful_attack",
            payload_type="direct_override",
            system="S1",
            actor_model="model-a",
            horizon=1,
            turns=["q1", "q2"],
        ),
        EpisodeRecord(
            episode_id="model-a:S1:budget-limit-rule:stateful:d3:direct_override",
            task_id="budget-limit-rule",
            family="policy_memory",
            episode_kind="stateful_attack",
            payload_type="direct_override",
            system="S1",
            actor_model="model-a",
            horizon=3,
            turns=["q1", "q2"],
        ),
        EpisodeRecord(
            episode_id="model-a:S1:budget-limit-rule:stateful:d1:contextual_drift",
            task_id="budget-limit-rule",
            family="policy_memory",
            episode_kind="stateful_attack",
            payload_type="contextual_drift",
            system="S1",
            actor_model="model-a",
            horizon=1,
            turns=["q1", "q2"],
        ),
    ]

    def fail_run_episode(**kwargs):
        raise AssertionError("dry run should not execute episodes")

    monkeypatch.setattr(run_pilot_module, "build_episode_records", lambda: episodes)
    monkeypatch.setattr(run_pilot_module, "run_episode", fail_run_episode)

    run_pilot_module.main(
        [
            "--actor-model",
            "model-a",
            "--task-id",
            "budget-limit-rule",
            "--system",
            "S1",
            "--episode-kind",
            "stateful_attack",
            "--payload-type",
            "direct_override",
            "--horizon",
            "1",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert "pilot_episodes=1" in output
    assert "model-a:S1:budget-limit-rule:stateful:d1:direct_override" in output
    assert "contextual_drift" not in output


def test_run_pilot_can_clone_grid_for_explicit_unconfigured_actor(monkeypatch, capsys) -> None:
    episodes = [
        EpisodeRecord(
            episode_id="model_a:S0:budget-limit-rule:clean",
            task_id="budget-limit-rule",
            family="policy_memory",
            episode_kind="clean_control",
            payload_type="clean_control",
            system="S0",
            actor_model="model-a",
            horizon=1,
            turns=["q1"],
        )
    ]

    def fail_run_episode(**kwargs):
        raise AssertionError("dry run should not execute episodes")

    monkeypatch.setattr(run_pilot_module, "ACTOR_MODELS", ("model-a",))
    monkeypatch.setattr(run_pilot_module, "build_episode_records", lambda: episodes)
    monkeypatch.setattr(run_pilot_module, "run_episode", fail_run_episode)

    run_pilot_module.main(
        [
            "--actor-model",
            "model-b",
            "--task-id",
            "budget-limit-rule",
            "--system",
            "S0",
            "--episode-kind",
            "clean_control",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert "pilot_episodes=1" in output
    assert "model_b:S0:budget-limit-rule:clean" in output
