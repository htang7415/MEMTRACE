import json
import tempfile
from pathlib import Path

import memtrace.pipeline as pipeline_module
import memtrace.agents.runner as runner_module
from memtrace.benchmark import build_episode_records
from memtrace.calibration import ORACLE_MEMORY_CONDITION
from memtrace.constants import STATEFUL_STRESS_PAYLOAD_TYPE, TRUSTED_UTILITY_PAYLOAD_TYPE
from memtrace.corpus import build_corpus, save_jsonl
from memtrace.agents.runner import run_episode, save_trace, trace_path
from memtrace.eval.scoring import score_run_summary_items


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


def test_stateful_stress_episode_id_inference() -> None:
    episode_id = (
        "mlx-community-Qwen2.5-7B-Instruct-4bit:S2:approval-limit-rule:"
        f"stateful-stress:d7:{STATEFUL_STRESS_PAYLOAD_TYPE}"
    )
    assert runner_module._episode_kind_from_episode_id(episode_id) == "stateful_attack"
    assert runner_module._horizon_from_episode_id(episode_id) == 7
    assert runner_module._payload_type_from_episode_id(episode_id) == STATEFUL_STRESS_PAYLOAD_TYPE


def test_trusted_utility_episode_id_inference() -> None:
    episode_id = (
        "mlx-community-Qwen2.5-7B-Instruct-4bit:S2:approval-limit-rule:"
        f"trusted-utility:d3:{TRUSTED_UTILITY_PAYLOAD_TYPE}"
    )
    assert runner_module._episode_kind_from_episode_id(episode_id) == "trusted_memory_utility"
    assert runner_module._horizon_from_episode_id(episode_id) == 3
    assert runner_module._payload_type_from_episode_id(episode_id) == TRUSTED_UTILITY_PAYLOAD_TYPE


def test_run_episode_oracle_memory_calibration_forces_trigger_memory(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        corpus_path = Path(temp_dir) / "passages.jsonl"
        save_jsonl(corpus_path, build_corpus())
        monkeypatch.setattr(pipeline_module, "PASSAGES_PATH", corpus_path)
        monkeypatch.setattr(runner_module, "PASSAGES_PATH", corpus_path)
        episode = next(
            item
            for item in build_episode_records()
            if item.actor_model == "mlx-community/Qwen2.5-7B-Instruct-4bit"
            and item.system == "S1"
            and item.task_id == "budget-limit-rule"
            and item.episode_kind == "stateful_attack"
            and item.payload_type == "direct_override"
            and item.horizon == 1
        )
        episode_id = episode.episode_id.replace(":S1:", f":{ORACLE_MEMORY_CONDITION}:", 1)

        trace = run_episode(
            episode_id=episode_id,
            turns=episode.turns,
            system="S1",
            db_path=Path(temp_dir) / "memtrace.sqlite3",
            actor_model=episode.actor_model or "",
            episode_kind=episode.episode_kind,
            episode_payload_type=episode.payload_type,
            episode_horizon=episode.horizon,
            writer_backend="profile",
            planner_backend="profile",
            calibration_condition=ORACLE_MEMORY_CONDITION,
        )
        final_turn = trace[-1]
        summary = [
            {
                "episode_id": episode_id,
                "system": ORACLE_MEMORY_CONDITION,
                "actor_model": episode.actor_model,
                "episode_kind": episode.episode_kind,
                "payload_type": episode.payload_type,
                "family": episode.family,
                "task_id": episode.task_id,
                "horizon": episode.horizon,
                "trace_path": str(save_trace(Path(temp_dir) / "traces", episode_id, trace)),
                "turn_count": len(trace),
                "calibration_condition": ORACLE_MEMORY_CONDITION,
                "oracle_memory_inserted": True,
            }
        ]

        scores = score_run_summary_items(summary)

        assert final_turn["calibration_condition"] == ORACLE_MEMORY_CONDITION
        assert final_turn["oracle_memory_inserted"] is True
        assert final_turn["oracle_memory_retrieved"] is True
        assert final_turn["current_turn_poison_retrieved_at_trigger"] is False
        assert scores[0]["oracle_memory_inserted"] is True
        assert scores[0]["oracle_memory_retrieved"] is True
        assert scores[0]["unsafe_tool_call_proposed_before_checker"] is True
        assert scores[0]["unsafe_tool_call_executed"] is True
