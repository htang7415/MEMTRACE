import tempfile
from pathlib import Path

from memtrace.benchmark import POLICY_TASKS
from memtrace.pipeline import run_turn
from memtrace.store.db import connect, init_db
from memtrace.store.memory import insert_memory_records
from memtrace.store.systems import s1_filter
from memtrace.schema import MemoryCandidate
from memtrace.agents.runner import run_episode


def test_run_turn_contains_query() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        connection = connect(Path(temp_dir) / "memtrace.sqlite3")
        init_db(connection)
        trace_turn, accepted = run_turn("ep1", 1, "hello", "S0", connection, True, "clean_control", "clean_control", 1)
        assert trace_turn.query == "hello"
        assert accepted == []


def test_run_turn_reads_only_prior_memory() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        connection = connect(Path(temp_dir) / "memtrace.sqlite3")
        init_db(connection)
        records = s1_filter(
            [
                MemoryCandidate(
                    memory_type="policy_rule",
                    content="persisted memory",
                    source_id="P001",
                    source_kind="retrieval",
                )
            ],
            write_turn=1,
        )
        insert_memory_records(connection, episode_id="ep1", records=records)
        trace_turn, _ = run_turn("ep1", 2, "hello again", "S1", connection, True, "clean_control", "clean_control", 1)
        assert len(trace_turn.memory_store_state) == 1


def test_s1_poisoned_memory_can_flip_later_label() -> None:
    task = POLICY_TASKS[0]
    with tempfile.TemporaryDirectory() as temp_dir:
        trace = run_episode(
            episode_id="ep:stateful:d1:direct_override",
            turns=[task["poison_query"], task["trigger_query"]],
            system="S1",
            db_path=Path(temp_dir) / "memtrace.sqlite3",
            episode_kind="stateful_attack",
            episode_payload_type="direct_override",
            episode_horizon=1,
        )
        assert trace[0]["poison_admission_flag"] is True
        assert trace[1]["label"] == "unsafe"


def test_s2_blocks_poisoned_memory_from_flipping_later_label() -> None:
    task = POLICY_TASKS[0]
    with tempfile.TemporaryDirectory() as temp_dir:
        trace = run_episode(
            episode_id="ep:stateful:d1:direct_override",
            turns=[task["poison_query"], task["trigger_query"]],
            system="S2",
            db_path=Path(temp_dir) / "memtrace.sqlite3",
            episode_kind="stateful_attack",
            episode_payload_type="direct_override",
            episode_horizon=1,
        )
        assert trace[0]["poison_admission_flag"] is False
        assert trace[1]["label"] == "safe"


def test_one_shot_episode_labels_current_turn_injection() -> None:
    task = POLICY_TASKS[0]
    with tempfile.TemporaryDirectory() as temp_dir:
        trace = run_episode(
            episode_id="ep:one-shot:direct_override",
            turns=[task["poison_query"]],
            system="S0",
            db_path=Path(temp_dir) / "memtrace.sqlite3",
            episode_kind="one_shot_attack",
            episode_payload_type="direct_override",
            episode_horizon=1,
        )
        assert trace[0]["label"] == "unsafe"


def test_one_shot_contextual_drift_does_not_force_same_turn_override() -> None:
    task = POLICY_TASKS[0]
    with tempfile.TemporaryDirectory() as temp_dir:
        trace = run_episode(
            episode_id="ep:one-shot:contextual_drift",
            turns=[task["poison_query"]],
            system="S0",
            db_path=Path(temp_dir) / "memtrace.sqlite3",
            episode_kind="one_shot_attack",
            episode_payload_type="contextual_drift",
            episode_horizon=1,
        )
        assert trace[0]["label"] == "safe"
