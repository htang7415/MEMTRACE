import tempfile
from pathlib import Path

import pytest

from memtrace.core.schema import MemoryCandidate, MemoryRecord
from memtrace.backends.store.db import connect, init_db
from memtrace.backends.store.memory import insert_memory_records, load_memory_records, resolve_visible_memory
from memtrace.backends.store.systems import s0_filter, s1_filter, s2_filter, s2_filter_with_rejections


def _record(task_id: str, memory_type: str, content: str, source_id: str, write_turn: int) -> MemoryRecord:
    return MemoryRecord(
        memory_type=memory_type,
        content=content,
        source_id=source_id,
        source_kind="retrieval",
        task_id=task_id,
        trust_level="low",
        write_turn=write_turn,
        validation_status="accepted",
    )


def test_s0_rejects_all() -> None:
    candidate = MemoryCandidate(
        memory_type="policy_rule",
        content="x",
        source_id="P001",
        source_kind="retrieval",
    )
    assert s0_filter([candidate], write_turn=1) == []
    assert len(s1_filter([candidate], write_turn=1)) == 1
    assert len(s2_filter([candidate], write_turn=1, allowlisted_source_ids={"P001"})) == 1


def test_s2_rejects_unallowlisted_retrieval() -> None:
    candidate = MemoryCandidate(
        memory_type="policy_rule",
        content="x",
        source_id="P999",
        source_kind="retrieval",
    )
    assert s2_filter([candidate], write_turn=1, allowlisted_source_ids={"P001"}) == []
    accepted, rejected = s2_filter_with_rejections([candidate], write_turn=1, allowlisted_source_ids={"P001"})
    assert accepted == []
    assert len(rejected) == 1
    assert rejected[0].validation_status == "rejected_provenance"


def test_memory_store_reads_only_prior_turns() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        connection = connect(Path(temp_dir) / "memtrace.sqlite3")
        init_db(connection)
        records = s1_filter(
            [
                MemoryCandidate(
                    memory_type="policy_rule",
                    content="remember this",
                    source_id="P001",
                    source_kind="retrieval",
                    task_id="task-a",
                )
            ],
            write_turn=2,
        )
        insert_memory_records(connection, episode_id="ep1", records=records)
        assert load_memory_records(connection, episode_id="ep1", max_turn=2) == []
        loaded = load_memory_records(connection, episode_id="ep1", max_turn=3)
        assert len(loaded) == 1
        assert loaded[0].task_id == "task-a"


def test_memory_store_can_filter_by_task_id() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        connection = connect(Path(temp_dir) / "memtrace.sqlite3")
        init_db(connection)
        records = s1_filter(
            [
                MemoryCandidate(
                    memory_type="policy_rule",
                    content="task a memory",
                    source_id="P001",
                    source_kind="retrieval",
                    task_id="task-a",
                ),
                MemoryCandidate(
                    memory_type="policy_rule",
                    content="task b memory",
                    source_id="P002",
                    source_kind="retrieval",
                    task_id="task-b",
                ),
            ],
            write_turn=1,
        )
        insert_memory_records(connection, episode_id="ep1", records=records)
        loaded = load_memory_records(connection, episode_id="ep1", max_turn=2, task_id="task-a")
        assert [record.content for record in loaded] == ["task a memory"]


def test_resolve_visible_memory_default_mode_is_a_no_op() -> None:
    records = [
        _record("task-a", "policy_rule", "first", "P001", write_turn=1),
        _record("task-a", "policy_rule", "second", "P002", write_turn=3),
    ]
    assert resolve_visible_memory(records, current_turn=5) == records


def test_resolve_visible_memory_latest_wins_keeps_newest_per_slot() -> None:
    records = [
        _record("task-a", "policy_rule", "legit limit", "P001", write_turn=1),
        _record("task-a", "policy_rule", "poisoned override", "P002", write_turn=4),
        _record("task-b", "tool_argument", "other slot", "P003", write_turn=2),
    ]
    visible = resolve_visible_memory(records, current_turn=5, conflict_resolution="latest_wins_per_task_and_type")
    assert [record.content for record in visible] == ["other slot", "poisoned override"]


def test_resolve_visible_memory_latest_wins_preserves_distinct_slots() -> None:
    records = [
        _record("task-a", "policy_rule", "policy fact", "P001", write_turn=1),
        _record("task-a", "tool_argument", "argument fact", "P002", write_turn=2),
    ]
    visible = resolve_visible_memory(records, current_turn=5, conflict_resolution="latest_wins_per_task_and_type")
    assert {record.content for record in visible} == {"policy fact", "argument fact"}


def test_resolve_visible_memory_ttl_drops_records_older_than_window() -> None:
    records = [
        _record("task-a", "policy_rule", "stale", "P001", write_turn=1),
        _record("task-a", "policy_rule", "fresh", "P002", write_turn=4),
    ]
    visible = resolve_visible_memory(records, current_turn=5, ttl_turns=1)
    assert [record.content for record in visible] == ["fresh"]


def test_resolve_visible_memory_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown memory conflict resolution mode"):
        resolve_visible_memory([], current_turn=1, conflict_resolution="bogus")
