import tempfile
from pathlib import Path

from memtrace.schema import MemoryCandidate
from memtrace.store.db import connect, init_db
from memtrace.store.memory import insert_memory_records, load_memory_records
from memtrace.store.systems import s0_filter, s1_filter, s2_filter


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
                )
            ],
            write_turn=2,
        )
        insert_memory_records(connection, episode_id="ep1", records=records)
        assert load_memory_records(connection, episode_id="ep1", max_turn=2) == []
        loaded = load_memory_records(connection, episode_id="ep1", max_turn=3)
        assert len(loaded) == 1
