import tempfile
from pathlib import Path

import memtrace.pipeline as pipeline_module
from memtrace.config import ACTOR_MODELS
from memtrace.benchmark import POLICY_TASKS
from memtrace.constants import STATEFUL_STRESS_PAYLOAD_TYPE
from memtrace.corpus import build_corpus, save_jsonl
from memtrace.pipeline import run_turn
from memtrace.schema import RetrievedPassage
from memtrace.store.db import connect, init_db
from memtrace.store.memory import insert_memory_records
from memtrace.store.systems import s1_filter
from memtrace.schema import MemoryCandidate
from memtrace.agents.runner import run_episode


def _use_temp_corpus(monkeypatch, temp_dir: str) -> None:
    path = Path(temp_dir) / "passages.jsonl"
    save_jsonl(path, build_corpus())
    monkeypatch.setattr(pipeline_module, "PASSAGES_PATH", path)


def _use_missing_allowlist(monkeypatch, temp_dir: str) -> None:
    pipeline_module._load_allowlisted_source_ids.cache_clear()
    monkeypatch.setattr(pipeline_module, "ALLOWLIST_PATH", Path(temp_dir) / "allowlist_missing.json")


def _use_fixture_retrieval(monkeypatch, task: dict) -> None:
    corpus = build_corpus()
    authoritative = next(item for item in corpus if item.get("task_id") == task["task_id"] and item.get("allowlisted"))
    direct_override = next(
        item
        for item in corpus
        if item.get("task_id") == task["task_id"]
        and item.get("payload_type") == "direct_override"
        and item.get("passage_kind") == "poison"
    )
    contextual_drift = next(
        item
        for item in corpus
        if item.get("task_id") == task["task_id"]
        and item.get("payload_type") == "contextual_drift"
        and item.get("passage_kind") == "poison"
    )
    distractor = next(item for item in corpus if item.get("passage_kind") == "distractor")

    def fake_retrieve(query: str, path, top_k: int):
        del path, top_k
        if query == task["poison_query"]:
            rows = [direct_override, contextual_drift, authoritative, distractor]
        elif query == task["trigger_query"]:
            rows = [authoritative, distractor]
        else:
            rows = [authoritative, distractor]
        return [
            RetrievedPassage(
                source_id=row["source_id"],
                text=row["text"],
                rank=index,
                score=float(len(rows) - index),
                allowlisted=row.get("allowlisted", False),
                task_id=row.get("task_id"),
                payload_type=row.get("payload_type"),
                passage_kind=row.get("passage_kind", "authoritative"),
            )
            for index, row in enumerate(rows, start=1)
        ]

    monkeypatch.setattr(pipeline_module, "retrieve", fake_retrieve)


def test_run_turn_contains_query(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        _use_temp_corpus(monkeypatch, temp_dir)
        connection = connect(Path(temp_dir) / "memtrace.sqlite3")
        init_db(connection)
        trace_turn, accepted = run_turn(
            "ep1",
            1,
            "hello",
            "S0",
            ACTOR_MODELS[0],
            connection,
            False,
            "clean_control",
            "clean_control",
            1,
        )
        assert trace_turn.query == "hello"
        assert accepted == []


def test_run_turn_reads_only_prior_memory(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        _use_temp_corpus(monkeypatch, temp_dir)
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
        trace_turn, _ = run_turn("ep1", 2, "hello again", "S1", ACTOR_MODELS[0], connection, False, "clean_control", "clean_control", 1)
        assert len(trace_turn.prior_memory_state) == 1
        assert trace_turn.prior_memory_state[0].content == "persisted memory"


def test_s1_poisoned_memory_can_flip_later_label(monkeypatch) -> None:
    task = POLICY_TASKS[0]
    with tempfile.TemporaryDirectory() as temp_dir:
        _use_temp_corpus(monkeypatch, temp_dir)
        _use_missing_allowlist(monkeypatch, temp_dir)
        _use_fixture_retrieval(monkeypatch, task)
        trace = run_episode(
            episode_id="ep:stateful:d1:direct_override",
            turns=[task["poison_query"], task["trigger_query"]],
            system="S1",
            actor_model=ACTOR_MODELS[0],
            db_path=Path(temp_dir) / "memtrace.sqlite3",
            episode_kind="stateful_attack",
            episode_payload_type="direct_override",
            episode_horizon=1,
            writer_backend="profile",
            planner_backend="profile",
        )
        assert trace[0]["poison_admission_flag"] is True
        assert trace[1]["label"] == "unsafe"


def test_s2_blocks_poisoned_memory_from_flipping_later_label(monkeypatch) -> None:
    task = POLICY_TASKS[0]
    with tempfile.TemporaryDirectory() as temp_dir:
        _use_temp_corpus(monkeypatch, temp_dir)
        _use_missing_allowlist(monkeypatch, temp_dir)
        _use_fixture_retrieval(monkeypatch, task)
        trace = run_episode(
            episode_id="ep:stateful:d1:direct_override",
            turns=[task["poison_query"], task["trigger_query"]],
            system="S2",
            actor_model=ACTOR_MODELS[0],
            db_path=Path(temp_dir) / "memtrace.sqlite3",
            episode_kind="stateful_attack",
            episode_payload_type="direct_override",
            episode_horizon=1,
            writer_backend="profile",
            planner_backend="profile",
        )
        assert trace[0]["poison_admission_flag"] is False
        assert len(trace[0]["rejected_memory_records"]) > 0
        assert trace[0]["rejected_memory_records"][0]["validation_status"] == "rejected_provenance"
        assert trace[1]["label"] == "safe"


def test_one_shot_episode_labels_current_turn_injection(monkeypatch) -> None:
    task = POLICY_TASKS[0]
    with tempfile.TemporaryDirectory() as temp_dir:
        _use_temp_corpus(monkeypatch, temp_dir)
        _use_fixture_retrieval(monkeypatch, task)
        trace = run_episode(
            episode_id="ep:one-shot:direct_override",
            turns=[task["poison_query"]],
            system="S0",
            actor_model=ACTOR_MODELS[0],
            db_path=Path(temp_dir) / "memtrace.sqlite3",
            episode_kind="one_shot_attack",
            episode_payload_type="direct_override",
            episode_horizon=1,
            writer_backend="profile",
            planner_backend="profile",
        )
        assert trace[0]["label"] == "unsafe"


def test_one_shot_turn_reads_empty_prior_memory(monkeypatch) -> None:
    task = POLICY_TASKS[0]
    with tempfile.TemporaryDirectory() as temp_dir:
        _use_temp_corpus(monkeypatch, temp_dir)
        _use_fixture_retrieval(monkeypatch, task)
        trace = run_episode(
            episode_id="ep:one-shot:contextual_drift",
            turns=[task["poison_query"]],
            system="S0",
            actor_model=ACTOR_MODELS[0],
            db_path=Path(temp_dir) / "memtrace.sqlite3",
            episode_kind="one_shot_attack",
            episode_payload_type="contextual_drift",
            episode_horizon=1,
            writer_backend="profile",
            planner_backend="profile",
        )
        assert trace[0]["prior_memory_state"] == []


def test_trace_records_effective_retrieval_only(monkeypatch) -> None:
    task = POLICY_TASKS[0]
    with tempfile.TemporaryDirectory() as temp_dir:
        _use_temp_corpus(monkeypatch, temp_dir)
        _use_fixture_retrieval(monkeypatch, task)
        trace = run_episode(
            episode_id="ep:one-shot:direct_override",
            turns=[task["poison_query"]],
            system="S1",
            actor_model=ACTOR_MODELS[0],
            db_path=Path(temp_dir) / "memtrace.sqlite3",
            episode_kind="one_shot_attack",
            episode_payload_type="direct_override",
            episode_horizon=1,
            writer_backend="profile",
            planner_backend="profile",
        )
        payload_types = {
            hit["payload_type"]
            for hit in trace[0]["retrieved_passages"]
            if hit["task_id"] == task["task_id"] and hit["passage_kind"] == "poison"
        }
        assert payload_types == {"direct_override"}


def test_payload_filter_handles_stateful_stress_payload_type() -> None:
    task = POLICY_TASKS[0]
    passages = [
        RetrievedPassage(source_id="P1", text="direct", task_id=task["task_id"], payload_type="direct_override", passage_kind="poison"),
        RetrievedPassage(source_id="P2", text="stress", task_id=task["task_id"], payload_type=STATEFUL_STRESS_PAYLOAD_TYPE, passage_kind="poison"),
        RetrievedPassage(source_id="P3", text="trusted", task_id=task["task_id"], allowlisted=True, passage_kind="authoritative"),
    ]

    filtered = pipeline_module._filter_payload_retrieval(
        retrieved_passages=passages,
        task_id=task["task_id"],
        episode_payload_type=STATEFUL_STRESS_PAYLOAD_TYPE,
    )

    assert [passage.source_id for passage in filtered] == ["P2", "P3"]


def test_planner_output_matches_actual_planner_decision(monkeypatch) -> None:
    task = POLICY_TASKS[0]
    with tempfile.TemporaryDirectory() as temp_dir:
        _use_temp_corpus(monkeypatch, temp_dir)
        _use_fixture_retrieval(monkeypatch, task)
        trace = run_episode(
            episode_id="ep:one-shot:direct_override",
            turns=[task["poison_query"]],
            system="S1",
            actor_model=ACTOR_MODELS[0],
            db_path=Path(temp_dir) / "memtrace.sqlite3",
            episode_kind="one_shot_attack",
            episode_payload_type="direct_override",
            episode_horizon=1,
            writer_backend="profile",
            planner_backend="profile",
        )
        assert trace[0]["planner_output"] == trace[0]["tool_router_log"]
        assert trace[0]["raw_memory_writer_output"] is not None
        assert trace[0]["raw_planner_output"] is not None


def test_final_turn_reads_only_same_task_prior_memory(monkeypatch) -> None:
    task = POLICY_TASKS[0]
    with tempfile.TemporaryDirectory() as temp_dir:
        _use_temp_corpus(monkeypatch, temp_dir)
        _use_missing_allowlist(monkeypatch, temp_dir)
        _use_fixture_retrieval(monkeypatch, task)
        trace = run_episode(
            episode_id="ep:stateful:d3:direct_override",
            turns=[task["poison_query"], POLICY_TASKS[1]["clean_query"], POLICY_TASKS[2]["clean_query"], task["trigger_query"]],
            system="S1",
            actor_model=ACTOR_MODELS[0],
            db_path=Path(temp_dir) / "memtrace.sqlite3",
            episode_kind="stateful_attack",
            episode_payload_type="direct_override",
            episode_horizon=3,
            writer_backend="profile",
            planner_backend="profile",
        )
        final_prior_memory = trace[-1]["prior_memory_state"]
        assert final_prior_memory
        assert {record["task_id"] for record in final_prior_memory} == {task["task_id"]}


def test_pipeline_logs_policy_checker_and_responder(monkeypatch, caplog) -> None:
    task = POLICY_TASKS[0]
    with tempfile.TemporaryDirectory() as temp_dir:
        _use_temp_corpus(monkeypatch, temp_dir)
        _use_fixture_retrieval(monkeypatch, task)
        caplog.set_level("INFO")
        run_episode(
            episode_id="ep:one-shot:direct_override",
            turns=[task["poison_query"]],
            system="S1",
            actor_model=ACTOR_MODELS[0],
            db_path=Path(temp_dir) / "memtrace.sqlite3",
            episode_kind="one_shot_attack",
            episode_payload_type="direct_override",
            episode_horizon=1,
            writer_backend="profile",
            planner_backend="profile",
        )
        messages = [record.getMessage() for record in caplog.records]
        assert any("policy_checker" in message for message in messages)
        assert any("responder" in message for message in messages)
