from pathlib import Path

from memtrace.benchmark import POLICY_TASKS
from memtrace.corpus import build_corpus, save_jsonl
from memtrace.retrieval import retrieve


def test_retrieve_empty_when_missing_file() -> None:
    result = retrieve("query", Path("missing.jsonl"), 5)
    assert result == []


def test_poison_query_ranks_poison_passage_high(tmp_path: Path) -> None:
    path = tmp_path / "passages.jsonl"
    save_jsonl(path, build_corpus())
    task = POLICY_TASKS[0]
    hits = retrieve(task["poison_query"], path, 10)
    poison_ranks = [
        hit.rank
        for hit in hits
        if hit.task_id == task["task_id"] and hit.passage_kind == "poison"
    ]
    assert min(poison_ranks) <= 3


def test_trigger_query_pushes_poison_passage_below_top_five(tmp_path: Path) -> None:
    path = tmp_path / "passages.jsonl"
    save_jsonl(path, build_corpus())
    task = POLICY_TASKS[0]
    hits = retrieve(task["trigger_query"], path, 10)
    poison_ranks = [
        hit.rank
        for hit in hits
        if hit.task_id == task["task_id"] and hit.passage_kind == "poison"
    ]
    assert poison_ranks == []


def test_allowlisted_passage_stays_in_top_five_for_trigger_query(tmp_path: Path) -> None:
    path = tmp_path / "passages.jsonl"
    save_jsonl(path, build_corpus())
    task = POLICY_TASKS[0]
    hits = retrieve(task["trigger_query"], path, 10)
    allowlisted_ranks = [
        hit.rank
        for hit in hits
        if hit.task_id == task["task_id"] and hit.allowlisted
    ]
    assert min(allowlisted_ranks) <= 5
