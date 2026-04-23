from pathlib import Path

import pytest

import memtrace.retrieval as retrieval_module
from memtrace.corpus import save_jsonl
from memtrace.retrieval import retrieve


def test_retrieve_empty_when_missing_file() -> None:
    result = retrieve("query", Path("missing.jsonl"), 5)
    assert result == []


def test_official_retrieval_requires_dense_index(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "passages.jsonl"
    save_jsonl(path, [{"source_id": "P001", "text": "finance update"}])
    monkeypatch.setattr(retrieval_module, "PASSAGES_PATH", path)
    monkeypatch.setattr(retrieval_module, "DENSE_INDEX_PATH", tmp_path / "missing-index.npz")
    with pytest.raises(RuntimeError, match="precomputed dense index"):
        retrieve("finance update", path, 5)


def test_non_official_retrieval_uses_lexical_fallback(tmp_path: Path) -> None:
    path = tmp_path / "passages.jsonl"
    save_jsonl(
        path,
        [
            {"source_id": "P002", "text": "finance finance guidance"},
            {"source_id": "P001", "text": "finance update"},
        ],
    )
    hits = retrieve("finance update", path, 2)
    assert [hit.source_id for hit in hits] == ["P001", "P002"]
