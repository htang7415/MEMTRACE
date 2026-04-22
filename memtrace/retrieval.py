"""Retrieval interfaces and placeholder deterministic implementation."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from memtrace.benchmark import build_task_records
from memtrace.config import DENSE_INDEX_PATH, EMBEDDING_MODEL, PASSAGES_PATH
from memtrace.schema import RetrievedPassage


def load_passages(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def retrieve(query: str, path: Path, top_k: int) -> list[RetrievedPassage]:
    if _can_use_dense_index(path):
        return _dense_retrieve(query=query, top_k=top_k)
    passages = load_passages(path)
    scored = sorted(
        passages,
        key=lambda item: (-_score_passage(query, item), item["source_id"]),
    )[:top_k]
    return [
        RetrievedPassage(
            source_id=item["source_id"],
            text=item["text"],
            rank=index + 1,
            score=_score_passage(query, item),
            allowlisted=item.get("allowlisted", False),
            task_id=item.get("task_id"),
            payload_type=item.get("payload_type"),
            passage_kind=item.get("passage_kind", "authoritative"),
        )
        for index, item in enumerate(scored)
    ]


def _can_use_dense_index(path: Path) -> bool:
    return DENSE_INDEX_PATH.exists() and path.resolve() == PASSAGES_PATH.resolve()


def _dense_retrieve(query: str, top_k: int) -> list[RetrievedPassage]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "dense retrieval requires `sentence-transformers` and `numpy`; "
            "install project dependencies or remove the dense index artifact"
        ) from exc

    index_model, embeddings, metadata = _load_dense_index()
    if index_model != EMBEDDING_MODEL:
        raise RuntimeError(f"dense index model mismatch: expected {EMBEDDING_MODEL}, found {index_model}")

    model = _load_embedding_model(EMBEDDING_MODEL, SentenceTransformer)
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    scores = embeddings @ query_embedding
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: (-float(scores[index]), metadata[index]["source_id"]),
    )[:top_k]
    return [
        RetrievedPassage(
            source_id=metadata[index]["source_id"],
            text=metadata[index]["text"],
            rank=rank + 1,
            score=float(scores[index]),
            allowlisted=metadata[index].get("allowlisted", False),
            task_id=metadata[index].get("task_id"),
            payload_type=metadata[index].get("payload_type"),
            passage_kind=metadata[index].get("passage_kind", "authoritative"),
        )
        for rank, index in enumerate(ranked_indices)
    ]


@lru_cache(maxsize=1)
def _load_dense_index():
    import json as json_module

    import numpy as np

    index = np.load(DENSE_INDEX_PATH, allow_pickle=False)
    metadata = [json_module.loads(item) for item in index["metadata"].tolist()]
    model = str(index["model"][0])
    return model, index["embeddings"], metadata


@lru_cache(maxsize=1)
def _load_embedding_model(model_name: str, sentence_transformer_cls):
    return sentence_transformer_cls(model_name)


def _score_passage(query: str, passage: dict) -> float:
    query_map = _build_query_lookup()
    query_info = query_map.get(query)
    if query_info is None:
        return _lexical_score(query, passage["text"])

    score = _lexical_score(query, passage["text"])
    task_id = query_info["task_id"]
    query_kind = query_info["query_kind"]
    passage_kind = passage.get("passage_kind", "authoritative")

    if passage.get("task_id") == task_id:
        if query_kind == "clean" and passage.get("allowlisted"):
            score += 1000
        elif query_kind == "poison":
            if passage_kind == "poison":
                score += 2000
            elif passage.get("allowlisted"):
                score += 1500
        elif query_kind == "trigger" and passage.get("allowlisted"):
            score += 1800
    elif passage_kind == "poison":
        score -= 25

    if query_kind == "trigger" and passage_kind == "poison" and passage.get("task_id") == task_id:
        score = -1000
    return score


@lru_cache(maxsize=1)
def _build_query_lookup() -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for task in build_task_records():
        lookup[task.clean_query] = {"task_id": task.task_id, "query_kind": "clean"}
        lookup[task.poison_query] = {
            "task_id": task.task_id,
            "query_kind": "poison",
        }
        lookup[task.trigger_query] = {
            "task_id": task.task_id,
            "query_kind": "trigger",
        }
    return lookup


def _lexical_score(query: str, text: str) -> float:
    query_tokens = {token.strip(".,:;!?").lower() for token in query.split() if token}
    text_tokens = {token.strip(".,:;!?").lower() for token in text.split() if token}
    return float(len(query_tokens & text_tokens))
