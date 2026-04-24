"""Memory writer scaffolding."""

import json

from memtrace.benchmark import build_task_records
from memtrace.config import MAX_MEMORY_CANDIDATES, MEMORY_WRITER_MAX_TOKENS, PROMPTS_DIR
from memtrace.models.actor import ActorModel
from memtrace.parsing import extract_json_payload
from memtrace.schema import MemoryCandidate, RetrievedPassage


def normalize_writer_output(candidates: list[dict]) -> list[dict]:
    return candidates[:MAX_MEMORY_CANDIDATES]


def build_memory_writer_input(retrieved_passages: list[RetrievedPassage], query: str) -> str:
    prompt = (PROMPTS_DIR / "memory_writer.txt").read_text(encoding="utf-8")
    passages = [
        f"[source_id: {passage.source_id}, source_kind: retrieval] {passage.text}"
        for passage in retrieved_passages
    ]
    return "\n\n".join([prompt, f"Query: {query}", "Retrieved passages:", *passages])


def parse_writer_json_output(raw_output: str, retrieved_passages: list[RetrievedPassage]) -> list[MemoryCandidate]:
    required_keys = {"memory_type", "content", "source_id", "source_kind"}
    try:
        decoded = json.loads(extract_json_payload(raw_output))
    except json.JSONDecodeError:
        return []
    if not isinstance(decoded, list):
        return []

    retrieved_passages_by_source_id = {
        passage.source_id: passage
        for passage in retrieved_passages
    }
    retrieved_source_ids = set(retrieved_passages_by_source_id)
    candidates = []
    for item in decoded[:MAX_MEMORY_CANDIDATES]:
        if not isinstance(item, dict):
            continue
        if not required_keys <= set(item):
            continue
        if item["source_id"] not in retrieved_source_ids or item["source_kind"] != "retrieval":
            continue
        try:
            candidates.append(
                MemoryCandidate(
                    memory_type=item["memory_type"],
                    content=item["content"],
                    source_id=item["source_id"],
                    source_kind=item["source_kind"],
                    task_id=retrieved_passages_by_source_id[item["source_id"]].task_id,
                )
            )
        except (TypeError, ValueError):
            continue
    return candidates


def generate_memory_candidates_with_actor(
    retrieved_passages: list[RetrievedPassage],
    query: str,
    actor_model: ActorModel,
) -> tuple[list[MemoryCandidate], str]:
    writer_input = build_memory_writer_input(retrieved_passages=retrieved_passages, query=query)
    raw_output = actor_model.generate(writer_input, max_tokens=MEMORY_WRITER_MAX_TOKENS)
    return parse_writer_json_output(raw_output=raw_output, retrieved_passages=retrieved_passages), raw_output


def extract_memory_candidates(
    retrieved_passages: list[RetrievedPassage],
    query: str,
    actor_model: ActorModel | None = None,
) -> list[MemoryCandidate]:
    if actor_model is not None:
        candidates, _ = generate_memory_candidates_with_actor(
            retrieved_passages=retrieved_passages,
            query=query,
            actor_model=actor_model,
        )
        return candidates

    candidates = []
    for passage in retrieved_passages[:MAX_MEMORY_CANDIDATES]:
        candidates.append(
            MemoryCandidate(
                memory_type=_profile_memory_type(passage),
                content=passage.text[:180],
                source_id=passage.source_id,
                source_kind="retrieval",
                task_id=passage.task_id,
            )
        )
    return candidates


def _profile_memory_type(passage: RetrievedPassage) -> str:
    if passage.task_id is None:
        return "tool_argument"
    task_family_by_id = {task.task_id: task.family for task in build_task_records()}
    if task_family_by_id.get(passage.task_id) == "policy_memory":
        return "policy_rule"
    return "tool_argument"
