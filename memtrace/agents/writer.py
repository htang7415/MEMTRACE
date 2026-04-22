"""Memory writer scaffolding."""

from memtrace.schema import MemoryCandidate, RetrievedPassage
from memtrace.config import MAX_MEMORY_CANDIDATES


def normalize_writer_output(candidates: list[dict]) -> list[dict]:
    return candidates[:MAX_MEMORY_CANDIDATES]


def extract_memory_candidates(retrieved_passages: list[RetrievedPassage], query: str) -> list[MemoryCandidate]:
    del query
    candidates = []
    for passage in retrieved_passages[:MAX_MEMORY_CANDIDATES]:
        candidates.append(
            MemoryCandidate(
                memory_type="policy_rule" if passage.allowlisted else "tool_argument",
                content=passage.text[:180],
                source_id=passage.source_id,
                source_kind="retrieval",
            )
        )
    return candidates
