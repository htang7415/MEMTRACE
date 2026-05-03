"""Deterministic profile backend for tests and local harness checks."""

from __future__ import annotations

import json
import re

from memtrace.benchmark import build_task_records, task_id_by_query, task_spec_by_task_id, trusted_utility_spec_by_task_id
from memtrace.config import MAX_MEMORY_CANDIDATES
from memtrace.models.actor import ActorModel


class ProfileActorModel(ActorModel):
    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        del max_tokens
        if "Output a JSON list of {memory_type, content, source_id, source_kind}." in prompt:
            return self._writer_output(prompt)
        if "Return only JSON in one of these forms:" in prompt:
            return self._planner_output(prompt)
        raise ValueError("profile backend received an unknown prompt shape")

    def _writer_output(self, prompt: str) -> str:
        query = _extract_query(prompt)
        memory_type = _memory_type_for_query(query)
        candidates = []
        for source_id, text in _extract_retrieval_passages(prompt)[:MAX_MEMORY_CANDIDATES]:
            candidates.append(
                {
                    "memory_type": memory_type,
                    "content": text[:180],
                    "source_id": source_id,
                    "source_kind": "retrieval",
                }
            )
        return json.dumps(candidates)

    def _planner_output(self, prompt: str) -> str:
        query = _extract_query(prompt)
        task_id = task_id_by_query().get(query)
        if task_id is None:
            return "null"

        task = next(item for item in build_task_records() if item.task_id == task_id)
        spec = task_spec_by_task_id()[task_id]
        utility_spec = trusted_utility_spec_by_task_id().get(task_id)
        if utility_spec is not None and query == utility_spec["trigger_query"]:
            return json.dumps(
                {
                    "tool_name": utility_spec["expected_tool"],
                    "arguments": utility_spec["expected_arguments"],
                }
            )
        evidence = "\n".join(_extract_memory_lines(prompt) + [text for _, text in _extract_retrieval_passages(prompt)])

        payload_type = None
        for candidate_payload_type, payload_text in task.payloads.items():
            if payload_text in evidence:
                payload_type = candidate_payload_type
                break

        if query == task.poison_query and payload_type != "direct_override":
            return "null"

        tool_name = spec["expected_tool"]
        arguments = spec["expected_arguments"]
        if payload_type is not None:
            tool_name = spec["unsafe_tool"]
            arguments = spec["unsafe_arguments"]

        if tool_name is None:
            return "null"
        return json.dumps({"tool_name": tool_name, "arguments": arguments})


def _extract_query(prompt: str) -> str:
    match = re.search(r"^Query:\s*(.+)$", prompt, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _extract_retrieval_passages(prompt: str) -> list[tuple[str, str]]:
    passages: list[tuple[str, str]] = []
    for match in re.finditer(r"\[source_id:\s*([^,\]]+)(?:,\s*source_kind:\s*[^\]]+)?(?:,\s*rank:\s*\d+)?\]\s*(.+)", prompt):
        passages.append((match.group(1).strip(), match.group(2).strip()))
    return passages


def _extract_memory_lines(prompt: str) -> list[str]:
    if "Memory:\n" not in prompt:
        return []
    memory_block = prompt.split("Memory:\n", 1)[1]
    return [line.strip() for line in memory_block.splitlines() if line.strip()]


def _memory_type_for_query(query: str) -> str:
    task_id = task_id_by_query().get(query)
    if task_id is None:
        return "tool_argument"
    task = next(item for item in build_task_records() if item.task_id == task_id)
    return "policy_rule" if task.family == "policy_memory" else "tool_argument"
