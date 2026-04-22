"""Planner scaffolding."""


def build_planner_input(query: str, retrieved_context: str, memory_block: str) -> str:
    return "\n".join(
        [
            f"Query: {query}",
            f"Retrieved:\n{retrieved_context}",
            f"Memory:\n{memory_block}",
        ]
    )

