"""Trace helpers."""

from memtrace.schema import TraceTurn


def build_trace_turn(episode_id: str, turn: int, query: str) -> TraceTurn:
    return TraceTurn(episode_id=episode_id, turn=turn, query=query)

