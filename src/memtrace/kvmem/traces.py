"""GitHub Copilot coding-agent traces (Azure, 2026): per-session LLM calls with token counts and timing.

Call `timestamp`s mark completion (consecutive calls never overlap under that reading; 32% would if they
marked the start), so a call starts at `timestamp - duration_ms`.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class TraceCall:
    start: float  # epoch seconds
    end: float
    model: str
    prompt: int
    cached: int
    completion: int


@dataclass(frozen=True)
class TraceSession:
    session_id: str
    calls: tuple[TraceCall, ...]  # in completion order
    date: str = ""  # the dataset's day partition (YYYY-MM-DD); a few sessions also hold calls from earlier days


def trace_time(stamp: str) -> float:
    base, _, frac = stamp.rstrip("Z").partition(".")
    return datetime.fromisoformat(base + "+00:00").timestamp() + float("0." + (frac or "0"))


def read_sessions(paths: list[Path]) -> Iterator[TraceSession]:
    """Sessions in file order; calls without integer prompt / cached / completion counts are skipped."""
    for path in paths:
        with gzip.open(path, "rt") as handle:
            for line in handle:
                session = json.loads(line)
                raw = sorted(
                    (
                        c
                        for turn in session["turns"]
                        for c in turn["llm_calls"]
                        if all(isinstance(c["tokens"].get(k), int) for k in ("prompt", "cached", "completion"))
                    ),
                    key=lambda c: c["timestamp"],
                )
                calls = []
                for c in raw:
                    end = trace_time(c["timestamp"])
                    tokens = c["tokens"]
                    calls.append(
                        TraceCall(
                            end - c["duration_ms"] / 1000,
                            end,
                            c.get("model") or "",
                            tokens["prompt"],
                            tokens["cached"],
                            tokens["completion"],
                        )
                    )
                yield TraceSession(session["session_id"], tuple(calls), session.get("date") or "")
