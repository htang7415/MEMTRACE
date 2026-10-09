"""GitHub Copilot coding-agent traces 2026 (Azure Public Dataset), streamed from the daily archives.

Each archive `copilot_agent/date.<day>.tar.gz` holds gzipped JSONL shards (`date=<day>/shard-NNNN.jsonl.gz`),
one session per line: turns of LLM calls with per-segment token counts, and tool batches. Metadata only:
no prompts, code, or tool output. `iter_*` yield the raw records; `read_sessions` yields typed sessions
with call timing.

Call `timestamp`s mark completion (consecutive calls never overlap under that reading; 32% would if they
marked the start), so a call starts at `timestamp - duration_ms`.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
import gzip
import json
from pathlib import Path
import tarfile
from typing import Any

from memtrace.datasets.sources import DATASET_ROOT, verified_path

DAYS = tuple(f"2026-06-0{d}" for d in range(1, 8))


def archive(day: str) -> str:
    return f"copilot_agent/date.{day}.tar.gz"


def archive_path(day: str, root: Path = DATASET_ROOT) -> Path:
    """The verified archive of one trace day."""
    return verified_path(archive(day), root)


def iter_sessions(day: str, root: Path = DATASET_ROOT, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Sessions of one day in archive order (shards as stored), without extracting to disk."""
    for n, session in enumerate(iter_archive(archive_path(day, root)), 1):
        yield session
        if limit is not None and n >= limit:
            return


def iter_archive(path: Path, shards: Collection[str] | None = None) -> Iterator[dict[str, Any]]:
    """Sessions of one daily archive file (already verified by the caller), optionally only the named
    shards (member names such as `date=2026-06-06/shard-0000.jsonl.gz`)."""
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            if not member.name.endswith(".jsonl.gz") or (shards is not None and member.name not in shards):
                continue
            with gzip.open(tar.extractfile(member)) as fh:  # type: ignore[arg-type]
                for line in fh:
                    yield json.loads(line)


@dataclass(frozen=True)
class TraceCall:
    start: float  # epoch seconds
    end: float
    model: str
    prompt: int
    cached: int
    completion: int
    turn_end: bool = False  # last call of its turn: the agent stopped calling tools and waits for the user


@dataclass(frozen=True)
class TraceSession:
    session_id: str
    calls: tuple[TraceCall, ...]  # in completion order
    date: str = ""  # the dataset's day partition (YYYY-MM-DD); a few sessions also hold calls from earlier days


def trace_time(stamp: str) -> float:
    base, _, frac = stamp.rstrip("Z").partition(".")
    return datetime.fromisoformat(base + "+00:00").timestamp() + float("0." + (frac or "0"))


def parse_session(session: dict[str, Any]) -> TraceSession:
    """Typed view of one raw session; calls without integer prompt / cached / completion counts are skipped."""
    raw: list[tuple[dict[str, Any], bool]] = []
    for turn in session["turns"]:
        valid = sorted(
            (
                c
                for c in turn["llm_calls"]
                if all(isinstance(c["tokens"].get(k), int) for k in ("prompt", "cached", "completion"))
            ),
            key=lambda c: c["timestamp"],
        )
        raw.extend((c, c is valid[-1]) for c in valid)
    raw.sort(key=lambda item: item[0]["timestamp"])
    calls = []
    for c, turn_end in raw:
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
                turn_end,
            )
        )
    return TraceSession(session["session_id"], tuple(calls), session.get("date") or "")


def read_sessions(paths: Iterable[Path], shards: Collection[str] | None = None) -> Iterator[TraceSession]:
    """Typed sessions, in file order, from daily archives (`.tar.gz`, optionally only `shards`) or single
    shard files (`.jsonl.gz`)."""
    for path in paths:
        if path.name.endswith(".tar.gz"):
            records: Iterable[dict[str, Any]] = iter_archive(path, shards)
            yield from map(parse_session, records)
        else:
            with gzip.open(path, "rt") as handle:
                for line in handle:
                    yield parse_session(json.loads(line))
