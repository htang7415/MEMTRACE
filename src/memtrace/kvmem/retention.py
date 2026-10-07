"""Phase 6 M1: how the provider's prompt cache behaves across the gaps of real agent sessions.

For each later call in a session, the reusable prefix is bounded by the previous call's prompt:
`reusable = min(prompt, previous prompt)`. Tokens in that bound that the provider did not serve from cache
are waste (`max(0, reusable - cached)`), attributed to the first cause that applies, in this order:

- model_switch: the call runs on a different model than the previous one (caches are per model);
- compaction: the prompt shrank by at least `COMPACTION_SHRINK`, so the context was compacted or edited;
- expiry: the gap since the previous call is at least `EXPIRY_GAP` (the common 5-minute cache lifetime);
- placement: the gap is under `PLACEMENT_GAP` and the prompt did not shrink, so the cache should still exist
  but the call missed it;
- placement_or_edit: as placement, but the prompt shrank slightly; a miss here is either placement or an edit
  near the start of the context, which the trace cannot tell apart;
- other: gaps in between, where expiry and placement cannot be told apart.

The trace's cached counts reflect one unnamed provider's policy: an observed baseline, not ground truth.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from memtrace.kvmem.traces import TraceCall, TraceSession, read_sessions

PLACEMENT_GAP = 10.0
EXPIRY_GAP = 300.0
COMPACTION_SHRINK = 0.10
GAP_BINS = ((10.0, "<10s"), (60.0, "10-60s"), (300.0, "1-5min"), (600.0, "5-10min"), (3600.0, "10-60min"))
CAUSES = ("model_switch", "compaction", "expiry", "placement", "placement_or_edit", "other")


@dataclass(frozen=True)
class LaterCall:
    previous: TraceCall
    call: TraceCall
    gap: float  # seconds from the previous call's completion to this call's start

    @property
    def reusable(self) -> int:
        return min(self.call.prompt, self.previous.prompt)

    @property
    def waste(self) -> int:
        return max(0, self.reusable - self.call.cached)

    @property
    def cause(self) -> str:
        if self.call.model != self.previous.model:
            return "model_switch"
        if self.call.prompt <= (1 - COMPACTION_SHRINK) * self.previous.prompt:
            return "compaction"
        if self.gap >= EXPIRY_GAP:
            return "expiry"
        if self.gap < PLACEMENT_GAP:
            return "placement" if self.call.prompt >= self.previous.prompt else "placement_or_edit"
        return "other"


@dataclass
class _Bin:
    calls: int = 0
    prompt: int = 0
    cached: int = 0
    ratios: list[float] = field(default_factory=list)


def later_calls(sessions: Iterable[TraceSession]) -> Iterator[LaterCall]:
    for session in sessions:
        for previous, call in zip(session.calls, session.calls[1:]):
            yield LaterCall(previous, call, call.start - previous.end)


def gap_bin(gap: float) -> str:
    return next((label for bound, label in GAP_BINS if gap < bound), ">1h")


def analyze(sessions: Iterable[TraceSession]) -> dict[str, object]:
    sessions = list(sessions)
    items = list(later_calls(sessions))
    labels = [label for _, label in GAP_BINS] + [">1h"]
    bins = {label: _Bin() for label in labels}
    causes = {cause: {"calls": 0, "waste": 0} for cause in CAUSES}
    grew_near_total_miss = 0
    for item in items:
        b = bins[gap_bin(item.gap)]
        b.calls += 1
        b.prompt += item.call.prompt
        b.cached += item.call.cached
        if item.call.prompt:
            b.ratios.append(item.call.cached / item.call.prompt)
        if item.waste:
            causes[item.cause]["calls"] += 1
            causes[item.cause]["waste"] += item.waste
            if item.cause == "placement" and item.call.cached < 0.1 * item.reusable:
                grew_near_total_miss += item.waste  # prompt only grew, yet (almost) nothing was cached
    gaps = sorted(item.gap for item in items)
    q = statistics.quantiles(gaps, n=100)
    later_prompt = sum(item.call.prompt for item in items)
    waste = sum(c["waste"] for c in causes.values())
    switch_calls = sum(1 for item in items if item.call.model != item.previous.model)
    return {
        "sessions": len(sessions),
        "sessions_with_later_calls": sum(1 for s in sessions if len(s.calls) >= 2),
        "calls": sum(len(s.calls) for s in sessions),
        "later_calls": len(items),
        "gap_seconds": {"median": statistics.median(gaps), "p90": q[89], "p99": q[98]},
        "hit_by_gap": {
            label: {
                "share_of_later_calls": b.calls / len(items),
                "hit_ratio_tokens": b.cached / b.prompt if b.prompt else None,
                "hit_ratio_mean_per_call": statistics.fmean(b.ratios) if b.ratios else None,
            }
            for label, b in bins.items()
        },
        "waste_tokens": waste,
        "waste_share_of_later_prompt": waste / later_prompt,
        "waste_by_cause": {
            cause: {"calls": c["calls"], "tokens": c["waste"], "share": c["waste"] / waste if waste else 0.0}
            for cause, c in causes.items()
        },
        "placement_near_total_miss_share_of_waste": grew_near_total_miss / waste if waste else 0.0,
        "model_switch_calls": switch_calls,
        "model_switch_share_of_later_calls": switch_calls / len(items),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths", nargs="*", help="trace shards (default: every downloaded day under data/public/copilot_agent)"
    )
    parser.add_argument("--out", type=Path, help="write the summary JSON here as well as to stdout")
    args = parser.parse_args()
    paths = sorted(Path(p) for p in args.paths or glob.glob("data/public/copilot_agent/date=*/shard-*.jsonl.gz"))
    if not paths:
        raise SystemExit("no trace shards found; run scripts/fetch_public_datasets.py copilot_agent")
    summary = json.dumps(analyze(read_sessions(paths)), indent=2)
    print(summary)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(summary + "\n")


if __name__ == "__main__":
    main()
