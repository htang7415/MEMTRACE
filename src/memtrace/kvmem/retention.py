"""Phase 6 M1: how the provider's prompt cache behaves across the gaps of real agent sessions.

For each later call in a session, the reusable prefix is bounded by the previous call's prompt:
`reusable = min(prompt, previous prompt)`. Tokens in that bound that the provider did not serve from cache
are waste (`max(0, reusable - cached)`), attributed to the first cause that applies, in this order:

- model_switch: the call runs on a different model than the previous one (caches are per model);
- compaction: the prompt shrank by at least `COMPACTION_SHRINK`, so the context was compacted or edited;
- expiry: the gap since the previous call is at least `EXPIRY_GAP` (the common 5-minute cache lifetime);
- placement: the gap is under `PLACEMENT_GAP` and the prompt did not shrink, so the cache should still exist
  but the call missed it -- or the prompt grew while earlier content (system prompt, tool list) was rewritten,
  which token counts cannot reveal, so this is an upper bound on true placement misses;
- placement_or_edit: as placement, but the prompt shrank slightly; a miss here is either placement or an edit
  near the start of the context, which the trace cannot tell apart;
- other: gaps in between, where expiry and placement cannot be told apart.

`retained(lifetime)` is what a cache with that lifetime (refreshed on every use) and perfect placement would
serve: the reusable prefix of every same-model call that was not compacted and came within the lifetime.
`working_set` is the KV that cache holds: each call's prompt + completion, from the call's start until the
session's next call or the lifetime after the call ends.

The trace's cached counts reflect one unnamed provider's policy: an observed baseline, not ground truth.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from memtrace.kvmem.costs import ANTHROPIC, breakeven_storage, relative_cost
from memtrace.kvmem.traces import TraceCall, TraceSession, read_sessions, trace_time

PLACEMENT_GAP = 10.0
EXPIRY_GAP = 300.0
COMPACTION_SHRINK = 0.10
GAP_BINS = ((10.0, "<10s"), (60.0, "10-60s"), (300.0, "1-5min"), (600.0, "5-10min"), (3600.0, "10-60min"))
CAUSES = ("model_switch", "compaction", "expiry", "placement", "placement_or_edit", "other")
LIFETIMES = {"5min": 300.0, "1h": 3600.0, "24h": 86400.0}
# KV-cache bytes per token in bf16: K and V x layers x KV heads x head dim x 2 bytes, from each model's
# config.json. Qwen2.5-72B stands in for the 70B class. Trace token counts come from the provider's own models,
# so these sizes answer "if this traffic were served by model X".
KV_BYTES_PER_TOKEN = {
    "Qwen3-4B": 2 * 36 * 8 * 128 * 2,
    "Qwen3-32B": 2 * 64 * 8 * 128 * 2,
    "Qwen2.5-72B": 2 * 80 * 8 * 128 * 2,
}


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
    def compacted(self) -> bool:
        return self.call.prompt <= (1 - COMPACTION_SHRINK) * self.previous.prompt

    def retained(self, lifetime: float) -> int:
        same_prefix = self.call.model == self.previous.model and not self.compacted
        return self.reusable if same_prefix and self.gap < lifetime else 0

    @property
    def cause(self) -> str:
        if self.call.model != self.previous.model:
            return "model_switch"
        if self.compacted:
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


def trace_window(sessions: list[TraceSession]) -> tuple[float, float]:
    """The UTC days the sessions are partitioned under, or the span of their calls if they carry no date."""
    dates = sorted({s.date for s in sessions if s.date})
    if dates:
        return trace_time(dates[0] + "T00:00:00Z"), trace_time(dates[-1] + "T00:00:00Z") + 86400.0
    calls = [call for session in sessions for call in session.calls]
    if not calls:
        return 0.0, 0.0
    return min(c.start for c in calls), max(c.end for c in calls)


def working_set(sessions: list[TraceSession], lifetime: float) -> tuple[float, int]:
    """Time-averaged and peak tokens held within `trace_window`."""
    start, end = trace_window(sessions)
    events: dict[float, int] = defaultdict(int)
    for session in sessions:
        for call, following in zip(session.calls, [*session.calls[1:], None]):
            until = call.end + lifetime if following is None else min(following.start, call.end + lifetime)
            held = call.prompt + call.completion
            events[call.start] += held
            events[until] -= held
    held, peak, area, now = 0, 0, 0.0, start
    for t in sorted(events):
        clipped = min(max(t, start), end)
        if start <= t:
            peak = max(peak, held)  # what was held up to t, including caches written before the window
        area += held * (clipped - now)
        now = clipped
        held += events[t]
        if start <= t < end:
            peak = max(peak, held)
    return (area / (end - start) if end > start else 0.0), peak


def analyze(sessions: Iterable[TraceSession]) -> dict[str, object]:
    sessions = list(sessions)
    items = list(later_calls(sessions))
    labels = [label for _, label in GAP_BINS] + [">1h"]
    bins = {label: _Bin() for label in labels}
    causes = {cause: {"calls": 0, "waste": 0} for cause in CAUSES}
    grew_near_total_miss = 0
    # Paired test for placement_or_edit: at short gaps on the same model, does a slight shrink make a near-total
    # miss more likely than growth does? Placement should not care whether the prompt shrank; an edit does.
    short = {"grew": [0, 0], "shrank_slightly": [0, 0]}  # [calls, near-total misses]
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
        if item.gap < PLACEMENT_GAP and item.call.model == item.previous.model and not item.compacted:
            group = short["grew" if item.call.prompt >= item.previous.prompt else "shrank_slightly"]
            group[0] += 1
            group[1] += item.call.cached < 0.1 * item.reusable
    gaps = sorted(item.gap for item in items)
    p90: float | None
    p99: float | None
    if len(gaps) >= 2:
        q = statistics.quantiles(gaps, n=100)
        p90, p99 = q[89], q[98]
    else:  # quantiles() needs two points; a single gap is every percentile
        p90 = p99 = gaps[0] if gaps else None
    later_prompt = sum(item.call.prompt for item in items)
    total_prompt = sum(call.prompt for session in sessions for call in session.calls)
    waste = sum(c["waste"] for c in causes.values())
    switch_calls = sum(1 for item in items if item.call.model != item.previous.model)
    return {
        "sessions": len(sessions),
        "sessions_with_later_calls": sum(1 for s in sessions if len(s.calls) >= 2),
        "calls": sum(len(s.calls) for s in sessions),
        "later_calls": len(items),
        "gap_seconds": {"median": statistics.median(gaps), "p90": p90, "p99": p99} if gaps else None,
        "hit_by_gap": {
            label: {
                "share_of_later_calls": _ratio(b.calls, len(items)),
                "hit_ratio_tokens": b.cached / b.prompt if b.prompt else None,
                "hit_ratio_mean_per_call": statistics.fmean(b.ratios) if b.ratios else None,
            }
            for label, b in bins.items()
        },
        "waste_tokens": waste,
        "waste_share_of_later_prompt": _ratio(waste, later_prompt),
        "waste_by_cause": {
            cause: {"calls": c["calls"], "tokens": c["waste"], "share": c["waste"] / waste if waste else 0.0}
            for cause, c in causes.items()
        },
        "placement_near_total_miss_share_of_waste": grew_near_total_miss / waste if waste else 0.0,
        "short_gap_near_total_miss_rate": {
            k: misses / calls if calls else None for k, (calls, misses) in short.items()
        },
        "model_switch_calls": switch_calls,
        "model_switch_share_of_later_calls": _ratio(switch_calls, len(items)),
        "lifetimes": {name: _lifetime(sessions, items, seconds) for name, seconds in LIFETIMES.items()},
        "relative_cost": {
            p.name: relative_cost(total_prompt, sum(i.retained(p.lifetime) for i in items), p) for p in ANTHROPIC
        },
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _lifetime(sessions: list[TraceSession], items: list[LaterCall], lifetime: float) -> dict[str, object]:
    reusable = sum(item.reusable for item in items)
    retained = sum(item.retained(lifetime) for item in items)
    mean, peak = working_set(sessions, lifetime)
    start, end = trace_window(sessions)
    hours = (end - start) / 3600
    return {
        "retained_share_of_reusable": retained / reusable if reusable else 0.0,
        "observed_cached_share_of_reusable": sum(min(i.call.cached, i.reusable) for i in items) / reusable
        if reusable
        else 0.0,
        "working_set_tokens": {"mean": mean, "peak": peak},
        "breakeven_storage_per_token_hour": breakeven_storage(retained, mean * hours),
        "working_set_gb": {
            model: {"mean": mean * size / 1e9, "peak": peak * size / 1e9} for model, size in KV_BYTES_PER_TOKEN.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths", nargs="*", help="trace shards (default: every downloaded day under data/public/copilot_agent)"
    )
    parser.add_argument("--out", type=Path, help="write the summary JSON here as well as to stdout")
    parser.add_argument("--by-day", action="store_true", help="one summary per trace day (shard directory)")
    args = parser.parse_args()
    paths = sorted(Path(p) for p in args.paths or glob.glob("data/public/copilot_agent/date=*/shard-*.jsonl.gz"))
    if not paths:
        raise SystemExit("no trace shards found; run scripts/fetch_public_datasets.py copilot_agent")
    if args.by_day:
        days: dict[str, list[Path]] = defaultdict(list)
        for path in paths:
            days[path.parent.name.removeprefix("date=")].append(path)
        summary = json.dumps({day: analyze(read_sessions(shards)) for day, shards in sorted(days.items())}, indent=2)
    else:
        summary = json.dumps(analyze(read_sessions(paths)), indent=2)
    print(summary)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(summary + "\n")


if __name__ == "__main__":
    main()
