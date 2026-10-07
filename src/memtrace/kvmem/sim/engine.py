"""Phase 6 M2: deterministic trace-driven simulator of KV retention and placement across a replica pool.

Each call's start routes it to a replica and looks up its session's cached prefix there, in any tier; the lookup
consumes the entry. Its end stores the session's new context (prompt + completion) in that replica's GPU tier.
Tier capacity is the budget for these idle prefixes; KV of running calls is not counted. Over capacity, entries
are demoted a tier (dropped from the last), oldest first. Under a turn-aware policy the GPU tier demotes entries of
sessions whose turn ended first; lower tiers hold mostly such entries, so they stay LRU (evicting idle-turn KV
first there drops exactly what they are for).

A hit serves `min(prompt, prompt of the call that wrote the entry)` tokens, as M1's `reusable`, unless the model
changed or the prompt shrank by `COMPACTION_SHRINK` or more. A hit from a slower tier is used only if loading it
is faster than recomputing it at `prefill_tokens_per_second`. With one replica, unlimited GPU capacity and a
lifetime L, hits equal M1's `retained(L)` (tested).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from memtrace.kvmem.retention import COMPACTION_SHRINK, later_calls, trace_window
from memtrace.kvmem.sim.policies import Retention, session_hash
from memtrace.kvmem.sim.tiers import Entry, Tier, TierStore
from memtrace.kvmem.traces import TraceSession


@dataclass(frozen=True)
class SimConfig:
    replicas: int
    tiers: tuple[Tier, ...]  # GPU first
    kv_bytes_per_token: int
    retention: Retention
    router: str
    max_inflight: int = 8  # concurrent calls per replica before the router spills to the least-loaded one
    sticky_idle: float = math.inf  # approximate router: forget a session's replica after this long idle
    prefill_tokens_per_second: float = 10_000.0  # recompute speed, to price loads from slower tiers


@dataclass
class SimResult:
    calls: int = 0
    prompt_tokens: int = 0
    reusable_tokens: int = 0  # M1's reusable prefix of later calls (same model, not compacted)
    hit_tokens: dict[str, int] = field(default_factory=dict)  # by tier
    load_seconds: float = 0.0
    prefill_seconds_saved: float = 0.0  # recompute time avoided, net of load time
    spills: int = 0  # calls whose preferred replica was full
    gb_hours: dict[str, float] = field(default_factory=dict)  # memory-time held per tier, within the trace window

    def summary(self) -> dict[str, object]:
        hits = sum(self.hit_tokens.values())
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "reusable_tokens": self.reusable_tokens,
            "hit_tokens": self.hit_tokens,
            "hit_share_of_reusable": hits / self.reusable_tokens if self.reusable_tokens else 0.0,
            "prefill_saved_share_of_prompt": hits / self.prompt_tokens if self.prompt_tokens else 0.0,
            "load_seconds": self.load_seconds,
            "prefill_seconds_saved": self.prefill_seconds_saved,
            "spills": self.spills,
            "gb_hours": self.gb_hours,
        }


def simulate(sessions: list[TraceSession], config: SimConfig) -> SimResult:
    n, tiers = config.replicas, config.tiers
    stores = [[TierStore(t, config.kv_bytes_per_token) for t in tiers] for _ in range(n)]
    lifetime = config.retention.lifetime
    result = SimResult(hit_tokens={t.name: 0 for t in tiers})
    result.reusable_tokens = sum(i.retained(math.inf) for i in later_calls(sessions))

    # Events: (time, kind, session index, call index). At the same time, ends (kind 0) come before starts (kind 1),
    # so a call starting as another ends can use its KV; a zero-duration call's end (kind 2) follows its start.
    # One trace call has a negative duration; an end before the start is taken as the start.
    events = []
    for s, session in enumerate(sessions):
        for c, call in enumerate(session.calls):
            events.append((call.start, 1, s, c))
            events.append((max(call.end, call.start), 0 if call.end > call.start else 2, s, c))
    events.sort()

    inflight = [0] * n
    assigned: dict[tuple[int, int], int] = {}
    last_sent: dict[int, int] = {}
    last_end: dict[int, float] = {}
    holder: dict[int, int] = {}  # session -> replica holding its newest entry
    expiries: deque[tuple[float, int, Entry]] = deque()  # in last_use order: entries are written in time order
    window_start, window_end = trace_window(sessions)
    held = [0] * len(tiers)  # tokens per tier, summed over replicas
    area = [0.0] * len(tiers)
    now = window_start

    def advance(t: float) -> None:
        nonlocal now
        clipped = min(max(t, window_start), window_end)
        if clipped > now:
            for k in range(len(tiers)):
                area[k] += held[k] * (clipped - now)
            now = clipped

    def take(s: int, entry: Entry) -> None:
        """Remove a stored entry from wherever it is."""
        assert entry.where is not None
        r, k = entry.where
        stores[r][k].pop(s)
        held[k] -= entry.tokens
        entry.where = None
        if holder.get(s) == r:
            del holder[s]

    def place(s: int, entry: Entry, r: int, idle: bool) -> None:
        """Store an entry in replica r's GPU tier, demoting the oldest entries down the tiers to make room."""
        k = 0
        while True:
            store = stores[r][k]
            store.add(s, entry, idle)
            held[k] += entry.tokens
            entry.where = (r, k)
            if store.tokens <= store.capacity_tokens:
                return
            s, entry, idle = store.victim()
            held[k] -= entry.tokens
            entry.where = None
            k += 1
            idle = False  # turn-awareness applies to the GPU tier only
            if k == len(tiers):
                if holder.get(s) == r:
                    del holder[s]
                return

    for t, kind, s, c in events:
        while expiries and expiries[0][0] <= t:
            expires_at, s_exp, expired = expiries.popleft()
            if expired.where is not None:
                advance(expires_at)  # memory is freed when the entry expires, not at the next event
                take(s_exp, expired)
        advance(t)
        call = sessions[s].calls[c]
        if kind == 1:
            result.calls += 1
            result.prompt_tokens += call.prompt
            preferred = _preferred(config.router, s, sessions[s].session_id, n, last_sent, holder)
            if config.router == "approximate" and t - last_end.get(s, -math.inf) >= config.sticky_idle:
                preferred = None  # idle long enough that its KV is probably gone: place it afresh
            if preferred is not None and inflight[preferred] < config.max_inflight:
                r = preferred
            else:
                result.spills += preferred is not None
                r = min(range(n), key=inflight.__getitem__)
            inflight[r] += 1
            assigned[(s, c)] = r
            last_sent[s] = r
            found = next((e for store in stores[r] if (e := store.get(s))), None)
            if found is None or found.where is None:
                continue
            k = found.where[1]
            take(s, found)
            if found.model != call.model or call.prompt <= (1 - COMPACTION_SHRINK) * found.prompt:
                continue
            hit = min(call.prompt, found.prompt)
            recompute = hit / config.prefill_tokens_per_second
            load = 0.0 if k == 0 else hit * config.kv_bytes_per_token / (tiers[k].load_gb_per_s * 1e9)
            if load < recompute:
                result.hit_tokens[tiers[k].name] += hit
                result.load_seconds += load
                result.prefill_seconds_saved += recompute - load
        else:
            r = assigned.pop((s, c))
            inflight[r] -= 1
            last_end[s] = t
            entry = Entry(call.prompt + call.completion, t, call.model, call.prompt)
            place(s, entry, r, config.retention.turn_aware and call.turn_end)
            if entry.where is not None:
                holder[s] = r
            if math.isfinite(lifetime):
                expiries.append((t + lifetime, s, entry))
    advance(window_end)
    result.gb_hours = {
        tier.name: a * config.kv_bytes_per_token / 1e9 / 3600 for tier, a in zip(tiers, area, strict=True)
    }
    return result


def _preferred(
    router: str, s: int, session_id: str, n: int, last_sent: dict[int, int], holder: dict[int, int]
) -> int | None:
    if router == "least-loaded":
        return None
    if router == "session-key":
        return session_hash(session_id, n)
    if router == "approximate":
        return last_sent.get(s)
    if router == "precise":
        return holder.get(s)
    raise ValueError(f"unknown router {router!r}")
