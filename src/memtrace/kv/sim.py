"""Offline KV-cache simulator: replay agent sessions at trace timing over N replicas.

Each replica holds a fixed number of KV blocks (`block_tokens` each; 64 for the AgentX traces). A request
reuses the longest leading run of its prompt blocks that is resident on its replica (vLLM prefix caching),
allocates the rest plus its output, holds them for the trace's service time, then releases them. Released
blocks stay cached until evicted. Timing comes from the trace and does not react to hits, so policies are
compared on the same arrivals; the output is cache behaviour (hit rate, recomputed prefill tokens, why
misses happened, memory held), not latency.

Arrivals:
- steady: `concurrency` sessions run at once, in a seeded shuffle; a finished session is replaced by the
  next. Requests after `warmup_s` and before the last session starts are measured.
- trace: every session starts at its own `Session.start` (e.g. a Copilot trace day replayed in real time);
  requests from `warmup_s` after the first start onwards are measured.

Eviction (unreferenced blocks only):
- lru: least recently released first, tail blocks of a prompt first (vLLM's free-queue order).
- ewma: a block released by a stream is protected for `ewma_k` x that stream's average gap between
  calls; blocks whose protection lapsed go first (LRU), then the latest expected return.
- hint: protected for `hint_s` only if the stream's next call comes within `hint_s` (an ideal
  "paused for a tool, back soon" signal from the client or gateway).
- oracle: protected until the stream's true next call; an upper bound for pause-aware retention.
- predicted: like oracle, but the gap is the true gap times a log-normal error exp(N(0, noise_sigma)):
  how accurate a pause-length predictor (e.g. per tool) must be to recover the oracle's gain.
- turn: blocks of a session whose turn ended (it waits for its user) go first, LRU; blocks of a session
  between tool calls are kept until those are gone, oldest first.
A uniform TTL for every block would order evictions exactly like LRU, so retention only helps when it
differs between sessions; that is what the pause-aware policies test. `ttl_s` instead bounds memory:
a block unreferenced for `ttl_s` is dropped from every tier (a provider cache lifetime).

Lower tiers (`cpu_capacity_tokens`, `ssd_capacity_tokens` > 0): a block evicted from the GPU tier is
demoted to a per-replica LRU tier in host memory (vLLM KV offloading / LMCache style), and from there to
an SSD tier, instead of dropped. A request reuses its GPU-resident prefix, loads the continuing run found
in the lower tiers back to the GPU (exclusive: it leaves them), and recomputes only the rest. `load_cost`
and `ssd_load_cost` price a loaded token relative to a recomputed one (an assumption, reported with the
results).

Routing: round_robin (per request), least_loaded (fewest in-flight requests), session (sticky, random
assignment, like prompt_cache_key hashing), sticky (the replica that served the session's previous
request), kv_aware (the replica holding the longest prefix in any tier, known instantly: an idealized
precise prefix index), prefix_load (llm-d style: 3 x cached-prefix fraction + 2 x free in-flight capacity),
tiered_prefix_load (the same, counting the prefix held in lower tiers as cached). With `max_inflight` > 0,
session, sticky and kv_aware send a request whose replica is that busy to the least-loaded one instead.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import heapq
import math
import random
from typing import Sequence

from memtrace.kv.sessions import BLOCK_TOKENS, Request, Session

ROUTINGS = (
    "round_robin",
    "least_loaded",
    "session",
    "sticky",
    "kv_aware",
    "prefix_load",
    "tiered_prefix_load",
)
EVICTIONS = ("lru", "ewma", "hint", "oracle", "predicted", "turn")
ARRIVALS = ("steady", "trace")
EWMA_ALPHA = 0.3
EWMA_PRIOR_S = 2.0  # median gap between calls in the AgentX corpus


@dataclass(frozen=True)
class SimParams:
    replicas: int
    concurrency: int  # steady arrivals: sessions active at once; a finished session is replaced by the next one
    capacity_tokens: int  # KV capacity per replica
    routing: str = "prefix_load"
    eviction: str = "lru"
    ewma_k: float = 4.0
    hint_s: float = 60.0
    noise_sigma: float = 0.0
    warmup_s: float = 1800.0  # requests before this are not measured
    cpu_capacity_tokens: int = 0  # host-memory KV tier per replica; 0 = none
    load_cost: float = 0.2  # cost of loading a token from the CPU tier, relative to recomputing it
    ssd_capacity_tokens: int = 0  # SSD KV tier per replica, below the CPU tier; 0 = none
    ssd_load_cost: float = 0.5
    block_tokens: int = BLOCK_TOKENS
    arrivals: str = "steady"
    ttl_s: float | None = None  # unreferenced KV is dropped from every tier after this long; None = never
    max_inflight: int = 0  # 0 = no limit

    def __post_init__(self) -> None:
        if self.routing not in ROUTINGS:
            raise ValueError(f"routing {self.routing!r} not in {ROUTINGS}")
        if self.eviction not in EVICTIONS:
            raise ValueError(f"eviction {self.eviction!r} not in {EVICTIONS}")
        if self.arrivals not in ARRIVALS:
            raise ValueError(f"arrivals {self.arrivals!r} not in {ARRIVALS}")
        if min(self.cpu_capacity_tokens, self.ssd_capacity_tokens, self.load_cost, self.ssd_load_cost) < 0:
            raise ValueError("lower-tier capacities and load costs must be non-negative")
        if (self.ttl_s is not None and self.ttl_s <= 0) or self.max_inflight < 0 or self.block_tokens < 1:
            raise ValueError("ttl_s and block_tokens must be positive, max_inflight non-negative")
        steady = self.arrivals == "steady"
        if self.replicas < 1 or (steady and self.concurrency < 1) or self.capacity_tokens < self.block_tokens:
            raise ValueError("replicas, concurrency, and capacity_tokens must be positive")


class FreeBlocks:
    """Cached, unreferenced blocks in eviction order: lapsed protection (LRU) first, then the protected
    block with the latest expected return. With no protection this is exactly LRU."""

    def __init__(self) -> None:
        self.idle: OrderedDict[int, None] = OrderedDict()
        self.protected: dict[int, int] = {}  # key -> seq of its live heap entries
        self.by_end: list[tuple[float, int, int]] = []  # (protection end, seq, key)
        self.by_return: list[tuple[float, int, int]] = []  # (-expected return, seq, key)
        self.seq = 0

    def __len__(self) -> int:
        return len(self.idle) + len(self.protected)

    def release(self, key: int, now: float, window: float, expected: float) -> None:
        if window <= 0:
            self.idle[key] = None
            return
        self.seq += 1
        self.protected[key] = self.seq
        heapq.heappush(self.by_end, (now + window, self.seq, key))
        heapq.heappush(self.by_return, (-expected, self.seq, key))

    def take(self, key: int) -> None:
        """A cached block was hit (or expired): it is no longer evictable."""
        if key in self.idle:
            del self.idle[key]
        else:
            self.protected.pop(key, None)

    def evict(self, now: float) -> int | None:
        while self.by_end and self.by_end[0][0] <= now:
            _, seq, key = heapq.heappop(self.by_end)
            if self.protected.get(key) == seq:
                del self.protected[key]
                self.idle[key] = None
        if self.idle:
            return self.idle.popitem(last=False)[0]
        while self.by_return:
            _, seq, key = heapq.heappop(self.by_return)
            if self.protected.get(key) == seq:
                del self.protected[key]
                return key
        return None

    def compact(self) -> None:
        """Drop stale heap entries once they dominate (they are skipped lazily otherwise)."""
        live = len(self.protected)
        for name in ("by_end", "by_return"):
            heap = getattr(self, name)
            if len(heap) > 4 * live + 4096:
                kept = [e for e in heap if self.protected.get(e[2]) == e[1]]
                heapq.heapify(kept)
                setattr(self, name, kept)


class Replica:
    def __init__(self, capacity_blocks: int, cpu_blocks: int = 0, ssd_blocks: int = 0) -> None:
        self.capacity = capacity_blocks
        # Lower tiers, fastest first: LRU dicts of block -> time it was last released (for the TTL)
        self.cpu: OrderedDict[int, float | None] = OrderedDict()
        self.ssd: OrderedDict[int, float | None] = OrderedDict()
        self.lower = [(tier, cap) for tier, cap in ((self.cpu, cpu_blocks), (self.ssd, ssd_blocks)) if cap]
        self.cpu_capacity = cpu_blocks
        self.ref: dict[int, int] = {}  # resident prompt blocks -> reference count
        self.released: dict[int, float] = {}  # unreferenced resident blocks -> release time (for the TTL)
        self.resident = 0  # prompt blocks + in-flight output blocks
        self.free = FreeBlocks()
        self.inflight = 0
        self.requests = 0
        self.releases = 0

    def cached_prefix(self, keys: Sequence[int]) -> int:
        ref = self.ref
        n = 0
        for k in keys:
            if k not in ref:
                break
            n += 1
        return n

    def demote(self, key: int, released: float | None) -> None:
        """Move an evicted GPU block down the lower tiers; the last one drops its least recently used block."""
        for tier, cap in self.lower:
            tier[key] = released
            tier.move_to_end(key)
            if len(tier) <= cap:
                return
            key, released = tier.popitem(last=False)

    def cpu_run(self, keys: Sequence[int], start: int) -> int:
        """End index of the run of `keys` from `start` held in the lower tiers."""
        j = start
        while j < len(keys) and any(keys[j] in tier for tier, _ in self.lower):
            j += 1
        return j


@dataclass
class _Stream:
    ewma: float = EWMA_PRIOR_S
    last_end: float | None = None


class _Totals:
    def __init__(self) -> None:
        self.requests = 0
        self.prompt_tokens = 0
        self.hit_tokens = 0
        self.cpu_loaded = 0
        self.ssd_loaded = 0
        self.miss_cold = 0.0
        self.miss_evicted = 0.0
        self.miss_routing = 0.0
        self.overflow = 0
        self.first = math.inf
        self.last = -math.inf


class _Memory:
    """Blocks held per tier, summed over replicas, integrated over time from `start` (block-seconds)."""

    def __init__(self, start: float) -> None:
        self.start = start
        self.now = start
        self.held = [0, 0, 0]  # GPU, CPU, SSD
        self.area = [0.0, 0.0, 0.0]

    def advance(self, t: float) -> None:
        if t > self.now:
            for i in range(3):
                self.area[i] += self.held[i] * (t - self.now)
            self.now = t


def simulate(
    sessions: Sequence[Session], p: SimParams, seed: int, record: dict[tuple[int, int], int] | None = None
) -> dict[str, float]:
    """Replay `sessions` under `p`; returns metrics over the measured requests. With `record`, also fills it
    with (session index, request index) -> prompt tokens reused (from the GPU or loaded from a lower tier)."""
    rng = random.Random(seed)
    noise = random.Random(seed + 1)  # separate stream: predictor error must not change the session order
    bt = p.block_tokens
    trace = p.arrivals == "trace"
    order = list(range(len(sessions)))
    if not trace:
        rng.shuffle(order)
        if len(order) <= p.concurrency:
            raise ValueError(f"need more than {p.concurrency} sessions to reach steady state, have {len(order)}")
    replicas = [
        Replica(p.capacity_tokens // bt, p.cpu_capacity_tokens // bt, p.ssd_capacity_tokens // bt)
        for _ in range(p.replicas)
    ]
    events: list[tuple[float, int, int, tuple]] = []  # (time, kind, seq, payload); kind 0 = done first
    seq = 0
    streams: dict[tuple[int, int], _Stream] = {}
    sticky: dict[int, int] = {}
    last_replica: dict[int, Replica] = {}
    rr = 0
    seen: set[int] = set()
    tot = _Totals()
    evictions = 0
    pending = iter(order)
    last_start = math.inf
    measure_from = (min((s.start for s in sessions), default=0.0) if trace else 0.0) + p.warmup_s
    mem = _Memory(measure_from)
    expiries: deque[tuple[float, Replica, int, float]] = deque()  # release order = expiry order (one TTL)

    def push(t: float, kind: int, payload: tuple) -> None:
        nonlocal seq
        seq += 1
        heapq.heappush(events, (t, kind, seq, payload))

    def start(s_idx: int, t0: float) -> None:
        sticky[s_idx] = rng.randrange(p.replicas)
        s = sessions[s_idx]
        for i, r in enumerate(s.requests):
            push(t0 + r.t, 2, (s_idx, t0, i, r))
        if not trace:
            push(t0 + s.span, 1, (s_idx,))

    def least_loaded() -> Replica:
        return min(replicas, key=lambda r: r.inflight)

    def route(s_idx: int, keys: list[int]) -> Replica:
        nonlocal rr
        if p.routing == "round_robin":
            rr += 1
            return replicas[rr % p.replicas]
        if p.routing == "least_loaded":
            return least_loaded()
        if p.routing in ("session", "sticky", "kv_aware"):
            if p.routing == "session":
                chosen: Replica | None = replicas[sticky[s_idx]]
            elif p.routing == "sticky":
                chosen = last_replica.get(s_idx)
            else:
                held = [r.cpu_run(keys, r.cached_prefix(keys)) for r in replicas]
                best = max(range(p.replicas), key=lambda i: (held[i], -replicas[i].inflight, -i))
                chosen = replicas[best] if held[best] else None
            if chosen is None or (p.max_inflight and chosen.inflight >= p.max_inflight):
                return least_loaded()
            return chosen
        most = max(r.inflight for r in replicas) + 1
        n = max(len(keys), 1)

        def prefix(r: Replica) -> int:
            gpu = r.cached_prefix(keys)
            return r.cpu_run(keys, gpu) if p.routing == "tiered_prefix_load" else gpu

        best = max(
            range(p.replicas),
            key=lambda i: (
                3 * prefix(replicas[i]) / n + 2 * (1 - replicas[i].inflight / most),
                -replicas[i].inflight,
                -i,
            ),
        )
        return replicas[best]

    def expire(until: float) -> None:
        """Drop blocks unreferenced for `ttl_s`, in time order, from whichever tier holds them."""
        while expiries and expiries[0][0] <= until:
            at, rep, key, released = expiries.popleft()
            mem.advance(max(at, mem.now))
            if rep.released.get(key) == released and rep.ref.get(key) == 0:
                rep.free.take(key)
                del rep.ref[key], rep.released[key]
                rep.resident -= 1
                mem.held[0] -= 1
            else:
                for tier, _ in rep.lower:
                    if tier.get(key, -1.0) == released:
                        del tier[key]
                        mem.held[1 + (tier is rep.ssd)] -= 1
                        break

    if trace:
        for s_idx in order:
            start(s_idx, sessions[s_idx].start)
    else:
        for s_idx in (next(pending) for _ in range(p.concurrency)):
            start(s_idx, rng.uniform(0.0, p.warmup_s))

    while events:
        now, kind, _, payload = heapq.heappop(events)
        if now >= last_start:
            break
        if p.ttl_s is not None:
            expire(now)
        mem.advance(now)
        if kind == 1:  # session finished: start the next one
            nxt = next(pending, None)
            if nxt is None:
                last_start = now
            else:
                start(nxt, now)
            continue
        if kind == 0:  # request done: release its blocks
            rep, keys, out_blocks, window, expected = payload
            rep.inflight -= 1
            rep.resident -= out_blocks
            mem.held[0] -= out_blocks
            ref, free = rep.ref, rep.free
            for k in reversed(keys):  # tail first, so tails are evicted first
                c = ref[k] - 1
                ref[k] = c
                if c == 0:
                    free.release(k, now, window, expected)
                    rep.released[k] = now
                    if p.ttl_s is not None:
                        expiries.append((now + p.ttl_s, rep, k, now))
            rep.releases += 1
            if rep.releases % 256 == 0:
                free.compact()
            continue
        s_idx, t0, r_idx, req = payload
        req: Request
        base = s_idx << 32
        keys = [base + h for h in req.blocks.tolist()]
        rep = route(s_idx, keys)
        last_replica[s_idx] = rep
        hit = rep.cached_prefix(keys)
        ref, free = rep.ref, rep.free
        for k in keys[:hit]:
            c = ref[k]
            if c == 0:
                free.take(k)
                del rep.released[k]
            ref[k] = c + 1
        loaded_end = rep.cpu_run(keys, hit)
        from_ssd = 0
        for k in keys[hit:loaded_end]:  # moves back to the GPU tier below
            if k in rep.cpu:
                del rep.cpu[k]
                mem.held[1] -= 1
            else:
                del rep.ssd[k]
                mem.held[2] -= 1
                from_ssd += 1
        missed = keys[loaded_end:]
        cold = routed = 0
        for k in missed:
            if k not in seen:
                cold += 1
                seen.add(k)
            elif any(k in other.ref or other.cpu_run([k], 0) for other in replicas if other is not rep):
                routed += 1
        out_blocks = -(-req.out_tokens // bt)
        need = len(keys) - hit + out_blocks
        overflow = False
        while rep.resident + need > rep.capacity:
            victim = free.evict(now)
            if victim is None:
                overflow = True
                break
            del ref[victim]
            before = len(rep.cpu), len(rep.ssd)
            rep.demote(victim, rep.released.pop(victim, None))
            mem.held[1] += len(rep.cpu) - before[0]
            mem.held[2] += len(rep.ssd) - before[1]
            rep.resident -= 1
            evictions += 1
            mem.held[0] -= 1
        for k in keys[hit:]:
            c = ref.get(k)
            if c is None:
                ref[k] = 1
                rep.resident += 1
                mem.held[0] += 1
            else:  # resident but past a broken prefix: recomputed, shares the slot
                if c == 0:
                    free.take(k)
                    del rep.released[k]
                ref[k] = c + 1
        rep.resident += out_blocks
        mem.held[0] += out_blocks
        rep.inflight += 1
        rep.requests += 1
        if now >= measure_from:
            hit_tokens = min(hit * bt, req.in_tokens)
            loaded_tokens = min(loaded_end * bt, req.in_tokens) - hit_tokens
            ssd_tokens = min(from_ssd * bt, loaded_tokens)
            miss_tokens = req.in_tokens - hit_tokens - loaded_tokens
            tot.requests += 1
            tot.prompt_tokens += req.in_tokens
            tot.hit_tokens += hit_tokens
            tot.cpu_loaded += loaded_tokens - ssd_tokens
            tot.ssd_loaded += ssd_tokens
            tot.first, tot.last = min(tot.first, now), max(tot.last, now)
            if missed:
                per_block = miss_tokens / len(missed)
                tot.miss_cold += cold * per_block
                tot.miss_routing += routed * per_block
                tot.miss_evicted += (len(missed) - cold - routed) * per_block
            tot.overflow += overflow
            if record is not None:
                record[(s_idx, r_idx)] = hit_tokens + loaded_tokens
        st = streams.setdefault((s_idx, req.stream), _Stream())
        if st.last_end is not None:
            st.ewma = EWMA_ALPHA * max(now - st.last_end, 0.0) + (1 - EWMA_ALPHA) * st.ewma
        end = now + req.dur
        st.last_end = end
        window, expected = _protection(p, st, end, t0 + req.next_t, noise, req.turn_end)
        push(end, 0, (rep, keys, out_blocks, window, expected))

    if tot.requests == 0:
        raise ValueError("no requests in the measurement window; lower warmup_s or raise the session count")
    end_time = last_start if math.isfinite(last_start) else mem.now
    mem.advance(end_time)
    per_replica = [r.requests for r in replicas]
    mean_req = sum(per_replica) / len(per_replica)
    prompt = tot.prompt_tokens
    loaded = tot.cpu_loaded + tot.ssd_loaded
    reusable = prompt - tot.miss_cold
    return {
        "requests_measured": float(tot.requests),
        "window_s": float(last_start - p.warmup_s) if math.isfinite(last_start) else float(tot.last - tot.first),
        "prompt_tokens_per_request": prompt / tot.requests,
        "token_hit_rate": tot.hit_tokens / prompt,
        "hit_share_of_reusable": (tot.hit_tokens + loaded) / reusable if reusable else 0.0,
        "recomputed_tokens_per_request": (prompt - tot.hit_tokens - loaded) / tot.requests,
        "cpu_loaded_share": tot.cpu_loaded / prompt,
        "ssd_loaded_share": tot.ssd_loaded / prompt,
        "prefill_cost_tokens_per_request": (
            prompt - tot.hit_tokens - (1 - p.load_cost) * tot.cpu_loaded - (1 - p.ssd_load_cost) * tot.ssd_loaded
        )
        / tot.requests,
        "miss_cold_share": tot.miss_cold / prompt,
        "miss_evicted_share": tot.miss_evicted / prompt,
        "miss_routing_share": tot.miss_routing / prompt,
        "hit_rate_upper_bound": 1 - tot.miss_cold / prompt,
        "overflow_rate": tot.overflow / tot.requests,
        "evictions_per_request": evictions / max(sum(per_replica), 1),
        "load_imbalance": max(per_replica) / mean_req if mean_req else float("nan"),
        "gpu_token_hours": mem.area[0] * bt / 3600,
        "cpu_token_hours": mem.area[1] * bt / 3600,
        "ssd_token_hours": mem.area[2] * bt / 3600,
    }


def _protection(
    p: SimParams, st: _Stream, end: float, next_arrival: float, noise: random.Random, turn_end: bool = False
) -> tuple[float, float]:
    """(protection window, expected return) for blocks this request releases at `end`."""
    if p.eviction == "lru":
        return 0.0, end
    if p.eviction == "turn":
        # Turn ended: plain LRU. Between tool calls: protected for good, and evicted oldest first among the
        # protected (the latest "expected return" goes first, so it is the negated release time).
        return (0.0, end) if turn_end else (math.inf, -end)
    if p.eviction == "ewma":
        return p.ewma_k * st.ewma, end + st.ewma
    gap = next_arrival - end
    if p.eviction == "hint":
        return (p.hint_s, end + p.hint_s) if gap <= p.hint_s else (0.0, end)
    if not (math.isfinite(gap) and gap > 0):
        return 0.0, end
    if p.eviction == "predicted":
        gap *= math.exp(noise.gauss(0.0, p.noise_sigma))
    return gap, end + gap  # oracle, or predicted with error
