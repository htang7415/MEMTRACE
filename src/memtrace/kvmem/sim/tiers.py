"""Memory tiers of one replica: GPU first, then slower tiers that evicted KV is demoted to."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class Tier:
    name: str
    capacity_gb: float  # per replica
    load_gb_per_s: float  # bandwidth when a hit here is loaded back into the GPU (unused for the GPU tier)


@dataclass(eq=False)
class Entry:
    tokens: int
    last_use: float  # end of the call that wrote it
    model: str
    prompt: int  # that call's prompt length: the next call can reuse at most this much (as in M1)
    where: tuple[int, int] | None = None  # (replica, tier) while stored


class TierStore:
    """Cached session prefixes in one tier, keyed by session, oldest first. Entries of sessions whose turn has
    ended sit in `idle`, which is evicted before `active`; policies that are not turn-aware use `active` only."""

    def __init__(self, tier: Tier, kv_bytes_per_token: int) -> None:
        self.tier = tier
        self.capacity_tokens = tier.capacity_gb * 1e9 / kv_bytes_per_token  # may be infinite
        self.active: OrderedDict[int, Entry] = OrderedDict()
        self.idle: OrderedDict[int, Entry] = OrderedDict()
        self.tokens = 0

    def add(self, session: int, entry: Entry, idle: bool) -> None:
        (self.idle if idle else self.active)[session] = entry
        self.tokens += entry.tokens

    def get(self, session: int) -> Entry | None:
        return self.active.get(session) or self.idle.get(session)

    def pop(self, session: int) -> Entry | None:
        entry = self.active.pop(session, None) or self.idle.pop(session, None)
        if entry:
            self.tokens -= entry.tokens
        return entry

    def victim(self) -> tuple[int, Entry, bool]:
        """Remove and return the next entry to evict: the oldest idle one, else the oldest active one."""
        store = self.idle if self.idle else self.active
        session, entry = store.popitem(last=False)
        self.tokens -= entry.tokens
        return session, entry, store is self.idle
