"""Retention policies (what a replica keeps) and placement policies (which replica a call goes to)."""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Retention:
    name: str
    lifetime: float = math.inf  # KV idle this long is dropped from every tier
    turn_aware: bool = False  # KV of a session whose turn ended is evicted (demoted) before in-turn KV


RETENTION = {
    "lru": Retention("lru"),  # vLLM's default: keep until space is needed, oldest first
    "ttl-5min": Retention("ttl-5min", 300.0),
    "ttl-1h": Retention("ttl-1h", 3600.0),
    "ttl-24h": Retention("ttl-24h", 86400.0),
    "session": Retention("session", turn_aware=True),  # keep through tool-call gaps, demote on user idle
}

# Placement: each router names its preferred replica (or None); a full preferred replica, or no preference,
# sends the call to the least-loaded replica.
#   least-loaded  no cache affinity
#   session-key   a stable hash of the session, like OpenAI's prompt_cache_key
#   approximate   the replica the router last sent the session to (it does not see evictions), like llm-d's
#                 approximate prefix index
#   precise       the replica that holds the session's KV in any tier, like llm-d's KV-event-fed index
ROUTERS = ("least-loaded", "session-key", "approximate", "precise")


def session_hash(session: str, replicas: int) -> int:
    return zlib.crc32(session.encode()) % replicas
