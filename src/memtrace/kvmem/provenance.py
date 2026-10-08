"""Which code produced an output: recorded in every Phase 6 result so numbers can be tied to a commit.

`git_dirty` is true when `src/memtrace` differs from that commit (edits to docs or scripts do not count).
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone


def provenance() -> dict[str, object]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git(
        "status", "--porcelain", "--untracked-files=no", "--", "src/memtrace"
    )  # the code that computes results
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
