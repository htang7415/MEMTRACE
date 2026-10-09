from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from memtrace.datasets import public as fetch

PAYLOAD = b"x" * 1000


class _Response(io.BytesIO):
    def __init__(self, body: bytes, status: int, length: int) -> None:
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Length": str(length)}


def _server(script: list[Any]) -> Any:
    """Each urlopen call takes the next step: an exception to raise, or how many bytes to send before dropping."""

    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        step = script.pop(0)
        if isinstance(step, Exception):
            raise step
        offset = int((request.get_header("Range") or "bytes=0-")[6:-1])
        return _Response(PAYLOAD[offset : offset + step], 206 if offset else 200, len(PAYLOAD) - offset)

    return urlopen


def test_download_retries_a_failed_first_connection_and_resumes_a_cut_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", _server([TimeoutError("first connect"), 400, 10_000]))
    size, sha = fetch._download("https://example.invalid/f", tmp_path / "f")
    assert (size, sha) == (1000, hashlib.sha256(PAYLOAD).hexdigest())
    assert (tmp_path / "f").read_bytes() == PAYLOAD


def test_download_gives_up_after_the_last_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", _server([TimeoutError()] * 3))
    with pytest.raises(OSError, match="incomplete after 3 attempts"):
        fetch._download("https://example.invalid/f", tmp_path / "f", attempts=3)
    assert not (tmp_path / "f").exists()
