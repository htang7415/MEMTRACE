from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from memtrace.datasets import sources

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
    sources.download_file(url="https://example.invalid/f", dest=tmp_path / "f")
    assert hashlib.sha256((tmp_path / "f").read_bytes()).hexdigest() == hashlib.sha256(PAYLOAD).hexdigest()


def test_download_gives_up_after_the_last_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", _server([TimeoutError()] * 3))
    with pytest.raises(OSError, match="incomplete after 3 attempts"):
        sources.download_file(url="https://example.invalid/f", dest=tmp_path / "f", attempts=3)
    assert not (tmp_path / "f").exists()
