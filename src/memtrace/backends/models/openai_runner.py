"""OpenAI-compatible actor runner (local `mlx_lm.server`)."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Iterable

from memtrace.config import TEMPERATURE, TOP_P
from memtrace.backends.models.actor import ActorModel

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_DEFAULT_BASE_URL = "http://localhost:8000/v1"
_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_RETRY_BASE_DELAY_SECONDS = 1.0
_DEFAULT_TIMEOUT_SECONDS = 300.0

StreamOpener = Callable[[str, dict[str, Any], dict[str, str], float], Iterable[bytes]]


class OpenAICompatibleActorModel(ActorModel):
    """Actor backed by any server exposing the OpenAI `/v1/chat/completions` API.

    Responses are streamed so each call records serving telemetry in `last_call`:
    time to first token (TTFT), time per output token (TPOT), end-to-end latency,
    token usage (including prefix-cache hits), and retry count. `last_call` is
    thread-local so concurrent episodes sharing one actor do not overwrite each
    other's telemetry. Server-side batching means temperature=0 output is not
    guaranteed to be bitwise reproducible across load levels.
    """

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        retry_base_delay: float = _DEFAULT_RETRY_BASE_DELAY_SECONDS,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        opener: StreamOpener | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model_name)
        resolved_base_url = base_url or os.environ.get("MEMTRACE_OPENAI_BASE_URL") or _DEFAULT_BASE_URL
        self._url = resolved_base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key or os.environ.get("MEMTRACE_OPENAI_API_KEY")
        self._max_attempts = max_attempts
        self._retry_base_delay = retry_base_delay
        self._timeout = timeout
        self._open = opener or _urllib_stream
        self._sleep = sleep
        self._clock = clock
        self._local = threading.local()
        self._extra_body = dict(extra_body or {})

    @property
    def last_call(self) -> dict[str, Any] | None:
        return getattr(self._local, "last_call", None)

    @property
    def last_reasoning(self) -> str:
        """Thinking text streamed in a separate field by the last call on this thread."""
        return getattr(self._local, "last_reasoning", "")

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "stream": True,
            "stream_options": {"include_usage": True},
            **self._extra_body,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        retries = 0
        for attempt in range(1, self._max_attempts + 1):
            start = self._clock()
            try:
                return self._consume(self._open(self._url, payload, headers, self._timeout), start, retries)
            except Exception as exc:
                if attempt < self._max_attempts and _is_retryable(exc):
                    retries += 1
                    self._sleep(self._retry_base_delay * attempt)
                    continue
                raise RuntimeError(f"OpenAI-compatible generation failed after {attempt} attempt(s)") from exc
        raise AssertionError("unreachable")

    def _consume(self, lines: Iterable[bytes], start: float, retries: int) -> str:
        pieces: list[str] = []
        reasoning_pieces: list[str] = []
        first_token_at: float | None = None
        content_chunks = 0
        usage: dict[str, Any] | None = None
        for raw_line in lines:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                text = delta.get("content")
                # Servers that split out model thinking stream it as `reasoning` (mlx_lm) or
                # `reasoning_content` (vLLM reasoning parsers). It is generated output, so it
                # counts toward TTFT/TPOT, but it is not part of the returned answer.
                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                if text or reasoning:
                    if first_token_at is None:
                        first_token_at = self._clock()
                    content_chunks += 1
                if text:
                    pieces.append(text)
                if reasoning:
                    reasoning_pieces.append(reasoning)
        end = self._clock()

        output_tokens = int(usage["completion_tokens"]) if usage else content_chunks
        ttft = (first_token_at - start) if first_token_at is not None else None
        decode_seconds = (end - first_token_at) if first_token_at is not None else None
        self._local.last_call = {
            "backend": "openai",
            "ttft_seconds": ttft,
            "tpot_seconds": (decode_seconds / (output_tokens - 1)) if decode_seconds and output_tokens > 1 else None,
            "e2e_seconds": end - start,
            "prompt_tokens": int(usage["prompt_tokens"]) if usage else None,
            "cached_prompt_tokens": _cached_prompt_tokens(usage),
            "output_tokens": output_tokens,
            "output_tokens_source": "usage" if usage else "stream_chunks",
            "retries": retries,
        }
        self._local.last_reasoning = "".join(reasoning_pieces)
        return "".join(pieces).strip()


def _urllib_stream(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> Iterable[bytes]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        yield from response


def _cached_prompt_tokens(usage: dict[str, Any] | None) -> int | None:
    details = (usage or {}).get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    return int(cached) if cached is not None else None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _RETRYABLE_STATUS_CODES
    return isinstance(exc, (urllib.error.URLError, ConnectionError, TimeoutError))
