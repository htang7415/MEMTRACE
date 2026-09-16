"""Gemini actor runner (Google Gemini API)."""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from memtrace.config import TEMPERATURE, TOP_P
from memtrace.backends.models.actor import ActorModel

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_RETRY_BASE_DELAY_SECONDS = 1.0


class GeminiActorModel(ActorModel):
    """Actor backed by the Gemini API.

    Unlike the local MLX/profile backends, Gemini responses are not bitwise
    reproducible even at temperature=0 -- treat results from this backend as
    a single provider sample, not a deterministic re-run.
    """

    def __init__(
        self,
        model_name: str,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        retry_base_delay: float = _DEFAULT_RETRY_BASE_DELAY_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(model_name)
        self._max_attempts = max_attempts
        self._retry_base_delay = retry_base_delay
        self._sleep = sleep
        self.last_usage: dict[str, int] | None = None
        self.last_latency_seconds: float | None = None
        self.last_retry_count = 0

        if client is not None:
            self._client = client
            return

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "Gemini actor inference requires a GEMINI_API_KEY environment variable (or an explicit api_key)."
            )
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Gemini actor inference requires the `google-genai` package") from exc
        self._client = genai.Client(api_key=resolved_key)

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        config: dict[str, Any] = {"temperature": TEMPERATURE, "top_p": TOP_P}
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens

        self.last_retry_count = 0
        last_exc: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            start = time.monotonic()
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
            except Exception as exc:
                last_exc = exc
                self.last_latency_seconds = time.monotonic() - start
                if attempt < self._max_attempts and _is_retryable(exc):
                    self.last_retry_count += 1
                    self._sleep(self._retry_base_delay * attempt)
                    continue
                raise RuntimeError(f"Gemini generation failed after {attempt} attempt(s)") from exc

            self.last_latency_seconds = time.monotonic() - start
            self.last_usage = _usage_from_response(response)
            text = getattr(response, "text", None)
            if text is None:
                raise RuntimeError("Gemini response contained no text output")
            return text.strip()

        raise RuntimeError("Gemini generation failed") from last_exc


def _is_retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if isinstance(code, int) and code in _RETRYABLE_STATUS_CODES:
        return True
    return isinstance(exc, TimeoutError)


def _usage_from_response(response: Any) -> dict[str, int] | None:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_token_count", None) or 0,
        "output_tokens": getattr(usage, "candidates_token_count", None) or 0,
        "total_tokens": getattr(usage, "total_token_count", None) or 0,
    }
