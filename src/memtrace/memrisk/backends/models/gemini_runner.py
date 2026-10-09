"""Gemini actor runner (Google Gemini API)."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable

from memtrace.memrisk.config import TEMPERATURE, TOP_P
from memtrace.memrisk.backends.models.actor import ActorModel

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_RETRY_BASE_DELAY_SECONDS = 1.0
_DEFAULT_BUDGET_USD = 1.0

# USD per 1M tokens: (input, cached input, output). Source: docs/gpu_llm_api_pricing.md.
# Thinking tokens bill as output. A model missing here cannot be budgeted, so it is refused.
GEMINI_PRICES_USD_PER_MTOK: dict[str, tuple[float, float, float]] = {
    "gemini-3.5-flash-lite": (0.30, 0.03, 2.50),
    "gemini-3.5-flash": (1.50, 0.15, 9.00),
}


class GeminiBudgetExceeded(RuntimeError):
    """Raised before a call once the process-wide Gemini spend cap is reached."""


class GeminiSpendGuard:
    """Process-wide Gemini spend tracker with a hard cap, shared by all actors."""

    def __init__(self, budget_usd: float) -> None:
        self.budget_usd = budget_usd
        self.spent_usd = 0.0
        self._lock = threading.Lock()

    def check(self) -> None:
        with self._lock:
            if self.spent_usd >= self.budget_usd:
                raise GeminiBudgetExceeded(
                    f"Gemini spend ${self.spent_usd:.4f} reached the ${self.budget_usd:.2f} cap "
                    "(MEMTRACE_GEMINI_BUDGET_USD)"
                )

    def record(self, model_name: str, usage: dict[str, int] | None) -> float:
        if not usage:
            return 0.0
        cost = gemini_cost_usd(model_name, usage)
        with self._lock:
            self.spent_usd += cost
        return cost


_spend_guard: GeminiSpendGuard | None = None
_spend_guard_lock = threading.Lock()


def spend_guard() -> GeminiSpendGuard:
    global _spend_guard
    with _spend_guard_lock:
        if _spend_guard is None:
            _spend_guard = GeminiSpendGuard(float(os.environ.get("MEMTRACE_GEMINI_BUDGET_USD", _DEFAULT_BUDGET_USD)))
        return _spend_guard


def gemini_cost_usd(model_name: str, usage: dict[str, int]) -> float:
    input_price, cached_price, output_price = _prices(model_name)
    cached = usage.get("cached_tokens", 0)
    uncached = max(usage.get("prompt_tokens", 0) - cached, 0)
    output = usage.get("output_tokens", 0) + usage.get("thinking_tokens", 0)
    return (uncached * input_price + cached * cached_price + output * output_price) / 1_000_000


def _prices(model_name: str) -> tuple[float, float, float]:
    name = model_name.removeprefix("models/")
    if name not in GEMINI_PRICES_USD_PER_MTOK:
        raise RuntimeError(
            f"No price for Gemini model {model_name!r}; add it to GEMINI_PRICES_USD_PER_MTOK "
            "so the spend cap can be enforced."
        )
    return GEMINI_PRICES_USD_PER_MTOK[name]


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
        guard: GeminiSpendGuard | None = None,
    ) -> None:
        super().__init__(model_name)
        _prices(model_name)
        self._guard = guard if guard is not None else spend_guard()
        self._max_attempts = max_attempts
        self._retry_base_delay = retry_base_delay
        self._sleep = sleep
        self.last_usage: dict[str, int] | None = None
        self.last_cost_usd: float | None = None
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
            self._guard.check()
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
            self.last_cost_usd = self._guard.record(self.model_name, self.last_usage)
            text = getattr(response, "text", None)
            if text is None:
                raise RuntimeError("Gemini response contained no text output")
            return str(text).strip()

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
        "cached_tokens": getattr(usage, "cached_content_token_count", None) or 0,
        "thinking_tokens": getattr(usage, "thoughts_token_count", None) or 0,
    }
