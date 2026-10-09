from types import SimpleNamespace

import pytest

from memtrace.memrisk.backends.models.gemini_runner import (
    GeminiActorModel,
    GeminiBudgetExceeded,
    GeminiSpendGuard,
    gemini_cost_usd,
)
from memtrace.memrisk.backends.models.mlx_runner import load_actor


class _RetryableError(Exception):
    code = 503


class _FatalError(Exception):
    code = 400


class _FakeModels:
    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeClient:
    def __init__(self, responses: list) -> None:
        self.models = _FakeModels(responses)


def _response(text: str, prompt_tokens: int = 10, output_tokens: int = 5) -> SimpleNamespace:
    usage = SimpleNamespace(
        prompt_token_count=prompt_tokens,
        candidates_token_count=output_tokens,
        total_token_count=prompt_tokens + output_tokens,
    )
    return SimpleNamespace(text=text, usage_metadata=usage)


def test_generate_returns_text_and_records_usage() -> None:
    client = _FakeClient([_response("hello")])
    model = GeminiActorModel("gemini-3.5-flash-lite", client=client, sleep=lambda _: None)

    assert model.generate("prompt") == "hello"
    assert model.last_usage == {
        "prompt_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "cached_tokens": 0,
        "thinking_tokens": 0,
    }
    assert model.last_retry_count == 0
    assert model.last_latency_seconds is not None


def test_generate_passes_max_tokens_in_config() -> None:
    client = _FakeClient([_response("ok")])
    model = GeminiActorModel("gemini-3.5-flash-lite", client=client, sleep=lambda _: None)

    model.generate("prompt", max_tokens=64)

    assert client.models.calls[0]["config"]["max_output_tokens"] == 64
    assert client.models.calls[0]["model"] == "gemini-3.5-flash-lite"


def test_generate_omits_max_tokens_when_not_given() -> None:
    client = _FakeClient([_response("ok")])
    model = GeminiActorModel("gemini-3.5-flash-lite", client=client, sleep=lambda _: None)

    model.generate("prompt")

    assert "max_output_tokens" not in client.models.calls[0]["config"]


def test_generate_retries_on_transient_error_then_succeeds() -> None:
    client = _FakeClient([_RetryableError("server error"), _response("recovered")])
    model = GeminiActorModel("gemini-3.5-flash-lite", client=client, sleep=lambda _: None)

    assert model.generate("prompt") == "recovered"
    assert model.last_retry_count == 1


def test_generate_raises_after_exhausting_retries() -> None:
    client = _FakeClient([_RetryableError("a"), _RetryableError("b"), _RetryableError("c")])
    model = GeminiActorModel("gemini-3.5-flash-lite", client=client, max_attempts=3, sleep=lambda _: None)

    with pytest.raises(RuntimeError, match="failed after 3 attempt"):
        model.generate("prompt")


def test_generate_does_not_retry_non_retryable_error() -> None:
    client = _FakeClient([_FatalError("bad request")])
    model = GeminiActorModel("gemini-3.5-flash-lite", client=client, sleep=lambda _: None)

    with pytest.raises(RuntimeError):
        model.generate("prompt")

    assert model.last_retry_count == 0
    assert len(client.models.calls) == 1


def test_missing_api_key_raises_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiActorModel("gemini-3.5-flash-lite")


def test_load_actor_gemini_backend_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        load_actor("gemini-3.5-flash-lite", backend="gemini")


def test_cost_prices_cached_and_thinking_tokens() -> None:
    usage = {"prompt_tokens": 1_000_000, "cached_tokens": 400_000, "output_tokens": 100_000, "thinking_tokens": 100_000}

    # 600k uncached * $0.30 + 400k cached * $0.03 + 200k output * $2.50, per 1M tokens
    assert gemini_cost_usd("models/gemini-3.5-flash-lite", usage) == pytest.approx(0.18 + 0.012 + 0.5)


def test_unpriced_model_is_refused() -> None:
    with pytest.raises(RuntimeError, match="No price"):
        GeminiActorModel("gemini-unknown", client=_FakeClient([]), guard=GeminiSpendGuard(1.0))


def test_spend_guard_blocks_calls_after_cap() -> None:
    guard = GeminiSpendGuard(budget_usd=0.0001)
    client = _FakeClient([_response("first", prompt_tokens=1000, output_tokens=100), _response("second")])
    model = GeminiActorModel("gemini-3.5-flash-lite", client=client, sleep=lambda _: None, guard=guard)

    assert model.generate("prompt") == "first"
    assert model.last_cost_usd == pytest.approx((1000 * 0.30 + 100 * 2.50) / 1_000_000)
    with pytest.raises(GeminiBudgetExceeded):
        model.generate("prompt")
    assert len(client.models.calls) == 1
