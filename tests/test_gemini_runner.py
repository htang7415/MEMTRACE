from types import SimpleNamespace

import pytest

from memtrace.backends.models.gemini_runner import GeminiActorModel
from memtrace.backends.models.mlx_runner import load_actor


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
    model = GeminiActorModel("gemini-2.5-flash", client=client, sleep=lambda _: None)

    assert model.generate("prompt") == "hello"
    assert model.last_usage == {"prompt_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    assert model.last_retry_count == 0
    assert model.last_latency_seconds is not None


def test_generate_passes_max_tokens_in_config() -> None:
    client = _FakeClient([_response("ok")])
    model = GeminiActorModel("gemini-2.5-flash", client=client, sleep=lambda _: None)

    model.generate("prompt", max_tokens=64)

    assert client.models.calls[0]["config"]["max_output_tokens"] == 64
    assert client.models.calls[0]["model"] == "gemini-2.5-flash"


def test_generate_omits_max_tokens_when_not_given() -> None:
    client = _FakeClient([_response("ok")])
    model = GeminiActorModel("gemini-2.5-flash", client=client, sleep=lambda _: None)

    model.generate("prompt")

    assert "max_output_tokens" not in client.models.calls[0]["config"]


def test_generate_retries_on_transient_error_then_succeeds() -> None:
    client = _FakeClient([_RetryableError("server error"), _response("recovered")])
    model = GeminiActorModel("gemini-2.5-flash", client=client, sleep=lambda _: None)

    assert model.generate("prompt") == "recovered"
    assert model.last_retry_count == 1


def test_generate_raises_after_exhausting_retries() -> None:
    client = _FakeClient([_RetryableError("a"), _RetryableError("b"), _RetryableError("c")])
    model = GeminiActorModel("gemini-2.5-flash", client=client, max_attempts=3, sleep=lambda _: None)

    with pytest.raises(RuntimeError, match="failed after 3 attempt"):
        model.generate("prompt")


def test_generate_does_not_retry_non_retryable_error() -> None:
    client = _FakeClient([_FatalError("bad request")])
    model = GeminiActorModel("gemini-2.5-flash", client=client, sleep=lambda _: None)

    with pytest.raises(RuntimeError):
        model.generate("prompt")

    assert model.last_retry_count == 0
    assert len(client.models.calls) == 1


def test_missing_api_key_raises_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiActorModel("gemini-2.5-flash")


def test_load_actor_gemini_backend_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        load_actor("gemini-2.5-flash", backend="gemini")
