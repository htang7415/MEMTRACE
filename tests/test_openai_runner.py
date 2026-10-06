import json
import urllib.error

import pytest

from memtrace.backends.models.actor import ActorModel
from memtrace.backends.models.mlx_runner import load_actor
from memtrace.backends.models.openai_runner import OpenAICompatibleActorModel
from memtrace.core.agents.runner import _TelemetryRecorder


def _sse(*events, usage=None) -> list[bytes]:
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n".encode() for text in events]
    if usage is not None:
        lines.append(f"data: {json.dumps({'choices': [], 'usage': usage})}\n".encode())
    lines.append(b"data: [DONE]\n")
    return lines


class _FakeServer:
    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.requests: list[dict] = []

    def __call__(self, url, payload, headers, timeout):
        self.requests.append({"url": url, "payload": payload, "headers": headers})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return iter(item)


class _Clock:
    """Advances one second per reading."""

    def __init__(self) -> None:
        self.now = -1.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


def _model(server, **kwargs) -> OpenAICompatibleActorModel:
    return OpenAICompatibleActorModel(
        "Qwen/Qwen2.5-7B-Instruct",
        base_url="http://serving:8000/v1/",
        opener=server,
        sleep=lambda _: None,
        **kwargs,
    )


def test_generate_streams_text_and_records_serving_telemetry() -> None:
    usage = {"prompt_tokens": 40, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 32}}
    server = _FakeServer([_sse("hel", "lo", usage=usage)])
    model = _model(server, clock=_Clock())

    assert model.generate("prompt", max_tokens=64) == "hello"

    request = server.requests[0]
    assert request["url"] == "http://serving:8000/v1/chat/completions"
    assert request["payload"]["stream"] is True
    assert request["payload"]["max_tokens"] == 64
    assert request["payload"]["messages"] == [{"role": "user", "content": "prompt"}]
    # clock readings: start=0, first token=1, end=2; TPOT = 1s decode / (5 - 1) tokens
    assert model.last_call == {
        "backend": "openai",
        "ttft_seconds": 1.0,
        "tpot_seconds": 0.25,
        "e2e_seconds": 2.0,
        "prompt_tokens": 40,
        "cached_prompt_tokens": 32,
        "output_tokens": 5,
        "output_tokens_source": "usage",
        "retries": 0,
    }


def test_generate_falls_back_to_chunk_count_without_usage() -> None:
    model = _model(_FakeServer([_sse("a", "b", "c")]))

    assert model.generate("prompt") == "abc"
    assert model.last_call["output_tokens"] == 3
    assert model.last_call["output_tokens_source"] == "stream_chunks"
    assert model.last_call["prompt_tokens"] is None
    assert model.last_call["cached_prompt_tokens"] is None


def test_generate_retries_transient_errors_then_succeeds() -> None:
    unavailable = urllib.error.HTTPError("http://serving", 503, "busy", {}, None)
    server = _FakeServer([unavailable, _sse("ok")])
    model = _model(server)

    assert model.generate("prompt") == "ok"
    assert model.last_call["retries"] == 1
    assert len(server.requests) == 2


def test_generate_does_not_retry_client_errors() -> None:
    bad_request = urllib.error.HTTPError("http://serving", 400, "bad", {}, None)
    server = _FakeServer([bad_request])
    model = _model(server)

    with pytest.raises(RuntimeError, match="after 1 attempt"):
        model.generate("prompt")
    assert len(server.requests) == 1


def test_api_key_sent_as_bearer_token() -> None:
    server = _FakeServer([_sse("ok")])
    _model(server, api_key="secret").generate("prompt")

    assert server.requests[0]["headers"]["Authorization"] == "Bearer secret"


def test_load_actor_openai_backend_reads_base_url_from_env(monkeypatch) -> None:
    monkeypatch.setenv("MEMTRACE_OPENAI_BASE_URL", "http://localhost:9000/v1")

    model = load_actor("Qwen/Qwen2.5-7B-Instruct", backend="openai")

    assert isinstance(model, OpenAICompatibleActorModel)
    assert model._url == "http://localhost:9000/v1/chat/completions"


def test_telemetry_recorder_tags_calls_with_role_and_skips_backends_without_telemetry() -> None:
    calls: list[dict] = []
    served = _model(_FakeServer([_sse("ok")]))
    _TelemetryRecorder(served, "planner", calls).generate("prompt")

    class _Plain(ActorModel):
        def generate(self, prompt: str, max_tokens: int | None = None) -> str:
            return "plain"

    _TelemetryRecorder(_Plain("m"), "writer", calls).generate("prompt")

    assert len(calls) == 1
    assert calls[0]["role"] == "planner"
    assert calls[0]["backend"] == "openai"


def test_last_call_is_thread_local() -> None:
    import threading

    model = _model(_FakeServer([_sse("main")]))
    model.generate("prompt")
    seen = {}
    worker = threading.Thread(target=lambda: seen.setdefault("worker", model.last_call))
    worker.start()
    worker.join()

    assert model.last_call is not None
    assert seen["worker"] is None


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_separately_streamed_reasoning_counts_for_timing_but_not_answer(field) -> None:
    lines = [
        f"data: {json.dumps({'choices': [{'delta': {field: 'think'}}]})}\n".encode(),
        f"data: {json.dumps({'choices': [{'delta': {field: 'ing'}}]})}\n".encode(),
        b"data: [DONE]\n",
    ]
    model = _model(_FakeServer([lines]), clock=_Clock())

    assert model.generate("prompt") == ""
    assert model.last_reasoning == "thinking"
    # clock readings: start=0, first reasoning token=1, end=2
    assert model.last_call["ttft_seconds"] == 1.0
    assert model.last_call["output_tokens"] == 2


def test_extra_body_is_merged_into_every_request() -> None:
    server = _FakeServer([_sse("ok", usage={"prompt_tokens": 1, "completion_tokens": 1})])
    model = _model(server, extra_body={"ignore_eos": True})

    model.generate("prompt", max_tokens=8)

    assert server.requests[0]["payload"]["ignore_eos"] is True
    assert server.requests[0]["payload"]["max_tokens"] == 8


def test_stream_without_done_marker_is_an_error_not_a_short_answer() -> None:
    cut_off = [f"data: {json.dumps({'choices': [{'delta': {'content': 'Okay'}}]})}\n".encode()]
    model = _model(_FakeServer([cut_off]), max_attempts=1)

    with pytest.raises(RuntimeError) as excinfo:
        model.generate("prompt")

    assert "without [DONE]" in str(excinfo.value.__cause__)
