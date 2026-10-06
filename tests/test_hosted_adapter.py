import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from memtrace.serving.hosted_adapter import Accounting, make_handler, upstream_payload

PRICES = (0.30, 0.03, 2.50)


class _FakeUpstream:
    def __init__(self, lines=None, error=None):
        self.lines = lines or []
        self.error = error
        self.requests = []

    def __call__(self, request):
        self.requests.append(json.loads(request.data))
        assert request.headers["Authorization"] == "Bearer test-key"
        if self.error:
            raise self.error
        return _Response(self.lines)


class _Response(io.BytesIO):
    def __init__(self, lines):
        super().__init__(b"".join(lines))
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)


def _serve(accounting, upstream):
    handler = make_handler(
        accounting, base_url="https://hosted/v1", api_key="test-key", model="hosted-m", opener=upstream
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _post(server, body):
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=10)


def test_payload_drops_engine_only_fields_and_pins_hosted_model() -> None:
    payload = upstream_payload(
        {"model": "Qwen/Qwen3-0.6B", "messages": [], "max_tokens": 8, "ignore_eos": True, "stream": True}, "hosted-m"
    )

    assert payload == {
        "model": "hosted-m",
        "messages": [],
        "max_tokens": 8,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def test_streams_through_and_accounts_spend() -> None:
    usage = {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_tokens_details": {"cached_tokens": 400}}
    lines = [
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
        f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode(),
        b"data: [DONE]\n\n",
    ]
    accounting = Accounting(1.0, PRICES)
    server = _serve(accounting, _FakeUpstream(lines))
    try:
        with _post(server, {"messages": [], "stream": True, "ignore_eos": True}) as response:
            body = response.read()
    finally:
        server.shutdown()

    assert b"[DONE]" in body and b'"hi"' in body
    assert accounting.running == 0 and accounting.succeeded == 1
    # 600 uncached x $0.30 + 400 cached x $0.03 + 100 output x $2.50, per 1M tokens
    assert accounting.spent_usd == pytest.approx((600 * 0.30 + 400 * 0.03 + 100 * 2.50) / 1e6)


def test_spend_cap_returns_503_without_calling_upstream() -> None:
    accounting = Accounting(0.0, PRICES)
    upstream = _FakeUpstream()
    server = _serve(accounting, upstream)
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post(server, {"messages": []})
    finally:
        server.shutdown()

    assert excinfo.value.code == 503
    assert upstream.requests == []


def test_upstream_error_is_relayed_and_counted() -> None:
    error = urllib.error.HTTPError("https://hosted", 429, "rate limited", {}, io.BytesIO(b'{"error":"quota"}'))
    accounting = Accounting(1.0, PRICES)
    server = _serve(accounting, _FakeUpstream(error=error))
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post(server, {"messages": []})
    finally:
        server.shutdown()

    assert excinfo.value.code == 429
    assert accounting.failed == 1 and accounting.running == 0


def test_metrics_expose_vllm_names_the_epp_reads() -> None:
    accounting = Accounting(2.0, PRICES)
    assert accounting.start()
    text = accounting.metrics("hosted-m")

    assert 'vllm:num_requests_running{model_name="hosted-m"} 1' in text
    assert 'vllm:num_requests_waiting{model_name="hosted-m"} 0' in text
    assert "memtrace_hosted_budget_usd" in text


def test_chunked_request_body_is_forwarded_not_dropped() -> None:
    # Envoy forwards ext_proc-processed bodies with Transfer-Encoding: chunked and no Content-Length.
    import http.client

    lines = [b"data: [DONE]\n\n"]
    upstream = _FakeUpstream(lines)
    server = _serve(Accounting(1.0, PRICES), upstream)
    try:
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": True}).encode()
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        conn.request("POST", "/v1/chat/completions", body=iter([body[:10], body[10:]]), encode_chunked=True)
        response = conn.getresponse()
        response.read()
    finally:
        server.shutdown()

    assert response.status == 200
    assert upstream.requests[0]["messages"] == [{"role": "user", "content": "hi"}]
