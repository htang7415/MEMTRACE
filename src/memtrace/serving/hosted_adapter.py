"""Hosted-model adapter: present a hosted OpenAI-compatible API (Gemini) as an llm-d InferencePool endpoint.

The llm-d EPP routes only to pods and reads vLLM Prometheus metrics from them. This adapter runs as a pod,
forwards `/v1/chat/completions` to the hosted API with the key from the environment, and exports the
metrics the EPP scorers read (`vllm:num_requests_running`, `vllm:num_requests_waiting`, ...), so a hosted
model can join the pool as an overflow tier. It also enforces a hard spend cap and exports spend.

Standard library only: it runs in a stock `python:3.12-slim` container from a ConfigMap
(deploy/kind/gemini-adapter.yaml), without installing MEMTRACE.

Environment: HOSTED_BASE_URL, HOSTED_API_KEY, HOSTED_MODEL, PRICE_INPUT/PRICE_CACHED/PRICE_OUTPUT (USD per
1M tokens), BUDGET_USD, PORT.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterable

# Request fields the hosted API accepts; everything else (e.g. vLLM's `ignore_eos`) is dropped.
FORWARDED_FIELDS = (
    "messages",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "stream",
    "stream_options",
    "tools",
    "tool_choice",
    "stop",
)


class Accounting:
    """In-flight requests, completed requests, and spend; thread-safe."""

    def __init__(self, budget_usd: float, prices_per_mtok: tuple[float, float, float]) -> None:
        self.budget_usd = budget_usd
        self.prices = prices_per_mtok
        self.running = 0
        self.succeeded = 0
        self.failed = 0
        self.spent_usd = 0.0
        self.prompt_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()

    def start(self) -> bool:
        with self._lock:
            if self.spent_usd >= self.budget_usd:
                return False
            self.running += 1
            return True

    def finish(self, usage: dict[str, Any] | None, ok: bool) -> None:
        with self._lock:
            self.running -= 1
            if ok:
                self.succeeded += 1
            else:
                self.failed += 1
            if usage:
                prompt = int(usage.get("prompt_tokens") or 0)
                cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
                output = int(usage.get("completion_tokens") or 0)
                price_in, price_cached, price_out = self.prices
                self.spent_usd += ((prompt - cached) * price_in + cached * price_cached + output * price_out) / 1e6
                self.prompt_tokens += prompt
                self.output_tokens += output

    def metrics(self, model: str) -> str:
        labels = f'{{model_name="{model}"}}'
        with self._lock:
            return "".join(
                [
                    f"vllm:num_requests_running{labels} {self.running}\n",
                    f"vllm:num_requests_waiting{labels} 0\n",
                    f"vllm:kv_cache_usage_perc{labels} 0\n",
                    f"vllm:prefix_cache_queries_total{labels} 0\n",
                    f"vllm:prefix_cache_hits_total{labels} 0\n",
                    f'vllm:request_success_total{{model_name="{model}",finished_reason="stop"}} {self.succeeded}\n',
                    f"memtrace_hosted_failed_total{labels} {self.failed}\n",
                    f"memtrace_hosted_spend_usd{labels} {self.spent_usd:.6f}\n",
                    f"memtrace_hosted_budget_usd{labels} {self.budget_usd:.2f}\n",
                    f"memtrace_hosted_prompt_tokens_total{labels} {self.prompt_tokens}\n",
                    f"memtrace_hosted_output_tokens_total{labels} {self.output_tokens}\n",
                ]
            )


def upstream_payload(body: dict[str, Any], model: str) -> dict[str, Any]:
    """Keep only fields the hosted API accepts and pin the hosted model."""
    payload = {key: body[key] for key in FORWARDED_FIELDS if key in body}
    payload["model"] = model
    if payload.get("stream"):
        payload["stream_options"] = {**(payload.get("stream_options") or {}), "include_usage": True}
    return payload


def usage_from_stream_line(line: bytes) -> dict[str, Any] | None:
    text = line.decode("utf-8", "replace").strip()
    if not text.startswith("data:"):
        return None
    data = text[5:].strip()
    if not data or data == "[DONE]":
        return None
    try:
        return json.loads(data).get("usage") or None
    except json.JSONDecodeError:
        return None


Opener = Callable[[urllib.request.Request], Any]


def make_handler(
    accounting: Accounting, *, base_url: str, api_key: str, model: str, opener: Opener = urllib.request.urlopen
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            if self.path == "/metrics":
                self._send(200, accounting.metrics(model).encode(), "text/plain; version=0.0.4")
            elif self.path == "/health":
                self._send(200, b"ok", "text/plain")
            elif self.path == "/v1/models":
                self._send(200, json.dumps({"object": "list", "data": [{"id": model, "object": "model"}]}).encode())
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if self.path not in ("/v1/chat/completions", "/reset_prefix_cache"):
                self._send(404, b"not found", "text/plain")
                return
            body = json.loads(self._read_body() or b"{}")
            if self.path == "/reset_prefix_cache":  # hosted cache is not ours to reset
                self._send(200, b"", "text/plain")
                return
            if not accounting.start():
                self._send(503, json.dumps({"error": "hosted spend cap reached"}).encode())
                return
            usage, ok = None, False
            try:
                request = urllib.request.Request(
                    base_url.rstrip("/") + "/chat/completions",
                    data=json.dumps(upstream_payload(body, model)).encode(),
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                    method="POST",
                )
                with opener(request) as upstream:
                    if body.get("stream"):
                        usage = self._relay_stream(upstream)
                    else:
                        raw = upstream.read()
                        usage = json.loads(raw).get("usage")
                        self._send(200, raw)
                ok = True
            except urllib.error.HTTPError as exc:
                self._send(exc.code, exc.read() or b"{}")
            except OSError as exc:
                self._send(502, json.dumps({"error": f"upstream: {exc}"}).encode())
            finally:
                accounting.finish(usage, ok)

        def _read_body(self) -> bytes:
            # Envoy forwards bodies it has processed (ext_proc) with chunked transfer encoding and no
            # Content-Length; reading only Content-Length bytes would forward an empty request.
            if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                chunks = []
                while True:
                    size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                    if size == 0:
                        self.rfile.readline()  # blank line after the last chunk (no trailers expected)
                        return b"".join(chunks)
                    chunks.append(self.rfile.read(size))
                    self.rfile.readline()  # CRLF after each chunk
            return self.rfile.read(int(self.headers.get("Content-Length") or 0))

        def _relay_stream(self, upstream: Iterable[bytes]) -> dict[str, Any] | None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            usage = None
            for line in upstream:
                usage = usage_from_stream_line(line) or usage
                self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            return usage

        def _send(self, status: int, payload: bytes, content_type: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: Any) -> None:
            pass

    return Handler


def main() -> None:
    prices = (float(os.environ["PRICE_INPUT"]), float(os.environ["PRICE_CACHED"]), float(os.environ["PRICE_OUTPUT"]))
    accounting = Accounting(float(os.environ.get("BUDGET_USD", "1")), prices)
    handler = make_handler(
        accounting,
        base_url=os.environ["HOSTED_BASE_URL"],
        api_key=os.environ["HOSTED_API_KEY"].strip(),  # Secret created from a file may end in a newline
        model=os.environ["HOSTED_MODEL"],
    )
    ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8000"))), handler).serve_forever()


if __name__ == "__main__":
    main()
