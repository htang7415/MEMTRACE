"""Closed-loop engine benchmark: replay a workload at fixed concurrency and record telemetry."""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from memtrace.backends.models.openai_runner import OpenAICompatibleActorModel
from memtrace.serving.workloads import Request

_PREFIX_METRICS = ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total")


def run_requests(
    model: OpenAICompatibleActorModel, requests: list[Request], concurrency: int
) -> tuple[list[dict[str, Any]], float]:
    """Send every request with at most `concurrency` in flight; return per-request telemetry and wall time."""

    def one(index: int, request: Request) -> dict[str, Any]:
        try:
            text = model.generate(request.prompt, max_tokens=request.max_tokens)
            return {
                "index": index,
                "ok": True,
                **(model.last_call or {}),
                "output_text": text,
                "reasoning_text": model.last_reasoning,
            }
        except RuntimeError as exc:
            return {"index": index, "ok": False, "error": str(exc.__cause__ or exc)}

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        records = list(pool.map(one, range(len(requests)), requests))
    return records, time.monotonic() - started


def scrape_prefix_cache(base_url: str) -> dict[str, float] | None:
    """Sum vLLM prefix-cache counters from the engine's Prometheus endpoint, or None if it has none."""
    metrics_url = re.sub(r"/v1/?$", "", base_url.rstrip("/")) + "/metrics"
    try:
        with urllib.request.urlopen(metrics_url, timeout=5) as response:
            text = response.read().decode("utf-8")
    except OSError:
        return None
    return _sum_counters(text, _PREFIX_METRICS)


def scrape_pool(context: str, selector: str, namespace: str = "default") -> dict[str, dict[str, float]]:
    """Per-pod vLLM counters for an InferencePool's pods, read through the Kubernetes API proxy."""
    pods = subprocess.run(
        [
            "kubectl",
            "--context",
            context,
            "-n",
            namespace,
            "get",
            "pods",
            "-l",
            selector,
            "-o",
            "jsonpath={.items[*].metadata.name}",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    result = {}
    for pod in pods:
        raw = subprocess.run(
            [
                "kubectl",
                "--context",
                context,
                "get",
                "--raw",
                f"/api/v1/namespaces/{namespace}/pods/{pod}:8000/proxy/metrics",
            ],
            capture_output=True,
            text=True,
        ).stdout
        totals = _sum_counters(raw, (*_PREFIX_METRICS, "vllm:request_success_total"))
        result[pod] = totals
    return result


def pool_delta(before: dict[str, dict[str, float]], after: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Requests served and prefix-cache hit rate per pod, plus load imbalance (max/mean requests)."""
    pods = {}
    for pod, end in after.items():
        start = before.get(pod, {})
        delta = {name: end[name] - start.get(name, 0.0) for name in end}
        queries = delta["vllm:prefix_cache_queries_total"]
        pods[pod] = {
            "requests": delta["vllm:request_success_total"],
            "prefix_hit_rate": delta["vllm:prefix_cache_hits_total"] / queries if queries else None,
        }
    requests = [p["requests"] for p in pods.values()]
    queries = sum(
        after[p]["vllm:prefix_cache_queries_total"] - before.get(p, {}).get("vllm:prefix_cache_queries_total", 0.0)
        for p in after
    )
    hits = sum(
        after[p]["vllm:prefix_cache_hits_total"] - before.get(p, {}).get("vllm:prefix_cache_hits_total", 0.0)
        for p in after
    )
    mean = sum(requests) / len(requests) if requests else 0.0
    return {
        "pods": pods,
        "pool_prefix_hit_rate": hits / queries if queries else None,
        "load_imbalance": max(requests) / mean if mean else None,
    }


def _sum_counters(text: str, names: tuple[str, ...]) -> dict[str, float]:
    """Sum Prometheus counters across label sets. Names without the `_total` suffix (older vLLM,
    llm-d-inference-sim) count toward the `_total` name."""
    totals = {name: 0.0 for name in names}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        metric = re.split(r"[{ ]", line, maxsplit=1)[0]
        name = metric if metric in totals else metric + "_total"
        if name in totals:
            totals[name] += float(line.rsplit(" ", 1)[1])
    return totals


class PowerSampler:
    """Background `macmon pipe` reader: power (W) and RAM use, sampled every `interval_ms`.

    `sys_w` is whole-system power and is the basis for energy figures. On macOS 27 / M4,
    macmon's per-cluster `cpu_power` reads 0 even under full CPU load, so CPU-only engines
    are visible only through `sys_w`; `cpu_w` is kept to show that gap, not to be used.
    """

    def __init__(self, interval_ms: int = 250, command: str = "macmon", max_plausible_w: float = 100.0) -> None:
        # macmon's sys_power occasionally reports impossible values under load (run means of
        # 200+ W on a Mac mini M4 whose rated maximum is about 65 W); samples above
        # `max_plausible_w` are dropped and counted rather than averaged in.
        self.max_plausible_w = max_plausible_w
        self._command = [command, "pipe", "-i", str(interval_ms)]
        self._interval = interval_ms / 1000
        self.samples: list[dict[str, float]] = []
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> PowerSampler:
        try:
            self._process = subprocess.Popen(
                self._command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
        except FileNotFoundError:
            return self
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.samples.append(
                {
                    "cpu_w": sample.get("cpu_power", 0.0),
                    "gpu_w": sample.get("gpu_power", 0.0),
                    "ane_w": sample.get("ane_power", 0.0),
                    "sys_w": sample.get("sys_power", 0.0),
                    "ram_bytes": (sample.get("memory") or {}).get("ram_usage", 0),
                    "swap_bytes": (sample.get("memory") or {}).get("swap_usage", 0),
                }
            )

    def __exit__(self, *exc: object) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summary(self) -> dict[str, float] | None:
        valid = [s for s in self.samples if 0 < s["sys_w"] <= self.max_plausible_w]
        if not valid:
            return None
        n = len(valid)
        return {
            "samples": n,
            "dropped_implausible_samples": len(self.samples) - n,
            "mean_cpu_w": sum(s["cpu_w"] for s in valid) / n,
            "mean_gpu_w": sum(s["gpu_w"] for s in valid) / n,
            "mean_ane_w": sum(s["ane_w"] for s in valid) / n,
            "mean_sys_w": sum(s["sys_w"] for s in valid) / n,
            "max_sys_w": max(s["sys_w"] for s in valid),
            "peak_ram_gb": max(s["ram_bytes"] for s in self.samples) / 1e9,
            "peak_swap_gb": max(s["swap_bytes"] for s in self.samples) / 1e9,
        }
