"""Targets run by the repository's shell tooling: single engines (scripts/engines.sh) and llm-d on kind
(scripts/stack.sh: the Kubernetes InferencePool with the custom capacity scorers, the GPU + CPU pool, and
optional hosted overflow). The scripts stay the one place that knows the engine command lines and the
cluster; these targets call them and read the pool's per-pod vLLM counters."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from memtrace.harness.pickers import PICKERS, EndpointPicker
from memtrace.harness.targets import Target, _check_keys

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE_PORTS = {"vllm-metal": 8200, "vllm-cpu": 8100, "mlx-lm": 8300}  # as scripts/engines.sh starts them
GATEWAY_URL = "http://localhost:30080"
CONTEXT = "kind-memtrace"
POOL_SELECTOR = "app=qwen3-0-6b-inference-pool"
_POOL_COUNTERS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:request_success_total",
    "memtrace_hosted_spend_usd",
)


def _bash(script: str, env: Mapping[str, str] | None = None) -> None:
    subprocess.run(["bash", "-c", script], cwd=REPO_ROOT, env={**os.environ, **(env or {})}, check=True)


class Engine(Target):
    """One engine started and stopped by scripts/engines.sh (vllm-metal, vLLM CPU in Docker, mlx_lm.server)."""

    kind = "engine"

    def __init__(self, params: Mapping[str, Any]) -> None:
        _check_keys("engine", params, {"engine", "model"})
        self.engine = str(params["engine"])
        if self.engine not in ENGINE_PORTS:
            raise ValueError(f"engine must be one of {sorted(ENGINE_PORTS)}")
        self.model = str(params.get("model", "Qwen/Qwen3-0.6B"))
        self.base_urls = [f"http://127.0.0.1:{ENGINE_PORTS[self.engine]}"]

    def __enter__(self) -> "Engine":
        _bash(f"source scripts/engines.sh && start_engine {self.engine}", {"MODEL": self.model})
        return self

    def __exit__(self, *exc: object) -> None:
        _bash(f"source scripts/engines.sh && stop_engine {self.engine}", {"MODEL": self.model})

    def picker(self) -> EndpointPicker:
        return PICKERS["round_robin"](1)

    def request_options(self) -> dict[str, Any]:
        return {"extra_body": {"model": self.model}}

    def describe(self) -> dict[str, Any]:
        return {"kind": "engine", "engine": self.engine, "model": self.model, "url": self.base_urls[0]}


class KindLlmd(Target):
    """llm-d on the kind cluster, reached through its gateway. The cluster itself comes from `make up`.

    Before a trial the pool is (re)built when its setup differs from the last trial's in this process (`pool`:
    `sims` with `replicas` simulators, or `hetero`: host vllm-metal + CPU simulator, with `hosted` Gemini
    capacity, `overlay`, and `kv_events`), then the EPP `policy` is applied (a cold restart of EPP and pool)
    and, by default, engine caches are emptied, as the study scripts did before every run."""

    kind = "llmd_kind"
    prepared: str | None = None  # setup applied by this process, shared across trials

    def __init__(self, params: Mapping[str, Any]) -> None:
        _check_keys(
            "llmd_kind",
            params,
            {"pool", "replicas", "policy", "hosted", "overlay", "kv_events", "reset_caches", "sim_args"},
        )
        self.pool = str(params.get("pool", "hetero"))
        if self.pool not in ("sims", "hetero"):
            raise ValueError("llmd_kind.pool must be sims or hetero")
        self.replicas = int(params.get("replicas", 4))
        self.policy = str(params["policy"])
        self.hosted = int(params.get("hosted", 0))
        self.overlay = str(params.get("overlay", "hetero"))
        self.kv_events = bool(params.get("kv_events", False))
        self.reset_caches = bool(params.get("reset_caches", self.pool == "hetero"))
        self.sim_args = params.get("sim_args")
        self.model = "Qwen/Qwen3-4B" if self.overlay == "qwen3-4b" else "Qwen/Qwen3-0.6B"
        self.base_urls = [GATEWAY_URL]
        self.env = {"OVERLAY": self.overlay, **({"MODEL_4B": "1"} if self.overlay == "qwen3-4b" else {})}
        if self.sim_args:
            self.env["SIM_ARGS"] = str(self.sim_args)
        self.before: dict[str, dict[str, float]] = {}

    def setup_steps(self) -> list[str]:
        if self.pool == "sims":
            return [f"sims {self.replicas}"]
        steps = ["wait_gpu_free"]
        if self.overlay == "precise":
            steps.append("render_up")
        steps += ["hetero", "capacity_epp"]
        if self.kv_events:
            steps.append("kv_events")
        steps.append(f"hosted {self.hosted}" if self.hosted else "hosted_down")
        return steps

    def trial_steps(self) -> list[str]:
        steps = [f"policy {self.policy}"]
        if self.reset_caches:
            steps.append("reset_caches")
        if self.kv_events:
            steps.append("kv_events_forward")  # the cache reset drops the EPP's event subscriber
        return steps

    def __enter__(self) -> "KindLlmd":
        signature = json.dumps([self.setup_steps(), self.env], sort_keys=True)
        steps = self.trial_steps() if KindLlmd.prepared == signature else self.setup_steps() + self.trial_steps()
        for step in steps:
            _bash(f"scripts/stack.sh {step}", self.env)
        KindLlmd.prepared = signature
        self.before = scrape_pool(CONTEXT, POOL_SELECTOR)
        return self

    def picker(self) -> EndpointPicker:
        return PICKERS["round_robin"](1)

    def request_options(self) -> dict[str, Any]:
        return {"extra_body": {"model": self.model}}

    def describe(self) -> dict[str, Any]:
        weight = REPO_ROOT / "data" / "hetero" / "gpu_weight.txt"
        return {
            "kind": "llmd_kind",
            "pool": self.pool,
            "replicas": self.replicas if self.pool == "sims" else None,
            "policy": self.policy,
            "hosted": self.hosted,
            "overlay": self.overlay,
            "model": self.model,
            "gpu_weight": weight.read_text().strip() if self.pool == "hetero" and weight.exists() else None,
        }

    def collect(self) -> dict[str, Any]:
        return pool_delta(self.before, scrape_pool(CONTEXT, POOL_SELECTOR))


def scrape_pool(context: str, selector: str, namespace: str = "default") -> dict[str, dict[str, float]]:
    """Per-pod vLLM counters for an InferencePool's pods, read through the Kubernetes API proxy."""
    kubectl = ["kubectl", "--context", context]
    pods = subprocess.run(
        [*kubectl, "-n", namespace, "get", "pods", "-l", selector, "-o", "jsonpath={.items[*].metadata.name}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    result = {}
    for pod in pods:
        raw = subprocess.run(
            [*kubectl, "get", "--raw", f"/api/v1/namespaces/{namespace}/pods/{pod}:8000/proxy/metrics"],
            capture_output=True,
            text=True,
        ).stdout
        result[pod] = sum_counters(raw, _POOL_COUNTERS)
    return result


def pool_delta(before: dict[str, dict[str, float]], after: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Requests served and prefix-cache hit rate per pod, plus load imbalance (max/mean requests)."""
    pods: dict[str, dict[str, float | None]] = {}
    served: list[float] = []
    queries = hits = 0.0
    for pod, end in after.items():
        start = before.get(pod, {})
        delta = {name: end[name] - start.get(name, 0.0) for name in end}
        pod_queries = delta["vllm:prefix_cache_queries_total"]
        queries += pod_queries
        hits += delta["vllm:prefix_cache_hits_total"]
        served.append(delta["vllm:request_success_total"])
        pods[pod] = {
            "requests": delta["vllm:request_success_total"],
            "prefix_hit_rate": delta["vllm:prefix_cache_hits_total"] / pod_queries if pod_queries else None,
        }
        if delta.get("memtrace_hosted_spend_usd"):  # hosted-model adapter pods report spend
            pods[pod]["spend_usd"] = delta["memtrace_hosted_spend_usd"]
    mean = sum(served) / len(served) if served else 0.0
    return {
        "pods": pods,
        "pool_prefix_hit_rate": hits / queries if queries else None,
        "load_imbalance": max(served) / mean if mean else None,
    }


def sum_counters(text: str, names: tuple[str, ...]) -> dict[str, float]:
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
