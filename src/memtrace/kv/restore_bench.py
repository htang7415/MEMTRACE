"""Restore vs recompute: when is loading a prefix's KV back from the SSD tier faster than computing it again, and is
the output the same?

For each model and prefix length, each repetition takes a fresh random prompt P of that many tokens. A stock engine
(no tier) measures it cold and as a GPU hit: the recompute baseline. An engine with a host pool and an SSD tier then
measures the same prompts three ways:

1. cold: P has never been seen, so the engine computes the whole prefix (recompute);
2. GPU hit: P again right away, served from the GPU prefix cache;
3. restore: P again after filler prompts have pushed it out of the GPU cache and the host pool, so its KV comes
   back from the SSD tier (the engine's KV-connector counter confirms how many tokens were loaded). Before it, the
   run waits for the tier's write-through to go idle and records the wait (`offload_drain_s`): this isolates the
   cost of a restore; the resume scenario (R1) measures restores under the contention instead.

Each decodes the same greedy continuation; the run records whether the restored one matches the GPU hit's (the
same cached-prefix path, so it should) and the cold one's.

    memtrace run restore-bench experiments/s1_restore_bench.yaml [--out data/runs]
"""

from __future__ import annotations

from argparse import ArgumentParser
import http.client
import json
from pathlib import Path
import random
import shutil
import time
from typing import Any
import urllib.request
from urllib.parse import urlsplit

import yaml

from memtrace.harness.bundle import RunBundle
from memtrace.harness.targets import make_target, scrape_vllm_counters
from memtrace.kv.resume import arm_params, dir_gib, kv_store_root

SPEC_SCHEMA = "memtrace-restore-v1"
VOCAB = (1_000, 150_000)  # token ids inside a Qwen-sized vocabulary, clear of special tokens


def complete(base_url: str, model: str, prompt: list[int], max_tokens: int, timeout_s: float) -> tuple[float, str]:
    """Greedy streaming /v1/completions on token ids; (seconds to the first token, the full text)."""
    parts = urlsplit(base_url)
    conn = http.client.HTTPConnection(parts.hostname or "localhost", parts.port, timeout=timeout_s)
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
    }
    started = time.perf_counter()
    try:
        conn.request("POST", "/v1/completions", body=json.dumps(body), headers={"content-type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"http {resp.status}: {resp.read(300).decode('utf-8', 'replace')}")
        ttft: float | None = None
        text = []
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            # The first streamed choice carries the first generated token, even when its text is empty (random
            # token ids can decode to a partial UTF-8 sequence).
            for choice in json.loads(line[5:]).get("choices") or []:
                ttft = time.perf_counter() - started if ttft is None else ttft
                text.append(choice.get("text") or "")
        if ttft is None:
            raise RuntimeError("no tokens in the response")
        return ttft, "".join(text)
    finally:
        conn.close()


OFFLOAD_JOBS = ("vllm:kv_offload_tiering_active_cascade_jobs", "vllm:kv_offload_tiering_active_promotion_jobs")


def offload_jobs(base_url: str) -> float:
    """Store (cascade) and load (promotion) jobs the offload tier has in flight; 0 without a tier."""
    with urllib.request.urlopen(base_url + "/metrics", timeout=5) as resp:
        text = resp.read().decode("utf-8", "replace")
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if not line.startswith("#") and line.split("{", 1)[0].split(" ", 1)[0] in OFFLOAD_JOBS
    )


def drain_offload(base_url: str, timeout_s: float, poll_s: float = 0.1) -> float:
    """Wait until the offload tier is idle (the write-through of everything computed so far has reached the disk);
    returns the seconds waited. A request that arrives while stores are in flight does not find its blocks."""
    started = time.perf_counter()
    while offload_jobs(base_url) > 0:
        if time.perf_counter() - started > timeout_s:
            raise TimeoutError(f"offload tier still busy after {timeout_s} s")
        time.sleep(poll_s)
    return time.perf_counter() - started


def random_prompt(rng: random.Random, tokens: int) -> list[int]:
    return [rng.randrange(*VOCAB) for _ in range(tokens)]


def measure(
    base_url: str,
    model: str,
    rng: random.Random,
    *,
    prefix_tokens: int,
    flush_tokens: int,
    filler_tokens: int,
    max_tokens: int,
    timeout_s: float,
    restore: bool = True,
) -> dict[str, float]:
    """One repetition at one prefix length: cold, GPU hit, then (with `restore`) a restore after `flush_tokens` of
    filler."""
    prompt = random_prompt(rng, prefix_tokens)
    cold, cold_text = complete(base_url, model, prompt, max_tokens, timeout_s)
    gpu, gpu_text = complete(base_url, model, prompt, max_tokens, timeout_s)
    if not restore:
        return {"ttft_cold_ms": cold * 1000, "ttft_gpu_hit_ms": gpu * 1000}
    for _ in range(-(-flush_tokens // filler_tokens)):  # distinct fillers evict P from the GPU and the host pool
        complete(base_url, model, random_prompt(rng, filler_tokens), 1, timeout_s)
    drained = drain_offload(base_url, timeout_s)
    before = scrape_vllm_counters(base_url).get("external_prefix_cache_hits_total", 0.0)
    restored, restore_text = complete(base_url, model, prompt, max_tokens, timeout_s)
    loaded = scrape_vllm_counters(base_url).get("external_prefix_cache_hits_total", 0.0) - before
    return {
        "ttft_cold_ms": cold * 1000,
        "ttft_gpu_hit_ms": gpu * 1000,
        "ttft_restore_ms": restored * 1000,
        "restore_speedup": cold / restored,
        "restored_share": loaded / prefix_tokens,
        # A restore loads the same KV a GPU hit reads, so their continuations should match exactly. A cold request
        # computes its last prompt token inside a prefill chunk, a cached one on a different kernel path, so on
        # random-token prompts a near-tie can flip a greedy token: identity with cold is reported, not required.
        "output_same_as_gpu_hit_rate": float(restore_text == gpu_text),
        "output_same_as_cold_rate": float(restore_text == cold_text),
        "offload_drain_s": drained,
    }


def engine_lifetimes(spec: dict[str, Any]) -> list[tuple[str, dict[str, Any], str, list[tuple[int, int]]]]:
    """(model name, model, engine kind, [(prefix tokens, repeat)]) per engine start. A stock engine serves all of a
    model's repetitions (the recompute baseline). A tier engine serves one repetition with an empty store: every
    computed token is written through to the SSD, so a shared store fills to its cap, and the disk tier's LRU then
    deletes the oldest blocks, which are the prompt under test."""
    out = []
    for name, model in spec["models"].items():
        lengths = [n for n in spec["prefix_tokens"] if n <= int(model.get("max_prefix_tokens", 10**9))]
        items = [(n, rep) for n in lengths for rep in range(int(spec.get("repeats", 3)))]
        out.append((name, model, "stock", items))
        out.extend((name, model, "tier", [item]) for item in items)
    return out


def load_spec(path: Path) -> dict[str, Any]:
    spec: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise ValueError(f"{path}: schema_version must be {SPEC_SCHEMA}")
    for key in ("name", "models", "prefix_tokens", "kv_tier", "target"):
        if key not in spec:
            raise ValueError(f"{path}: missing {key}")
    return spec


def run(spec_path: Path, out_root: Path) -> Path:
    spec = load_spec(spec_path)
    seed0 = int(spec.get("seed", 0))
    bundle = RunBundle(spec, out_root)
    for name, model, engine_kind, items in engine_lifetimes(spec):
        tiered = engine_kind == "tier"
        target = {**spec["target"], **model["target"]}
        store = kv_store_root(spec) / bundle.run_id / name
        shutil.rmtree(store, ignore_errors=True)  # an empty store for every tier engine
        params = arm_params(target, {"kv_tier": spec["kv_tier"]} if tiered else None, store)
        gpu_tokens = int(model["gpu_blocks"]) * 16
        pool_tokens = int(spec["kv_tier"]["host_pool_gib"] * 1024**3 / model["kv_bytes_per_token"])
        flush = int((gpu_tokens + pool_tokens) * float(spec.get("flush_factor", 1.5)))
        engine_label = f"{name}-{engine_kind}" + ("-" + "-".join(f"{n}r{rep}" for n, rep in items) if tiered else "")
        with make_target("vllm_metal", params, bundle.log_dir(engine_label)) as engine:
            base_url, described = engine.base_urls[0], engine.describe()
            complete(base_url, str(target["model"]), random_prompt(random.Random(-1), 64), 1, 600)  # warm up
            for n, rep in items:
                cell_id = f"model={name}/engine={engine_kind}/prefix_tokens={n}"
                seed = seed0 * 1_000_000 + n * 100 + rep
                params_cell = {"model": name, "engine": engine_kind, "prefix_tokens": n}
                with bundle.trial(cell_id, params_cell, repeat=rep, seed=seed) as trial:
                    trial.target = described
                    trial.metrics.update(
                        measure(
                            base_url,
                            str(target["model"]),
                            random.Random(seed),
                            prefix_tokens=n,
                            flush_tokens=flush,
                            filler_tokens=int(spec.get("filler_tokens", 2048)),
                            max_tokens=int(spec.get("max_tokens", 32)),
                            timeout_s=float(spec.get("timeout_s", 900)),
                            restore=tiered,
                        )
                    )
                    if tiered:
                        trial.metrics["kv_store_gib"] = dir_gib(store)
        shutil.rmtree(store, ignore_errors=True)
    return bundle.finish()


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(prog="memtrace run restore-bench", description=__doc__.split("\n\n")[0])
    parser.add_argument("spec", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/runs"))
    args = parser.parse_args(argv)
    run(args.spec, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
