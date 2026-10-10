"""The resume scenario: when a paused agent session returns, how much of its prefix does the serving stack still hold?

Agent sessions from a GitHub Copilot coding-agent day are replayed onto a real engine as full chat histories
(deterministic filler text at `scale` x each message's tokens, as in K9), closed-loop per session: a call is sent
its recorded pause after the session's previous call returns, as the agent would send it. Pauses are capped at
`idle_cap_s` and divided by `time_scale`. `concurrency` sessions run at once; a finished session is replaced by
the next one of a seeded order, and the first ones join at a random point of their life so the engine holds
sessions of mixed ages from the start.

Before each call the engine's own `/tokenize` renders the prompt exactly as it will be served, so every call
records the token-level prefix it shares with the session's previous call, rounded down to whole KV blocks (its
*reusable* prefix), next to what the engine served from cache (`cached_tokens`). A call is *resumed* when its
recorded pause is at least `resume_gap_s`. The headline per trial is the share of resumed calls' reusable prefix
the engine recomputed, and their TTFT.

Arms change the serving stack (`target` overrides); every arm replays the same sessions in the same order with the
same pauses within a repeat. Timing is closed-loop, so it follows each arm's own speed.

    memtrace run resume experiments/r0_resume_baseline.yaml [--out data/runs]
"""

from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import http.client
import json
import math
from pathlib import Path
import random
import shutil
import sys
import threading
import time
from typing import Any, Callable, Iterator, Sequence
from urllib.parse import urlsplit

import numpy as np
import yaml

from memtrace.datasets.loaders.copilot import iter_archive
from memtrace.datasets.sources import verified_path
from memtrace.harness.provenance import make_provenance
from memtrace.harness.results import RESULT_SCHEMA_VERSION, ExperimentResult, TrialResult, aggregate_cells
from memtrace.harness.runner import wait_for_quiet_host
from memtrace.harness.spec import QuietHost
from memtrace.harness.stamps import utc_now_iso
from memtrace.harness.system_info import host_swap_pages, keep_awake, sleep_clock
from memtrace.harness.targets import make_target, scrape_vllm_counters
from memtrace.kv.gateway_replay import ChatCall, Renderer, stream_chat
from memtrace.kv.policy_traces import _end_s, calls_of, history, sample_ids

SPEC_SCHEMA = "memtrace-resume-v1"
# Gap since the session's previous call ended, as in the retention analysis and K11.
GAP_BINS = (
    ("lt10s", 0.0, 10.0),
    ("10_60s", 10.0, 60.0),
    ("1_5min", 60.0, 300.0),
    ("5_10min", 300.0, 600.0),
    ("10_60min", 600.0, 3600.0),
    ("gt1h", 3600.0, math.inf),
)


@dataclass(frozen=True)
class ResumeSession:
    id: str
    calls: tuple[ChatCall, ...]  # ChatCall.t is the call's start in trace seconds from the session's first call
    gaps: tuple[float | None, ...]  # trace seconds from the previous call's end to this call's start (None: first)


def parse_session(session: dict[str, Any]) -> ResumeSession:
    """A Copilot session's calls in order, with each call's recorded pause."""
    first_len: dict[str, int] = {}
    calls: list[ChatCall] = []
    gaps: list[float | None] = []
    prev_end: float | None = None
    t0: float | None = None
    for c in calls_of(session):
        dur = float(c["duration_ms"] or 0) / 1000
        start = _end_s(c["timestamp"]) - dur
        t0 = start if t0 is None else t0
        # per-message counts drift between calls for unchanged messages: keep the first one (as K9)
        msgs = tuple((m["role"], m["content"], first_len.setdefault(m["content"], m["tokens"])) for m in history(c))
        calls.append(ChatCall(t=start - t0, dur=dur, messages=msgs, out_tokens=int(c["tokens"].get("completion") or 0)))
        gaps.append(None if prev_end is None else max(0.0, start - prev_end))
        prev_end = start + dur if prev_end is None else max(prev_end, start + dur)
    return ResumeSession(id=session["session_id"], calls=tuple(calls), gaps=tuple(gaps))


def load_sessions(archive: Path, sessions: int, seed: int, cache: Path) -> list[ResumeSession]:
    """Seeded sessions of one Copilot day (rows cached locally; the derivation is deterministic)."""
    if not cache.exists():
        ids = sample_ids(archive, sessions, seed)
        rows = [s for s in iter_archive(archive) if s["session_id"] in ids]
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".part")
        tmp.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows), encoding="utf-8")
        tmp.replace(cache)
    rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line]
    parsed = (parse_session(r) for r in rows)
    return sorted((s for s in parsed if len(s.calls) >= 2), key=lambda s: s.id)


def common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    """Length of the longest common prefix of two token lists."""
    n = min(len(a), len(b))
    if n == 0:
        return 0
    diff = np.flatnonzero(np.asarray(a[:n]) != np.asarray(b[:n]))
    return int(diff[0]) if len(diff) else n


def tokenize(base_url: str, model: str, messages: list[dict[str, Any]], timeout_s: float) -> list[int]:
    """The prompt's token ids as the engine renders it (vLLM's /tokenize, same chat template as serving)."""
    parts = urlsplit(base_url)
    conn = http.client.HTTPConnection(parts.hostname or "localhost", parts.port, timeout=timeout_s)
    body = {
        "model": model,
        "messages": messages,
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        conn.request("POST", "/tokenize", body=json.dumps(body), headers={"content-type": "application/json"})
        resp = conn.getresponse()
        data = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"tokenize http {resp.status}: {data[:300].decode('utf-8', 'replace')}")
        return [int(t) for t in json.loads(data)["tokens"]]
    finally:
        conn.close()


@dataclass
class CallRecord:
    session: str
    call: int
    gap_s: float | None  # recorded pause before this call (trace seconds); None for a session's first replayed call
    sent_s: float  # seconds from the replay's start
    status: str
    ttft_s: float | None = None
    e2e_s: float | None = None
    prompt_tokens: int = 0
    cached_tokens: int = 0
    reusable_tokens: int = 0  # prefix shared with the previous replayed call, in whole KV blocks
    error: str | None = None


def replay(
    sessions: Sequence[ResumeSession],
    *,
    send: Callable[[ResumeSession, int, list[int] | None], tuple[CallRecord, list[int] | None]],
    concurrency: int,
    horizon_s: float,
    stagger_s: float,
    idle_cap_s: float,
    time_scale: float,
    seed: int,
) -> list[CallRecord]:
    """Run `concurrency` session slots closed-loop until `horizon_s`; returns every call's record.

    `send(session, call_index, previous_tokens)` performs one call and returns its record and the tokens it sent
    (None when it failed). A slot replays sessions back to back from one shared seeded order; its first session
    joins at a uniform random call (mixed ages), later ones from their first call."""
    rng = random.Random(seed)
    order = list(range(len(sessions)))
    rng.shuffle(order)
    starts = [rng.uniform(0.0, stagger_s) for _ in range(concurrency)]
    if concurrency > len(order):
        raise ValueError(f"concurrency {concurrency} exceeds the {len(order)} sessions")
    joins = [rng.randrange(len(sessions[order[i]].calls) - 1) for i in range(concurrency)]
    # Each session is replayed at most once: a second pass would find its own text already cached.
    queue: Iterator[int] = iter(order[concurrency:])
    lock = threading.Lock()
    records: list[CallRecord] = []
    t0 = time.perf_counter()

    def now() -> float:
        return time.perf_counter() - t0

    def slot(i: int) -> None:
        time.sleep(starts[i])
        s_idx, first = order[i], joins[i]
        while now() < horizon_s:
            s = sessions[s_idx]
            prev: list[int] | None = None
            done_at = now()
            for k in range(first, len(s.calls)):
                if k > first:
                    pause = min(float(s.gaps[k] or 0.0), idle_cap_s) / time_scale
                    if done_at + pause >= horizon_s:
                        return
                    time.sleep(max(0.0, done_at + pause - now()))
                record, prev = send(s, k, prev if k > first else None)
                record.gap_s = s.gaps[k] if k > first else None
                with lock:
                    records.append(record)
                done_at = now()
                if done_at >= horizon_s:
                    return
            with lock:
                nxt = next(queue, None)
            if nxt is None:
                return
            s_idx, first = nxt, 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for f in [pool.submit(slot, i) for i in range(concurrency)]:
            f.result()
    return sorted(records, key=lambda r: r.sent_s)


def make_sender(
    base_url: str,
    model: str,
    render: Renderer,
    *,
    block_tokens: int,
    output_scale: float,
    max_output_tokens: int,
    timeout_s: float,
    t0: float,
) -> Callable[[ResumeSession, int, list[int] | None], tuple[CallRecord, list[int] | None]]:
    def send(s: ResumeSession, k: int, prev: list[int] | None) -> tuple[CallRecord, list[int] | None]:
        call = s.calls[k]
        messages = render.messages(s.id, call)
        sent = time.perf_counter() - t0
        try:
            tokens = tokenize(base_url, model, messages, timeout_s)
            reusable = 0 if prev is None else common_prefix(prev, tokens) // block_tokens * block_tokens
            body = {
                "model": model,
                "messages": messages,
                "max_tokens": max(1, min(max_output_tokens, round(call.out_tokens * output_scale))),
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False},
            }
            started = time.perf_counter()
            ttft, usage, _ = stream_chat(base_url, body, timeout_s)
            record = CallRecord(
                session=s.id,
                call=k,
                gap_s=None,
                sent_s=round(sent, 3),
                status="ok",
                ttft_s=ttft,
                e2e_s=time.perf_counter() - started,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                cached_tokens=int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)),
                reusable_tokens=reusable,
            )
            return record, tokens
        except Exception as exc:  # noqa: BLE001 - every failure is recorded, none stops the replay
            error = f"{type(exc).__name__}: {exc}"[:200]
            return CallRecord(
                session=s.id, call=k, gap_s=None, sent_s=round(sent, 3), status="error", error=error
            ), None

    return send


def _pct(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def recompute_share(records: Sequence[CallRecord]) -> float | None:
    """Share of the reusable prefix the engine computed again instead of serving it from cache."""
    reusable = sum(r.reusable_tokens for r in records)
    if not reusable:
        return None
    served = sum(min(r.cached_tokens, r.reusable_tokens) for r in records)
    return 1.0 - served / reusable


def resume_metrics(
    records: Sequence[CallRecord], *, warmup_s: float, resume_gap_s: float, failed_ttft_s: float
) -> dict[str, float]:
    """Trial metrics over the calls sent after `warmup_s`."""
    measured = [r for r in records if r.sent_s >= warmup_s]
    ok = [r for r in measured if r.status == "ok"]
    out: dict[str, float] = {
        "calls": float(len(measured)),
        "errors": float(len(measured) - len(ok)),
        "error_rate": (len(measured) - len(ok)) / len(measured) if measured else 0.0,
    }
    reusing = [r for r in ok if r.reusable_tokens > 0]
    resumed = [r for r in reusing if r.gap_s is not None and r.gap_s >= resume_gap_s]
    resumed_all = [r for r in measured if r.gap_s is not None and r.gap_s >= resume_gap_s]
    out["resumed_calls"] = float(len(resumed))
    for name, group in (("all", reusing), ("resumed", resumed)):
        share = recompute_share(group)
        if share is not None:
            out[f"{name}_recompute_share"] = share
            out[f"{name}_recomputed_tokens_per_call"] = sum(
                r.reusable_tokens - min(r.cached_tokens, r.reusable_tokens) for r in group
            ) / len(group)
    for name, group in (("all", ok), ("resumed", resumed)):
        ttfts = [r.ttft_s * 1000 for r in group if r.ttft_s is not None]
        if ttfts:
            out[f"{name}_ttft_p50_ms"] = _pct(ttfts, 50)
            out[f"{name}_ttft_p95_ms"] = _pct(ttfts, 95)
    if resumed_all:  # a failed resumed call counts as waiting the full timeout
        waited = [r.ttft_s if r.status == "ok" and r.ttft_s is not None else failed_ttft_s for r in resumed_all]
        out["resumed_ttft_all_p95_ms"] = _pct([w * 1000 for w in waited], 95)
    for name, lo, hi in GAP_BINS:
        in_bin = [r for r in reusing if r.gap_s is not None and lo <= r.gap_s < hi]
        share = recompute_share(in_bin)
        if share is not None:
            out[f"gap_{name}_served_share"] = 1.0 - share
            out[f"gap_{name}_calls"] = float(len(in_bin))
    return out


def arm_params(target: dict[str, Any], arm: dict[str, Any] | None, store_dir: Path) -> dict[str, Any]:
    """Engine params for an arm: the shared target, the arm's `target` overrides, and its `kv_tier` as vllm-metal's
    own offload flags: a host pool of `host_pool_gib`, plus an `fs` disk tier under `store_dir` when the tier
    sets `max_size_gib`."""
    arm = arm or {}
    params = {**target, **arm.get("target", {})}
    tier = arm.get("kv_tier")
    if tier:
        args = [*params.get("extra_args", []), "--kv-offloading-size", str(tier["host_pool_gib"])]
        if "max_size_gib" in tier:
            disk = {"type": "fs", "root_dir": str(store_dir), "max_size_gib": tier["max_size_gib"]}
            config = {"kv_connector_extra_config": {"secondary_tiers": [disk]}}
            args += ["--kv-transfer-config", json.dumps(config, separators=(",", ":"))]
        params["extra_args"] = args
    return params


def kv_store_root(spec: dict[str, Any]) -> Path:
    """Where disk tiers keep their blocks. It must be on the internal SSD: on a non-APFS external volume vllm-metal
    falls back to buffered I/O, and a restore becomes slower than recomputing the prefix."""
    return Path(str(spec.get("kv_store_dir", "~/.cache/memtrace/kvstore"))).expanduser()


def dir_gib(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**3 if path.exists() else 0.0


def load_spec(path: Path) -> dict[str, Any]:
    spec: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise ValueError(f"{path}: schema_version must be {SPEC_SCHEMA}")
    for key in ("name", "day", "arms", "replay", "target"):
        if key not in spec:
            raise ValueError(f"{path}: missing {key}")
    return spec


def run(spec_path: Path, out_root: Path, log: Callable[[str], None] = lambda m: print(m, file=sys.stderr)) -> Path:
    spec = load_spec(spec_path)
    rp = spec["replay"]
    started_at = utc_now_iso()
    n, seed0 = int(spec.get("sessions", 200)), int(spec.get("seed", 0))
    rel = spec["day"]
    cache = (
        Path(spec.get("cache_dir", "artifacts/cache/resume")) / f"{Path(rel).name.split('.tar')[0]}.n{n}.s{seed0}.jsonl"
    )
    sessions = load_sessions(verified_path(rel), n, seed0, cache)
    render = Renderer(float(rp["scale"]))
    run_id = f"{datetime.now(tz=timezone.utc):%Y%m%dT%H%M%SZ}-{spec['name']}"
    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True)
    (out_dir / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    cells = {f"arm={arm}": {"arm": arm} for arm in spec["arms"]}
    quiet_host = QuietHost(**spec.get("quiet_host", {}))
    keep_awake()
    trials: list[TrialResult] = []
    for rep in range(int(spec.get("repeats", 1))):
        seed = seed0 * 1000 + rep
        for cell_id, cell in cells.items():
            label = f"{cell_id}/r{rep}"
            # A fresh, empty disk tier per trial: a store left by another arm or repeat would serve its blocks.
            store = kv_store_root(spec) / run_id / label.replace("/", "_")
            shutil.rmtree(store, ignore_errors=True)
            params = arm_params(spec["target"], spec["arms"][cell["arm"]], store)
            load, quiet, _ = wait_for_quiet_host(quiet_host)
            t_start, trial_started = time.perf_counter(), utc_now_iso()
            swap_before, slept = host_swap_pages(), sleep_clock()
            with make_target("vllm_metal", params, out_dir / "logs" / label.replace("/", "_")) as target:
                base_url, model = target.base_urls[0], str(params["model"])
                t0 = time.perf_counter()
                records = replay(
                    sessions,
                    send=make_sender(
                        base_url,
                        model,
                        render,
                        block_tokens=int(rp.get("block_tokens", 16)),
                        output_scale=float(rp["output_scale"]),
                        max_output_tokens=int(rp["max_output_tokens"]),
                        timeout_s=float(rp["timeout_s"]),
                        t0=t0,
                    ),
                    concurrency=int(rp["concurrency"]),
                    horizon_s=float(rp["horizon_s"]),
                    stagger_s=float(rp.get("stagger_s", 30.0)),
                    idle_cap_s=float(spec.get("idle_cap_s", 3600.0)),
                    time_scale=float(rp.get("time_scale", 1.0)),
                    seed=seed,
                )
                engine = scrape_vllm_counters(base_url)
                described = target.describe()
            swap_after = host_swap_pages()
            metrics = resume_metrics(
                records,
                warmup_s=float(rp["warmup_s"]),
                resume_gap_s=float(spec.get("resume_gap_s", 60.0)),
                failed_ttft_s=float(rp["timeout_s"]),
            )
            queries = engine.get("prefix_cache_queries_total", 0.0)
            metrics["engine_prefix_hit_ratio"] = (
                engine.get("prefix_cache_hits_total", 0.0) / queries if queries else 0.0
            )
            # Tokens served from the GPU prefix cache vs loaded back from the KV tier, over the whole trial.
            local, tier = (
                engine.get("prefix_cache_hits_total", 0.0),
                engine.get("external_prefix_cache_hits_total", 0.0),
            )
            metrics["engine_tier_hit_tokens"] = tier
            metrics["engine_tier_share_of_served"] = tier / (local + tier) if local + tier else 0.0
            metrics["kv_store_gib"] = dir_gib(store)
            shutil.rmtree(store, ignore_errors=True)
            if swap_before and swap_after:
                metrics["host_swapouts"] = float(swap_after["swapouts"] - swap_before["swapouts"])
            metrics["host_slept_s"] = slept()
            if metrics["host_slept_s"] > 5:  # latencies after a sleep come from a throttled dark wake
                quiet = False
                log(f"{label}: WARNING the host slept {metrics['host_slept_s']:.0f} s during the trial")
            log(f"{label}: " + " ".join(f"{k}={v:.4g}" for k, v in metrics.items()))
            trials.append(
                TrialResult(
                    trial_id=label,
                    cell_id=cell_id,
                    repeat=rep,
                    seed=seed,
                    status="ok",
                    started_at=trial_started,
                    duration_s=round(time.perf_counter() - t_start, 3),
                    host_load_1m_before=load,
                    quiet_host_ok=quiet,
                    metrics={k: round(v, 6) for k, v in metrics.items()},
                    requests_per_endpoint=[len(records)],
                    target=described,
                    error=None,
                )
            )
            with (out_dir / "requests.jsonl").open("a", encoding="utf-8") as fh:
                for r in records:
                    fh.write(json.dumps({"trial": label, **asdict(r)}) + "\n")
    result = ExperimentResult(
        schema_version=RESULT_SCHEMA_VERSION,
        run_id=run_id,
        name=spec["name"],
        description=spec.get("description", ""),
        spec=spec,
        provenance=make_provenance(spec, started_at, {"day": rel, "trials_completed": len(trials)}),
        trials=trials,
        cells=aggregate_cells(trials, cells),
    )
    (out_dir / "results.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    log(f"wrote {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(prog="memtrace run resume", description=__doc__.split("\n\n")[0])
    parser.add_argument("spec", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/runs"))
    args = parser.parse_args(argv)
    run(args.spec, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
