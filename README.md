# MEMTRACE

LLM serving and KV-cache research on one Apple Silicon Mac. Three parts:

1. **Serving platform:** an **llm-d control plane on kind** routing across **vLLM on the Apple Silicon GPU
   (`vllm-metal`)**, a simulated vLLM CPU tier, and optionally **Gemini Flash-Lite** as hosted overflow, tested with
   replayed GitHub Copilot agent traffic and injected failures.
2. **Agent-session KV memory:** how much KV cache to keep for agent sessions, for how long, in which memory tier, and
   how to route each call back to it, from a week of Copilot traces, a trace-driven simulator, and a check against a
   real engine.
3. **Memory-risk benchmark** (where the project started): follows a poisoned document through an agent's persistent
   memory to find where the attack breaks ([below](#memory-risk-benchmark)).

This is one machine, not a datacenter: Qwen3-0.6B for most results, simulated replicas where real ones do not fit.
Every number below says what was real and what was simulated. The project is complete.

## Results

| Question | Setup | Result |
| --- | --- | --- |
| Which engine per tier? | Qwen3-0.6B bf16; ShareGPT and Mooncake prompts, concurrency 1–8; real engines | `vllm-metal` 450 tok/s and 0.039 J/token on ShareGPT at concurrency 8 (2.2× `mlx_lm.server`, which crashed with a Metal OOM on long prompts and exposes no metrics); vLLM CPU 5.6–8.3× slower; BFCL tool-call accuracy 81.0–81.5% on every engine and batch size |
| Does cache-aware routing pay? | llm-d on kind, 4 simulated replicas calibrated to `vllm-metal`, synthetic agent sessions | Prefix-aware routing gives 1.8–2.3× the throughput of random; combined prefix + queue + KV is best at capacity and is the default |
| Does reactive autoscaling absorb bursts? | KEDA on in-flight requests, simulated replicas, 37 s burst | Reacts in 12 s, 4 replicas ready at 23 s: 63% SLO vs 100% with 4 fixed replicas; real cold starts (20–185 s) need warm headroom |
| Failure handling | Engine SIGKILL with 5 requests in flight (simulated pool) | 2/512 client failures (503); pool recovered in 5.2 s |
| Routing a GPU + CPU pool | Real `vllm-metal` + CPU simulator (11× slower) | Hardware-blind routing: 9–36 tok/s with 30 s timeouts; `capacity-load-scorer` 1.6–2.3× the default where the GPU is not cache-bound |
| Hosted overflow | + Gemini Flash-Lite via an adapter, overloaded local pool | 1.77 → 5.1–5.7 req/s, TTFT p95 13.5 → ~5 s, timeouts 14 → 0–1, $0.27–0.30 per 1k requests |
| Real agent traffic | 64 Copilot sessions (1,422 calls), open-loop at recorded timing; real GPU + CPU simulator | Capacity scorers finish 11–13% sooner but double TTFT p95 (9.2 vs 4.5 s): no policy wins both |
| Larger model | Qwen3-4B (MLX 4-bit) on `vllm-metal`; CPU simulator scaled by parameter count (an estimate) | Capacity scorers win both: −18% time to finish, TTFT p95 2.1 vs 3.45 s |
| Precise vs approximate prefix index | llm-d's precise index fed by KV events from both tiers, Copilot replay | No gain (728 vs 681 s to finish, same prefix-hit rates); approximate index kept |
| How long to keep agent KV? | All 7 days of the Copilot traces (301k sessions), the provider's own cached-token counts | The provider recomputed 8–12% of the prompt tokens that repeat the previous call. If every call reached its cache, a 5 min / 1 h / 24 h lifetime would serve 96 / 99.4 / 100% of the reusable prefix (provider: 89–92%), holding about 1× / 4× / 28× the KV of the 5 min cache |
| What buys the most reuse? | Trace-driven simulator, 16–64 replicas, Qwen3-4B KV sizes, GPU / RAM / SSD tiers; idealized routers | Routing to the cache first (least-loaded routing serves 3–21% of the reusable prefix, sticky routing 51–91%), then GPU budget, then host RAM (256 GB per replica: 97–99%). Knowing exactly where KV is adds 1–3 points; turn-aware retention about 1 |
| Does the simulator match a real engine? | One `vllm-metal` replica, KV fixed at 16k tokens, LRU, 120 Copilot sessions at real gaps | Within 1.6 points in every gap bin; hit or miss agrees on 884 of 886 calls. Covers one replica's GPU eviction only |

Per-run and per-day tables behind every row, with definitions: [results/](results/README.md).

Caveats: many runs paged on the host (swap counters are recorded per run), so differences under ~15% are not
claimed. The CPU tier is simulated in every mixed-pool result. In the KV study, the provider also caches prefixes
shared across sessions, which the simulator does not model, and routing, memory tiers, and lifetimes were tested in
simulation only.

## Architecture

```text
client ──> Istio gateway (kind, :30080) ──ext_proc──> llm-d EPP (v0.11.0 + MEMTRACE plugins)
                │                                        │ scrapes vLLM metrics from every pool endpoint
                ▼                                        ▼
        InferencePool ─┬─ vllm-gpu-relay pod ──TCP──> vllm-metal on the macOS host (Metal GPU)
                       ├─ vllm-cpu-sim pod   (llm-d-inference-sim calibrated to a real vLLM CPU pod)
                       └─ gemini-adapter pod (opt-in; vLLM-style metrics + hard spend cap)
```

- **EPP plugins** (Go, `epp-plugins/capacityload`): `capacity-load-scorer` (in-flight / measured capacity),
  `cache-aware-capacity-scorer` (expected completion time with prefix hits), `overflow-filter` (hosted endpoints only
  when the local pool is saturated).
- **Host GPU in the pool:** containers on macOS cannot use Metal, so the engine runs on the host and a relay pod stands
  in for it.
- **KV memory study** (`src/memtrace/kvmem/`): Copilot trace parser, retention analysis and cost model, a
  trace-driven simulator (`sim/`), and validation against a real engine.
- Images pinned by digest, router tag pinned, and a manifest per run (engine version, dataset hash, swap counters,
  replay scaling); Phase 6 outputs record the git commit that produced them.

## Quick start

Requirements: Apple Silicon Mac, Docker Desktop (VM: 8 GB, 10 CPUs), `kind`, `kubectl`, `helm`, `envsubst`
(`gettext`), Go (only to build the custom EPP image the first time), and `vllm-metal` installed with its official
installer into `~/.venv-vllm-metal`. Python environment: see [Install](#install).

```bash
make up                # kind + llm-d + GPU relay + CPU tier + custom EPP (about 3 minutes from nothing)
make up HOSTED=1       # also Gemini Flash-Lite as overflow (key in a local file, never committed; spend-capped)
make bench             # Copilot-replay routing comparison, 3 repetitions
make down              # stop the GPU engine, delete the cluster
scripts/stack.sh observability        # Prometheus, scrapes, alert rules (deploy/kind/observability/)
```

KV memory study (no cluster needed; the pilot needs the GPU):

```bash
python scripts/fetch_public_datasets.py copilot_agent          # 7 days, ~3.2 GB, SHA-256 pinned
memtrace report retention --by-day --out data/kvmem/m1-week.json
memtrace report kv-sim --day 2026-06-03 --replicas 64 --out data/kvmem/m2-2026-06-03.jsonl
scripts/studies/m3_pilot.sh                                     # real engine vs simulator
python scripts/results_page.py                                  # rebuild results/README.md
```

Other studies: `scripts/studies/` (`platform_studies.sh`, `copilot_study.sh`, `precise_study.sh`, `hetero_reps.sh`)
and `scripts/run_engine_baselines.sh`.

## Limits

- One machine, one GPU; no datacenter GPUs and no multi-node scale.
- CPU tier simulated (a real vLLM CPU pod next to the GPU engine pages the host); the 4B CPU tier is an estimate.
- Qwen3-0.6B for most results; hosted-model quality was not compared with the local model.
- `vllm-metal` cannot offload KV (its worker implements no vLLM KV connector), so RAM and SSD tiers are simulated with
  nominal bandwidths.
- The KV simulator is idealized: one cached prefix per session, no prefixes shared across sessions, no queueing; its
  routers are simplified forms of llm-d's scorers.
- Prefill/decode disaggregation, scale-from-zero, and a long-running production endpoint are out of scope.

## Memory-risk benchmark

Evaluates persistent-memory risk in tool-using agents as a causal chain: retrieval exposure, poisoned-memory
admission, delayed retrieval, unsafe proposals, policy-checker blocking, unsafe execution, and execution-format
failure. Report: [view online](https://htang7415.github.io/MEMTRACE/results/benchmark/memtrace_results.html) (source:
`results/benchmark/`; rebuild with `python results/benchmark/build.py`).

### Install

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install -e ".[retrieval]"   # optional: dense retrieval
python -m pip install -e ".[inference]"   # optional: MLX on Apple Silicon
python -m pip install -e ".[gemini]"      # optional: Gemini backend; requires GEMINI_API_KEY
```

Gemini calls share a process-wide spend cap (`MEMTRACE_GEMINI_BUDGET_USD`, default `1.0`); models without a price
entry are refused. Public datasets (Mooncake, Azure LLM inference, ShareGPT, BFCL, Copilot) are pinned to fixed
revisions and downloaded into the ignored `data/public/` with a SHA-256 manifest (`scripts/fetch_public_datasets.py`).
CI installs from lockfiles (`requirements/ci-lock.txt`, `requirements/canary-lock.txt`); regenerate them with
`pip-tools` after changing dependencies.

### CLI

```bash
memtrace assets build                                            # build and verify benchmark assets
memtrace --config configs/profile.toml run pilot --out-dir data/pilot/gate --force   # deterministic slice
memtrace evaluate gate --baseline tests/fixtures/regression/metrics.json --candidate data/pilot/gate/metrics.json
memtrace --config configs/mlx.toml run benchmark && memtrace evaluate score && memtrace evaluate validate
memtrace report explore --label unsafe                           # per-turn narrative of selected episodes
memtrace run engine-bench --help                                 # serving benchmark
memtrace report retention --help                                 # KV study (also kv-sim, kv-validate)
```

Against a local OpenAI-compatible server, set `MEMTRACE_OPENAI_BASE_URL` (and optionally `MEMTRACE_OPENAI_API_KEY`) and
use `configs/openai.toml`; calls are streamed and record TTFT, TPOT, latency, usage, and prefix-cache hits.
Configuration files use a `[memtrace]` TOML table; `MEMTRACE_*` environment variables override it.

## Development

```bash
python -m ruff check . && python -m ruff format --check .
python -m mypy
python -m pytest
```

CI (Python 3.11 and 3.13): lint, types, tests, a wheel smoke test, the regression gate (a deterministic 54-episode
slice diffed against `tests/fixtures/regression/`), shellcheck, and the Go plugin tests against the pinned
llm-d-router tree.

### Structure

- `src/memtrace/core/`, `backends/`, `evaluation/`, `commands/`: the memory-risk benchmark.
- `src/memtrace/serving/`: serving benchmark harness (workloads, open-loop replay, metrics, quality gate) and the
  hosted-model adapter.
- `src/memtrace/kvmem/`: agent-session KV memory (trace parser, retention analysis, cost model, simulator, validation).
- `deploy/kind/`, `epp-plugins/`, `scripts/`, `Makefile`: the serving platform.
- `results/`: published aggregate tables (`scripts/results_page.py`) and the benchmark report (`benchmark/`).
- `configs/`, `tests/`: configurations and deterministic tests (no model inference needed).
- `data/`, `artifacts/`, `figures/`: ignored local outputs.
