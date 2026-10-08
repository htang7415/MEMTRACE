# MEMTRACE

MEMTRACE is an LLM serving platform built on one Apple Silicon Mac: an **llm-d control plane on kind** routing
across **vLLM on the Apple Silicon GPU (`vllm-metal`)**, a **vLLM CPU tier** (a simulator calibrated to a real vLLM
CPU pod), and optionally **Gemini Flash-Lite** as hosted overflow. It is tested with replayed real agent traffic
(GitHub Copilot coding-agent traces), injected failures, and a task-level quality gate. It began as a benchmark of
persistent-memory risk in tool-using agents; that benchmark and its regression gate are still here
([below](#memory-risk-benchmark)).

This is one machine, not a datacenter: a 16 GB Mac mini, Qwen3-0.6B (Qwen3-4B 4-bit in one study), and simulated
replicas where real ones do not fit. Every number below names what was real and what was simulated.

## Architecture

```text
client ──> Istio gateway (kind, :30080) ──ext_proc──> llm-d EPP (endpoint picker, v0.11.0 + MEMTRACE plugins)
                │                                        │ scrapes vLLM metrics from every pool endpoint
                ▼                                        ▼
        InferencePool ─┬─ vllm-gpu-relay pod ──TCP──> vllm-metal on the macOS host (Metal GPU)
                       ├─ vllm-cpu-sim pod   (llm-d-inference-sim calibrated to a real vLLM CPU pod)
                       └─ gemini-adapter pod (opt-in; vLLM-style metrics + hard spend cap)
```

- **Custom EPP plugins** (Go, `epp-plugins/capacityload`, built into llm-d's EPP by `scripts/build_epp.sh`):
  `capacity-load-scorer` (load = in-flight / measured capacity), `cache-aware-capacity-scorer` (expected completion
  time with prefix hits), `overflow-filter` (hosted endpoints only when the local pool is saturated).
- **Host GPU in the pool:** containers on macOS cannot use Metal, so the engine runs on the host and a relay pod
  stands in for it; the EPP routes to it with the engine's real metrics.
- Pinned container images (by digest), pinned upstream router tag, and per-run manifests (engine version, dataset
  hash, swap counters, replay scaling factors).

## Quick start

Requirements: Apple Silicon Mac, Docker Desktop (VM: 8 GB, 10 CPUs), `kind`, `kubectl`, `helm`, `envsubst`
(`gettext`), Go (only to build the custom EPP image the first time), and `vllm-metal` installed with its official
installer into `~/.venv-vllm-metal`. The project's Python environment is described under [Install](#install).

```bash
make up                # kind + llm-d + GPU relay + CPU tier + custom EPP (about 3 minutes from nothing)
make up HOSTED=1       # also Gemini Flash-Lite as overflow (key in a local file, never committed; spend-capped)
make bench             # Copilot-replay routing comparison: combined vs capacity vs cache-cost, 3 repetitions
make down              # stop the GPU engine, delete the cluster
scripts/stack.sh observability           # Prometheus, scrapes, alert rules (deploy/kind/observability/)
```

`deploy/kind/observability/dashboard.json` is a Grafana dashboard (TTFT / TPOT / goodput / errors per tier, KV and
prefix-cache use, hosted spend); `alerts.yaml` alerts on an absent load metric, gateway timeouts, TTFT SLO, replica
down, and hosted spend. A stream cut by the gateway's 30 s timeout still returns HTTP 200; the timeout alert keys on
Envoy's `UT` response flag instead.

Other studies: `scripts/studies/platform_studies.sh` (`study`, `failover`, `burst` after `scripts/stack.sh
autoscaling`, `hetero_study`, `hosted_study`), `scripts/studies/precise_study.sh`, `scripts/run_engine_baselines.sh`.

## Results

All on one 16 GB Apple Silicon Mac. "Sim" = `llm-d-inference-sim` calibrated to a measured engine.

| Question | Setup | Result |
| --- | --- | --- |
| Which engine per tier? | Qwen3-0.6B bf16; ShareGPT and Mooncake tool-agent prompts, concurrency 1–8; real engines | `vllm-metal` 450 tok/s and 0.039 J/token on ShareGPT at concurrency 8 (2.2× `mlx_lm.server`, which crashed with a Metal OOM on long prompts and exposes no metrics); vLLM CPU 5.6–8.3× slower; BFCL tool-call accuracy 81.0–81.5% on every engine and batch size |
| Does cache-aware routing pay? | llm-d on kind, 4 sim replicas calibrated to `vllm-metal`, synthetic agent sessions | Prefix-aware routing gives 1.8–2.3× the throughput of random; combined prefix + queue + KV is best at capacity and is the default |
| Does reactive autoscaling absorb bursts? | KEDA on in-flight requests, sim replicas, 37 s burst | Reacts in 12 s, 4 replicas Ready at 23 s: 63% SLO vs 100% with 4 fixed replicas; real cold starts (20–185 s) need warm headroom |
| Failure handling | Engine SIGKILL with 5 requests in flight (sim pool) | 2/512 client failures (503), pool recovered in 5.2 s |
| Routing a GPU + CPU pool | Real `vllm-metal` + CPU sim (11× slower) | Hardware-blind routing: 9–36 tok/s with 30 s timeouts; `capacity-load-scorer` 1.6–2.3× the default where the GPU is not cache-bound |
| Hosted overflow | + Gemini Flash-Lite via adapter, overloaded local pool | 1.77 → 5.1–5.7 req/s, TTFT p95 13.5 → ~5 s, timeouts 14 → 0–1, $0.27–0.30 per 1k requests |
| Real agent traffic | 64 GitHub Copilot sessions (1,422 calls, 84% of prompt tokens cached), open-loop at recorded timing; real GPU + CPU sim | Capacity scorers finish 11–13% sooner but double TTFT p95 (9.2 vs 4.5 s): no policy wins both |
| Larger model | Qwen3-4B (MLX 4-bit) on `vllm-metal`; CPU sim scaled by parameter count (estimate) | Capacity scorers win both: −18% time to finish, TTFT p95 2.1 vs 3.45 s; the default's CPU calls time out |
| Precise vs approximate prefix index | llm-d precise index fed by KV events from both tiers, Copilot replay | No gain (681 vs 728 s, same prefix-hit rates); approximate index kept |
| How long to keep agent KV? | All 7 days of the Copilot traces (301k sessions), the provider's own cached-token counts | The provider recomputes 8–12% of reusable prompt tokens; 26–29% of that follows gaps of 5 min or more (expiry), and up to 26–47% is placement or a prompt edit. With perfect placement a 5 min / 1 h / 24 h cache keeps 95 / 98.4 / 99.1% of the reusable prefix, for about 4× / 28× the 5 min working set |
| What buys the most reuse? | Trace-driven simulator (`memtrace report kv-sim`), 16–64 replicas, Qwen3-4B KV sizes, GPU / RAM / SSD tiers | Placement first (least-loaded routing serves 3–21% of the reusable prefix, llm-d's approximate index 51–91%), then GPU budget, then a host-RAM tier (256 GB per replica: 97–99%). The precise index adds 1–3 points; turn-aware retention about 1; a 5 min lifetime costs 2–4 points but needs up to 8× less RAM memory-time |
| Is the simulator right? | One real `vllm-metal` replica, KV fixed at 16k tokens, 120 Copilot sessions at their real gaps | Within 1.6 points of the engine in every gap bin (exit criterion 5); hit or miss agrees on 884 of 886 calls |

Per-run and per-day tables behind every row: [results/](results/README.md).

Caveats: many runs paged on the 16 GB host (swap counters are recorded per run); differences under ~15% are not
claimed. The CPU tier is simulated in every mixed-pool result. The autoscaling and failover rows used an earlier GPU
simulator later found 1.8–2.5× faster than the real engine; their reaction times are bound by the metrics pipeline,
not engine speed, and the routing study was rerun with the recalibrated simulator.

## Limits

- One machine, one GPU; no datacenter GPUs and no multi-node scale.
- CPU tier simulated (a real vLLM CPU pod next to the GPU engine pages the host); the 4B CPU tier is an estimate.
- Qwen3-0.6B for most results; hosted-model quality was not compared with the local model.
- KV offloading cannot run on `vllm-metal` (its worker implements no vLLM KV connector), so the RAM and SSD tiers are
  simulated with nominal bandwidths; the simulator was validated on the GPU tier only.
- Prefill/decode disaggregation, scale-from-zero, and a long-running production endpoint are out of scope.

## Memory-risk benchmark

MEMTRACE evaluates persistent-memory risk in tool-using agents. It separates retrieval exposure, poisoned-memory admission, delayed retrieval, unsafe proposals, policy-checker blocking, unsafe execution, and execution-format failure.

### Install

Create an environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional backends:

```bash
python -m pip install -e ".[retrieval]"
python -m pip install -e ".[inference]"  # MLX on Apple Silicon
python -m pip install -e ".[gemini]"     # Gemini API backend; requires GEMINI_API_KEY
```

Gemini calls share a process-wide spend cap (`MEMTRACE_GEMINI_BUDGET_USD`, default `1.0`); a call is refused once
the cap is reached, and models without an entry in `GEMINI_PRICES_USD_PER_MTOK` are refused outright.

Public workload and evaluation datasets (Mooncake, Azure LLM inference, ShareGPT, BFCL) are pinned to fixed
revisions and downloaded into the ignored `data/public/` directory with a SHA-256 manifest:

```bash
python scripts/fetch_public_datasets.py            # or name a subset: mooncake azure_llm sharegpt bfcl
```

CI installs from locked requirements (`requirements/ci-lock.txt`, `requirements/canary-lock.txt`)
instead of floating version resolution, so the frozen regression baseline can't drift from an
unrelated dependency bump. Regenerate a lockfile after changing `pyproject.toml`'s dependencies:

```bash
python -m pip install pip-tools
python -m piptools compile --extra dev --output-file requirements/ci-lock.txt --strip-extras pyproject.toml
python -m piptools compile --extra retrieval --extra gemini --output-file requirements/canary-lock.txt --strip-extras pyproject.toml
```

### CLI

MEMTRACE exposes one command:

```bash
memtrace --help
```

Build and verify benchmark assets:

```bash
memtrace assets build
```

Run a portable profile-backend smoke test:

```bash
memtrace --config configs/profile.toml run pilot \
  --limit 2 \
  --out-dir data/pilot/profile-smoke \
  --force
```

Run the MLX benchmark:

```bash
memtrace --config configs/mlx.toml run benchmark
memtrace evaluate score
memtrace evaluate validate
```

Run against a local OpenAI-compatible server (`mlx_lm.server`, which provides continuous batching and a prompt cache).
Calls are streamed, each trace turn records TTFT/TPOT/end-to-end latency, token usage, and
prefix-cache hits in `inference_calls`, and the run writes latency percentiles and throughput to
`serving.json`. `--concurrency` keeps several episodes in flight to exercise server-side batching:

```bash
python -m mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8000
MEMTRACE_OPENAI_BASE_URL=http://localhost:8000/v1 \
  memtrace --config configs/openai.toml run pilot --concurrency 4 --out-dir data/pilot/served --force
```

`MEMTRACE_OPENAI_API_KEY` is sent as a bearer token when set.

Other workflows:

```bash
memtrace run calibration
memtrace run stateful-stress --dry-run
memtrace run trusted-utility --dry-run
memtrace run adversarial-mutation --dry-run
memtrace evaluate recompute --out artifacts/recomputed
memtrace evaluate audit
memtrace report tables
memtrace report figures
memtrace report explore --label unsafe --ambiguity-reason partial_argument_match
```

Arguments after a workflow name are forwarded to that workflow. Use, for example, `memtrace run pilot --help` for its detailed options.

### Configuration

Configuration files use a `[memtrace]` TOML table. Environment variables override file values:

```toml
[memtrace]
data_dir = "data"
memory_writer_backend = "profile"
planner_backend = "profile"
retrieval_backend = "lexical"
top_k = 5
```

Common overrides include `MEMTRACE_DATA_DIR`, `MEMTRACE_ACTOR_MODELS`, `MEMTRACE_RETRIEVAL_BACKEND`, `MEMTRACE_MEMORY_WRITER_BACKEND`, `MEMTRACE_PLANNER_BACKEND`, `MEMTRACE_PLANNER_PROMPT_PATH`, `MEMTRACE_MEMORY_CONFLICT_RESOLUTION` (`none` default, or `latest_wins_per_task_and_type`; see `configs/profile-latest-wins.toml`), and `MEMTRACE_MEMORY_TTL_TURNS`.

### Development

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m pytest
python -m build
```

CI runs these checks on Python 3.11 and 3.13, smoke-tests the built wheel, and runs a
required regression-gate job that rebuilds a deterministic benchmark slice and diffs its
metrics against a frozen baseline (`tests/fixtures/regression/`).

### Structure

- `src/memtrace/core/`: benchmark domain, schemas, traces, and agent workflows.
- `src/memtrace/backends/`: model, retrieval, storage, and tool adapters.
- `src/memtrace/evaluation/`: scoring, metrics, audit, and reporting.
- `src/memtrace/commands/`: internal implementations behind the CLI.
- `src/memtrace/serving/`: serving benchmark harness (workloads, open-loop replay, metrics) and the hosted-model adapter.
- `src/memtrace/kvmem/`: agent-session KV memory: Copilot trace parser, retention analysis, cost model, and the
  trace-driven KV simulator (`sim/`) with its validation against a real engine.
- `results/`: aggregate result tables, generated by `scripts/results_page.py`.
- `deploy/kind/`, `scripts/`, `epp-plugins/`, `Makefile`: the serving platform (manifests, study scripts, EPP plugins).
- `configs/`: versioned non-secret runtime configurations.
- `tests/`: deterministic tests; real model inference is not required.
- `data/`, `artifacts/`, and `figures/`: ignored runtime outputs.

Extension policy: add a backend behind an existing interface when a second implementation is needed; do not add a service, queue, database server, or deployment layer until a concrete use case requires it.

### Limitations

- Tasks are synthetic enterprise-assistant scenarios.
- The `profile` backend is a deterministic smoke-test fixture.
- Findings from one model or backend configuration should not be generalized across models.
