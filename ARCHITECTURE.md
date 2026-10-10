# Architecture

MEMTRACE is one Python package (`src/memtrace`) plus a Go gateway, Go scheduling plugins for llm-d, and a
TypeScript dashboard. Every measurement is an experiment spec in `experiments/`; every result is a bundle in
`data/runs/` (git-ignored) in one format; the dashboard publishes them.

## Experiment harness (`memtrace/harness`)

```text
experiments/*.yaml
 └─ spec ─► planner ─► runner
                        ├─ targets    (system under test: lifecycle, endpoints, picker)
                        ├─ workloads  (request streams and load settings)
                        ├─ loadgen    (open loop: Poisson or explicit arrivals; closed loop; agent session replay)
                        ├─ budget     (reserve/commit ledger for paid APIs)
                        └─ provenance (git state, host, tool versions, key redaction)
                      ─► data/runs/<run_id>/{results.json, requests.jsonl, spec.yaml, logs/}
                      ─► compare / perf_gate / dashboard_export
```

`memtrace run experiment SPEC [--set key=value]` runs a spec.

- **Targets** own lifecycle and expose OpenAI-compatible endpoints; they never define workload or SLO policy.
  `vllm_metal`, `llamacpp_replicas` (Metal or CPU), `engine` (vllm-metal, mlx_lm.server, vLLM CPU in Docker via
  `scripts/engines.sh`), `llmd` (llm-d EPP + Envoy in Docker Compose over vllm-metal or inference-sim workers),
  `llmd_kind` (llm-d on kind via `scripts/stack.sh`: simulated replicas or the GPU + CPU pool, an EPP policy per
  trial, hosted overflow, the precise prefix index), `sim_replicas`, `static_endpoints`, `gemini`, and
  `ai_gateway` (wraps any local target).
- **Workloads** produce request streams and never know which engine serves them: `rag_sessions` (multi-turn
  HotpotQA sharing a context), `synthetic_chat`, `trace_replay` (Azure LLM trace 2024 arrivals and lengths),
  `sharegpt`, `mooncake` (prefix blocks rendered as shared text), `agent_sessions` (each session re-sends its
  growing history), and `copilot_agent` (GitHub Copilot sessions replayed with their recorded gaps and cached
  prefixes; each session's calls run in order).
- **Latency** is measured from the scheduled arrival (open loop) or the send (closed loop, session replay), so
  queueing delay is not hidden; rejected and failed requests stay in the denominator. The SLO bounds TTFT,
  end-to-end time, and optionally time per output token.
- **Repeats and CIs**: each cell's metrics are means over repeats with a 95% Student-t interval. Trials record
  whether the host was quiet before they ran (load, foreign engine processes and containers) and how much it
  paged during the run.
- `harness/result.schema.json` is generated from the result dataclasses and is the dashboard contract;
  `python -m memtrace.harness schema --check` fails when it is stale.
- **Spend**: every paid request reserves its worst-case cost in a JSONL ledger outside the repository
  (`~/.memtrace/budget`) before it is sent, then commits provider-reported usage. The Go gateway shares the
  ledger (exclusive flock), so one hard cap holds across languages. The kind cluster's hosted adapter pod
  enforces its own cap (`HOSTED_BUDGET_USD`).
- **Keys** are loaded only at runtime (`memtrace.harness.secrets`); results record key presence, not value.

## Datasets (`memtrace/datasets`)

`datasets/manifest.yaml` pins every file a loader reads by SHA-256, in groups `memtrace data fetch --group`
selects: CRAG, BEIR, ShareGPT, Azure LLM traces (2023, 2024), Mooncake, BFCL, AgentX, the GitHub Copilot
coding-agent traces (June 1–7 2026) and those sessions rendered under each context policy, and BrowseComp-Plus.
Files land in `data/public/`; loaders verify a file before reading it. `loaders/copilot.py` streams the Copilot
daily archives as raw records or typed sessions.

## Serving platform on kind (`deploy/kind`, `epp-plugins/`, `scripts/stack.sh`)

llm-d's own kind environment (pinned v0.11.0) with an InferencePool over the host's vllm-metal (containers on
macOS cannot reach the Metal GPU, so a relay pod fronts it), a CPU-tier llm-d-inference-sim calibrated to an
11× slower engine, and optionally Gemini Flash-Lite through the hosted adapter pod
(`serving/adapters/hosted.py`), which exports vLLM-style metrics so the EPP can score it. `epp-plugins/` adds
capacity-aware scorers (load per unit of capacity, cache-discounted capacity, overflow filter) in its own Go module:
`cmd/epp` registers them in the EPP's public plugin registry and runs llm-d's runner unchanged, so no upstream file
is edited. `make up` brings the stack up; experiments switch policies per trial; `scripts/drills.sh`
holds the failover and autoscaling drills. Prometheus, alert rules and KEDA are in `deploy/kind/`.

## Agent context policies (`memtrace/agents`, `memtrace/evals`)

- `agents/context.py`: what an agent sends each step: `full`, `truncate`, `window`, `mask`, `summarize`, and
  `CacheAware` (`<policy>+cache`): an append-only view re-rendered by the base policy only past a token budget,
  with `min_growth` before the next trim. Property-tested: the task is kept, tool calls stay paired with their
  results, cache-aware views only append between edits.
- `evals/copilot_characterize.py` characterizes the Copilot traces (prompt sizes, tool-output share, context
  cuts, cache hits against pause length).
- `evals/context_eval.py` runs every policy on BrowseComp-Plus tasks with a Gemini agent; cost is what Gemini
  bills, correctness comes from an LLM judge calibrated against `evals/graders/calibration/`.
  `context_regrade.py` adds strict grading and Holm-corrected exact McNemar tests. Answers stay in a local
  `answers.jsonl` (BrowseComp-Plus text must not be published).
- `evals/e5.py` (QA with given context, BFCL tool calls, agentic HotpotQA) and `evals/e6.py` (Gemini implicit vs
  explicit caching vs the Batch API). `serving/quality.py` gates engine changes on BFCL with the same grader.

## KV memory (`memtrace/kv`)

- `kv/retention.py`: how the provider's prompt cache behaved across the gaps of the Copilot week, why reusable
  tokens were recomputed, and what a fixed cache lifetime would keep and hold (`memtrace report retention`).
- `kv/sim.py`: an offline KV-cache simulator over prefix-chained blocks: replicas with a GPU tier and optional
  CPU and SSD tiers, eviction (LRU, pause-aware, turn-aware) and cache lifetimes, placement (round robin,
  least-loaded, session hash, sticky, KV-aware, llm-d-style prefix+load), and steady-state or trace-timed
  arrivals. Sessions come from the AgentX traces, from Copilot sessions built from token counts (append-only
  prompts), or from Copilot sessions rendered under a context policy (`kv/policy_traces.py`).
  `memtrace run kv-sim SPEC` runs a spec.
- `kv/validate.py` replays a real engine's run through the simulator and compares prefix-cache hits per gap
  (K11); `kv/live.py` and `kv/gateway_replay.py` replay the same sessions in wall-clock time through llm-d or the
  gateway onto real engines.

## AI gateway (`gateway/`, Go)

- OpenAI-compatible proxy in front of a local fleet (vLLM replicas or an llm-d gateway) with policies
  `local_only`, `local_first` (overflow beyond an in-flight threshold), `local_first_slo` (overflow when
  in-flight × recent time per request exceeds `slo_ttft_s`), and `remote_only`. Failover retries a local
  connection error or 5xx on the remote before any byte reaches the client.
- Context management (`gateway/internal/ctxmgr`) applies `window+cache` or `mask+cache` in the request path; a
  Python-generated fixture keeps Go and `agents/context.py` identical. Decisions are reported in a response
  header (`X-Memtrace-Context`), Prometheus counters, and span attributes.
- The gateway process is the only holder of the provider key; it is redacted from forwarded errors and never
  appears in metrics or traces. Prometheus metrics (`memtrace_gateway_*`) and OpenTelemetry tracing (W3C
  `traceparent` always propagated; spans exported over OTLP when `OTEL_EXPORTER_OTLP_ENDPOINT` is set).

## Memory-risk benchmark (`memtrace/memrisk`)

Follows a poisoned document through an agent's persistent memory (planner, writer, retrieval, memory store) to
find where an attack breaks; configs in `configs/memrisk/`, a deterministic regression gate in CI, and its own
report in `results/benchmark/`.

## Observability and dashboard

- `deploy/observability/`: OpenTelemetry collector → Jaeger, Prometheus scraping the gateway, the llm-d EPP,
  vLLM and llama.cpp, and Grafana with a provisioned serving dashboard (Compose; every port on 127.0.0.1).
- `dashboard/`: a static Vite + React + Recharts site. `memtrace report dashboard` exports the latest complete
  run of each published experiment from `data/runs/` (validated against the result model), plus aggregates of
  the analyses that are not harness runs; TypeScript types are generated from the result JSON Schema, so schema
  drift fails the build.

## Components (`components.lock`)

Every external component MEMTRACE runs against is pinned in one KEY=value file: the `vllm-metal` and vLLM versions,
the llm-d router tag, container images with their digests, Helm charts, model revisions, and the tool versions last
used. Scripts source it (`scripts/images.sh`); `memtrace components check` compares it with this machine, and CI
fails when a manifest, script, or `epp-plugins/go.mod` names another version. Upstream packages are used through
their public interfaces only: flags, metrics, KV events, plugin registries. Nothing in them is patched, except the
simulator image K5 and K7 used (`deploy/inference-sim`), kept only to reproduce those runs.

## Local files

Git-ignored, next to the code: `data/public/` (pinned datasets), `data/runs/` (every result bundle), `data/` analyses
(retention, BFCL gate, drills), and `models/` (model weights: `HF_HOME` is `models/huggingface` when unset, set by
`memtrace` and the shell scripts; GGUF files in `models/gguf/`). Only the dashboard's exported snapshot, the results
page, and the benchmark report are published.
