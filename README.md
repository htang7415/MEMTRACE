# MEMTRACE

**[View the report →](https://htang7415.github.io/MEMTRACE/results/report/)** · [Full results](results/README.md) · [Memory-risk benchmark report](https://htang7415.github.io/MEMTRACE/results/benchmark/memtrace_results.html)

MEMTRACE studies how to serve LLM agents efficiently on a single Apple Silicon Mac: how to route requests across
GPU, CPU, and hosted tiers, and how much KV cache an agent session needs, for how long, and where.

## Overview

- **Serving platform.** An llm-d control plane on Kubernetes (kind) routes requests across vLLM on the Apple Silicon
  GPU, a simulated vLLM CPU tier, and optional Gemini Flash-Lite overflow, with custom scheduling plugins,
  Prometheus alerts, and a one-command setup.
- **Agent-session KV memory.** A week of real GitHub Copilot agent traffic (301k sessions), a trace-driven KV-cache
  simulator, and a check of that simulator against a real engine.
- **Memory-risk benchmark.** Follows a poisoned document through an agent's persistent memory to find where the
  attack breaks. This is where the project started; it has its own report (linked above).

Every result says what was measured on real engines and what was simulated.

## Results

| Question | Finding |
| --- | --- |
| Which engine serves the GPU tier? | `vllm-metal`: 455 tok/s on ShareGPT at concurrency 8, 2.2× `mlx_lm.server`; tool-call accuracy unchanged (81.0–81.5% on BFCL) across engines and batch sizes |
| Does cache-aware routing pay? | 1.8–2.3× the throughput of random routing on agent sessions (simulated replicas calibrated to the real engine) |
| How should a GPU + CPU pool be routed? | A capacity-aware scorer gives 1.6–2.3× the default's throughput where the GPU is not cache-bound and removes the timeouts caused by the 11× slower tier |
| Does hosted overflow help? | An overloaded local pool goes from 1.77 to 5.1–5.7 req/s and TTFT p95 from 13.5 s to ~5 s, at $0.27–0.30 per 1k requests |
| Which policy for real agent traffic? | On replayed Copilot sessions, capacity scorers finish 11–13% sooner but double TTFT p95; with a 4B model they win both |
| How long should agent KV be kept? | The provider recomputed 8–12% of reusable prompt tokens. With every call routed to its cache, a 5 min / 1 h / 24 h lifetime serves 96 / 99.4 / 100% of the reusable prefix for about 1× / 4× / 28× the memory |
| What buys the most cache reuse? | In simulation: routing each call back to its cache first, then a larger GPU cache, then host RAM (97–99% with 256 GB per replica); retention policy matters least |
| Is the simulator right? | Within 1.6 points of a real `vllm-metal` engine in every gap bin, for one replica's GPU cache |

Details, definitions, and every run: [full results](results/README.md).

## Getting started

Requirements: an Apple Silicon Mac, Docker Desktop (8 GB VM), `kind`, `kubectl`, `helm`, and
[`vllm-metal`](https://github.com/vllm-project/vllm-metal).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

make data          # download the public datasets (pinned and checksummed)
make up            # start the serving stack (about 3 minutes)
make bench         # replay Copilot agent traffic through three routing policies
make down          # stop everything
```

Analyze and simulate agent KV memory:

```bash
memtrace report retention --by-day               # provider cache behavior on the Copilot traces
memtrace report kv-sim --day 2026-06-03 --replicas 64 --out sweep.jsonl
memtrace report kv-validate --help               # compare the simulator with a real engine run
```

Run the memory-risk benchmark and its regression gate:

```bash
memtrace assets build
memtrace run pilot --help
memtrace evaluate gate --help
```

## Limitations

- One machine: no datacenter GPUs, no multi-node scale; Qwen3-0.6B for most results.
- The CPU tier is simulated; the host often paged, so differences under ~15% are not claimed.
- `vllm-metal` cannot offload KV, so RAM and SSD cache tiers were studied in simulation only.
- The KV simulator is idealized: one cached prefix per session, no prefixes shared across sessions, simplified
  routers, no queueing.

## License

[MIT](LICENSE)
