# MEMTRACE

**[View the results dashboard →](https://htang7415.github.io/MEMTRACE/)** · [Full results](results/README.md) · [Architecture](ARCHITECTURE.md)

MEMTRACE studies how to serve LLM agents efficiently on one Apple Silicon Mac. Agents re-send a growing history
every step, and engines are fast only when that history is already in the KV cache. MEMTRACE measures that
trade-off with real engines (vLLM on the Apple GPU via `vllm-metal`, llama.cpp, mlx_lm), llm-d request
scheduling, a Go AI gateway in front of a local fleet and the Gemini API, a week of GitHub Copilot agent
traffic, and a KV-cache simulator checked against a real engine. Every result reports what was measured on
real engines and what was simulated, with 95% confidence intervals where runs were repeated.

## Results

| Question | Finding |
| --- | --- |
| Which engine serves the GPU tier? | `vllm-metal`: 455 tok/s on ShareGPT at concurrency 8, 2.2× `mlx_lm.server` (one run per cell, E7); on the same Q8 GGUF it is within run-to-run noise of llama.cpp on Metal (3 repeats, E1); tool-call accuracy 81.0–81.5% on every engine and batch size (BFCL, 600 cases) |
| Does cache-aware routing pay? | 1.8–2.3× the throughput of random routing on agent sessions (llm-d over simulated replicas calibrated to the real engine; one run per cell, E8) |
| How should a GPU + CPU pool be routed? | Where the GPU is not cache-bound, a capacity-aware scorer gives 1.6–2.3× the default's throughput (single-run sweeps, E9b–c); over 6 repeats on agent sessions at concurrency 8 the two are within noise (62.7 vs 57.4 tok/s, E9) |
| Does hosted overflow help? | An overloaded local pool goes from 1.77 to 5.1–5.7 req/s and TTFT p95 from 13.5 s to ~5 s, at $0.27–0.30 per 1k requests (Gemini Flash-Lite in the llm-d pool; 3 repeats, E10) |
| Which policy for real agent traffic? | On replayed Copilot sessions, no routing policy is clearly better with Qwen3-0.6B (3 repeats, overlapping CIs, E11); with Qwen3-4B at 16 sessions, capacity-aware scorers cut errors from 11 to about 1 per run (E11b) |
| What do production agents send? | A 57k-token median prompt across 9.1M Copilot calls; tool output is 48% of prompt tokens |
| Should agents trim their context? | Naive trimming makes the engine recompute up to 2.6× more prefill; trimming only past a token budget keeps the prompt append-only (simulation, 3 repeats; ranking checked once through llm-d and once on `vllm-metal`; K6–K8, C1) |
| Can the gateway do it for every client? | On Copilot traffic onto Qwen3-8B, the gateway's `window+cache` cut recomputed prefill by 14–39% in all 4 paired runs (2 days × 2 repeats); `mask+cache` ranged from −31% to +37% (K9, K9b). In front of Gemini on 100 paired agent tasks: −50% prompt tokens [−80%, −20%], but cost only −12% [−36%, +13%] because trimmed prompts lose the provider's cache; accuracy 52% vs 44%, not a significant difference (C2) |
| How long should agent KV be kept? | The provider recomputed 8–12% of reusable prompt tokens. With every call routed to its cache, a 5 min / 1 h / 24 h lifetime serves 96 / 99.4 / 100% of the reusable prefix for about 1× / 4× / 30× the memory |
| What buys the most cache reuse? | In simulation of a Copilot day on 64 replicas: placement first (least-loaded keeps 5–11% of the reusable prefix, cache-affine placement 89–98%), then GPU cache size, then host RAM (97% with 256 GB per replica at 128 GB of GPU cache); retention policy matters least, as expected once capacity evicts blocks long before a 1 h lifetime ends (K10) |
| Is the simulator right? | Within 0.7 points of a real `vllm-metal` engine in every gap bin, for one replica's GPU cache (one run, K11) |

Details, definitions, and every run: [full results](results/README.md) and the [dashboard](https://htang7415.github.io/MEMTRACE/).

## Getting started

Requires Python 3.12+, Go, and Node 24. The serving platform also needs Docker Desktop (8 GB VM), `kind`,
`kubectl`, `helm`, and [`vllm-metal`](https://github.com/vllm-project/vllm-metal).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --require-hashes -r requirements/dev.lock && pip install --no-deps -e .

memtrace data fetch                                   # public datasets, pinned and SHA-256 checked
memtrace run experiment experiments/e2_prefix_caching.yaml   # any spec in experiments/; results in data/runs/
memtrace run kv-sim experiments/k10a_copilot_reuse_routing.yaml
cd dashboard && npm ci && npm run data && npm run dev # view the results
```

The serving platform on kind:

```bash
make up            # llm-d on kind with the GPU + CPU pool (about 3 minutes)
make bench         # replay Copilot agent traffic through three routing policies (E11)
make down
```

Datasets (`data/public/`), result bundles (`data/runs/`) and model weights (`models/`) stay in git-ignored folders of
the repository. `memtrace` and the scripts point Hugging Face at `models/huggingface` (an explicit `HF_HOME` wins);
GGUF files go in `models/gguf/`.

Paid-API runs read the key only at runtime (`GEMINI_API_KEY`) and reserve each request's worst-case cost against
a hard cap before sending it (`configs/pricing/gemini.yaml`).

## Upgrading components

`components.lock` pins every external component: `vllm-metal` and vLLM, the llm-d router, container images and
their digests, Helm charts, and model revisions. To try a new release, change it there, then:

```bash
memtrace components check   # this machine against the lock (engines, images, models, tools)
```

and rerun the experiments it affects; CI checks that `deploy/`, `scripts/` and `epp-plugins/go.mod` agree with the lock.

## Layout

| Path | What it is |
| --- | --- |
| `src/memtrace/harness` | Experiment specs, targets (engines, llm-d, the gateway, Gemini), workloads, load generation, 95% CIs, provenance |
| `src/memtrace/kv` | Provider-cache retention analysis, KV-cache simulator, live replays onto real engines |
| `src/memtrace/agents`, `evals` | Agent loops and context policies; accuracy and cost studies with a calibrated LLM judge |
| `src/memtrace/datasets`, `serving` | Pinned datasets and loaders; prompt generators, BFCL engine gate, hosted adapter |
| `gateway/`, `epp-plugins/` | Go AI gateway; custom llm-d scorers |
| `deploy/`, `scripts/` | kind and Docker Compose stacks, observability; stack and drill scripts |
| `experiments/` | Every study as a spec |
| `dashboard/` | Results site |

## Limitations

- One machine: no datacenter GPUs, no multi-node scale; small models (Qwen3-0.6B to 8B).
- The CPU tier is simulated; the host often paged, so differences under ~15% are not claimed.
- The `vllm-metal` release used here (0.30.0) cannot offload KV, so RAM and SSD cache tiers were studied in
  simulation only. SSD offload has since landed in `vllm-metal`; measuring it is the next step.
- The KV simulator assumes append-only prompts for the Copilot traces and idealized placement; it is checked
  against a real engine for one replica's GPU cache.

## License

[MIT](LICENSE)
