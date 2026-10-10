# Contributing

## Setup

Python 3.12 or newer, Go (the version in `gateway/go.mod`), and Node 24.

```bash
python3 -m venv .venv && . .venv/bin/activate
python -m pip install --require-hashes -r requirements/dev.lock
python -m pip install --no-deps -e .
```

Experiments on the Mac GPU use vLLM with the `vllm-metal` plugin from a separate environment (default
`~/.venv-vllm-metal/bin/vllm`). Model weights belong in the repository's git-ignored `models/` folder, not the user's
cache: `memtrace` and the scripts set `HF_HOME=models/huggingface` when it is unset, and GGUF files go in
`models/gguf/` (the specs reference them there). The serving platform on kind needs Docker Desktop, `kind`, `kubectl` and `helm`
(`make up`).

Keep the virtual environment on an internal disk if you can. On exFAT or other external volumes, macOS writes
`._*` AppleDouble files that break Python imports and pytest collection; remove them before testing or committing
(`find . -name '._*' -not -path './.git/*' -not -path './data/*' -not -path '*/node_modules/*' -delete`).

Datasets are downloaded into `data/public/` (git-ignored) and checked against pinned SHA-256 values:

```bash
memtrace data fetch                      # every group
memtrace data fetch --group copilot      # one group (repeat --group for more)
memtrace data verify
```

## Validate changes

These mirror the `ci` workflow:

```bash
python -m ruff check . && python -m ruff format --check .
python -m mypy
python -m memtrace.harness schema --check
python -m pytest -q
(cd gateway && gofmt -l . && go vet ./... && go test -race ./...)
shellcheck -x -P SCRIPTDIR -S warning scripts/*.sh
(cd dashboard && npm ci && npm run types && npx vitest run && npm run build)
```

Optional local checks: `cd dashboard && npm run data && npm run e2e` (Playwright over the exported results), and
the CPU performance smoke runs in `experiments/ci_smoke_*.yaml` followed by
`python -m memtrace.harness.perf_gate experiments/perf_baseline.yaml artifacts/ci/*/results.json`.

Keep changes focused and add tests for behavior changes. When the result dataclasses change, regenerate the
schema (`python -m memtrace.harness schema --write`) and the dashboard types (`npm run types`).

Every measurement is an experiment spec in `experiments/`; results go to `data/runs/` (git-ignored).
`dashboard/public/data/` is the published snapshot: refresh it with `memtrace report dashboard` after a new
published run and commit it; the `pages` workflow redeploys the site. Experiments that call paid APIs must go
through the spend ledger (`memtrace.harness.budget`) and never run in pull-request CI.

After changing dependencies, regenerate the locks with Python 3.12:

```bash
python -m piptools compile --extra dev --extra agents --allow-unsafe --generate-hashes --strip-extras \
  --output-file requirements/dev.lock pyproject.toml
python -m piptools compile --extra gemini --extra retrieval --allow-unsafe --strip-extras \
  --output-file requirements/canary.lock pyproject.toml
```

The canary lock pins versions without hashes: hashing every torch wheel downloads about 16 GB.

## CI

All checks run in `.github/workflows/ci.yml` (name `ci`) on pushes to `main`, pull requests, a daily schedule,
and manual dispatch.

| Job | What it checks |
| --- | --- |
| `python` | Ruff (lint and format), mypy, result schema is current, pytest on Python 3.12 and 3.13 with hash-locked dependencies (Go installed so the gateway end-to-end tests run); the wheel builds and installs |
| `go` | `gofmt`, `go vet`, `go test -race ./...` in `gateway/` |
| `epp-plugins` | The EPP module (MEMTRACE's scorers registered on llm-d's unmodified runner): `gofmt`, `go vet`, `go test`, the binary builds |
| `shellcheck` | The stack and drill scripts |
| `dashboard` | `npm ci`; generated TypeScript types match the result schema; Vitest; production build |
| `regression-gate` | The memory-risk benchmark's deterministic pilot slice against a frozen baseline |
| `perf-smoke` | `experiments/ci_smoke_sim.yaml` (llm-d-inference-sim) and `experiments/ci_smoke_llamacpp.yaml` (pinned llama.cpp CPU build, Qwen3-0.6B Q8_0, both SHA-256 verified), then the performance gate |
| `canary` | Scheduled or manual only, never blocking: the memory-risk benchmark's hosted-model path against Gemini, with the `GEMINI_API_KEY` repository secret (skipped without it) |

`experiments/perf_baseline.yaml` bounds each smoke experiment's cell means. The inference simulator has a fixed
latency model, so its bounds are tight; CPU inference on shared runners varies, so llama.cpp has a floor of 6
tokens/s and a 5 s TTFT p50 ceiling, which only a roughly 2× regression crosses. Any request error or failed trial
fails the gate. Recalibrate by running both experiments on the target runner and recording the observed values in
the file's comments.

To protect `main`, require a pull request and the status checks `ci / python (3.12)`, `ci / python (3.13)`,
`ci / go`, `ci / epp-plugins`, `ci / shellcheck`, `ci / dashboard`, `ci / regression-gate` and `ci / perf-smoke`.
