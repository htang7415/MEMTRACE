# MEMTRACE

MEMTRACE evaluates persistent-memory risk in tool-using agents. It separates retrieval exposure, poisoned-memory admission, delayed retrieval, unsafe proposals, policy-checker blocking, unsafe execution, and execution-format failure.

## Install

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

CI installs from locked requirements (`requirements/ci-lock.txt`, `requirements/canary-lock.txt`)
instead of floating version resolution, so the frozen regression baseline can't drift from an
unrelated dependency bump. Regenerate a lockfile after changing `pyproject.toml`'s dependencies:

```bash
python -m pip install pip-tools
python -m piptools compile --extra dev --output-file requirements/ci-lock.txt --strip-extras pyproject.toml
python -m piptools compile --extra retrieval --extra gemini --output-file requirements/canary-lock.txt --strip-extras pyproject.toml
```

## CLI

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

## Configuration

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

## Development

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

## Structure

- `src/memtrace/core/`: benchmark domain, schemas, traces, and agent workflows.
- `src/memtrace/backends/`: model, retrieval, storage, and tool adapters.
- `src/memtrace/evaluation/`: scoring, metrics, audit, and reporting.
- `src/memtrace/commands/`: internal implementations behind the CLI.
- `configs/`: versioned non-secret runtime configurations.
- `tests/`: deterministic tests; real model inference is not required.
- `data/`, `artifacts/`, and `figures/`: ignored runtime outputs.

Extension policy: add a backend behind an existing interface when a second implementation is needed; do not add a service, queue, database server, or deployment layer until a concrete use case requires it.

## Limitations

- Tasks are synthetic enterprise-assistant scenarios.
- The `profile` backend is a deterministic smoke-test fixture.
- Findings from one model or backend configuration should not be generalized across models.
