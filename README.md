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

CI runs these checks on Python 3.11 and 3.13 and smoke-tests the built wheel.

## Structure

- `src/memtrace/core/`: benchmark domain, schemas, traces, and agent workflows.
- `src/memtrace/backends/`: model, retrieval, storage, and tool adapters.
- `src/memtrace/evaluation/`: scoring, metrics, audit, and reporting.
- `src/memtrace/commands/`: internal implementations behind the CLI.
- `configs/`: versioned non-secret runtime configurations.
- `tests/`: deterministic tests; real model inference is not required.
- `data/`, `artifacts/`, and `figures/`: ignored runtime outputs.

See [the system design note](docs/memtrace_system_design.md) for dependency boundaries, pipeline, evaluation design, and extension policy.

## Limitations

- Tasks are synthetic enterprise-assistant scenarios.
- The `profile` backend is a deterministic smoke-test fixture.
- Findings from one model or backend configuration should not be generalized across models.
