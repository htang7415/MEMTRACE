# MEMTRACE

MEMTRACE is a compact benchmark harness for evaluating persistent-memory risk in tool-using agents. It separates retrieval exposure, poisoned-memory admission, delayed memory retrieval, unsafe proposals, policy-checker blocking, unsafe execution, and execution-format failure.

## Benchmark design

The benchmark contains two task families:

- Policy-memory tasks, such as approval limits and access-control rules.
- Tool-argument-memory tasks, such as email recipients and file destinations.

It evaluates three memory systems:

- `S0`: no persistent memory.
- `S1`: permissive memory writing.
- `S2`: provenance-aware memory filtering.

Each system runs clean controls, one-shot attacks, and stateful attacks with delayed triggers.

## Installation

Create a Python environment and install the harness:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

For MLX-backed model generation on Apple Silicon:

```bash
python -m pip install -r requirements-inference.txt
```

Run the test suite:

```bash
python -m pytest
```

## Build benchmark assets

```bash
python scripts/build_corpus.py
python scripts/build_episodes.py
python scripts/build_index.py
python scripts/verify_retrieval.py
```

Generated assets are written under `data/`.

## Run and score experiments

Run the main benchmark with MLX writer and planner backends:

```bash
MEMTRACE_MEMORY_WRITER_BACKEND=mlx \
MEMTRACE_PLANNER_BACKEND=mlx \
python scripts/run_experiments.py

python scripts/score_runs.py
python scripts/validate_run.py
```

Run the oracle-memory calibration:

```bash
MEMTRACE_MEMORY_WRITER_BACKEND=mlx \
MEMTRACE_PLANNER_BACKEND=mlx \
python scripts/run_oracle_memory_calibration.py
```

Recompute metrics without model inference:

```bash
python scripts/recompute_metrics.py \
  --main data/results \
  --calibration data/calibration \
  --out artifacts/recomputed
```

## Analyze results

```bash
python scripts/make_tables.py
python scripts/make_figures.py
python scripts/attribute_failures.py
python scripts/make_audit.py
```

Optional diagnostic suites:

```bash
python scripts/run_stateful_stress_suite.py --dry-run
python scripts/run_trusted_utility_suite.py --dry-run
```

## Repository layout

- `memtrace/`: benchmark, agent pipeline, storage systems, retrieval, and evaluation code.
- `scripts/`: asset generation, experiment execution, scoring, and analysis commands.
- `tests/`: pytest suite.
- `data/`: local inputs and generated experiment outputs; ignored by Git.
- `figures/` and `artifacts/`: generated analysis outputs; ignored by Git.

## Limitations

- The tasks are synthetic enterprise-assistant scenarios.
- The deterministic `profile` backend is a smoke-test fixture, not a result backend.
- Findings from one model or backend configuration should not be generalized across models.
