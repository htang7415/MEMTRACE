# MEMTRACE

MEMTRACE is a compact benchmark harness for evaluating persistent-memory risk in tool-using agents.
It separates current-turn retrieval exposure, poisoned-memory admission, delayed memory retrieval, unsafe proposal, policy-checker blocking, unsafe execution, and execution-format failure.

## Paper-Facing Scope

- 12 synthetic enterprise-assistant base tasks.
- 2 task families: policy memory and tool-argument memory.
- 2 payload types: direct override and contextual drift.
- 3 systems: `S0`, `S1`, and `S2`.
- 324 main traces: 108 traces per system.
- 72 oracle-retrieved-memory calibration traces for `S1-ORACLE-RETRIEVED-MEMORY`.
- One evaluated actor/backend pair in the audited pilot: `mlx-community/Qwen2.5-7B-Instruct-4bit` with MLX writer/planner backends.

## Setup

Install the no-inference harness and test dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

For MLX-backed model generation on Apple Silicon, install the inference extras:

```bash
python -m pip install -r requirements-inference.txt
```

Run tests:

```bash
python -m pytest
```

## Reproduce From Packaged Traces

The anonymous artifact can regenerate the reported tables without model inference:

```bash
python -m memtrace.validate_release
python -m memtrace.eval.metrics --main data/results --calibration data/calibration/oracle_memory --out artifacts/recomputed
```

After building the anonymous release bundle with `python scripts/export_release.py`, the packaged artifact-local commands are:

```bash
cd release
python scripts/validate_artifact.py
python scripts/aggregate_metrics.py --traces traces/v1_main_324 --out tables/main_metrics.json
python scripts/aggregate_metrics.py --traces traces/calibration_oracle_memory_72 --out tables/calibration_metrics.json
python scripts/build_croissant.py --validate
python scripts/run_smoke_test.py --config configs/scoring.yaml
```

## Regenerate Harness Outputs

Build benchmark assets:

```bash
python scripts/build_corpus.py
python scripts/build_episodes.py
python scripts/build_index.py
python scripts/verify_retrieval.py
```

Run the main benchmark and score traces:

```bash
MEMTRACE_MEMORY_WRITER_BACKEND=mlx MEMTRACE_PLANNER_BACKEND=mlx python scripts/run_experiments.py
python scripts/score_runs.py
python scripts/validate_run.py
```

Run the oracle-retrieved-memory calibration:

```bash
MEMTRACE_MEMORY_WRITER_BACKEND=mlx MEMTRACE_PLANNER_BACKEND=mlx python scripts/run_oracle_memory_calibration.py
```

Prepare or run the optional stateful stress extension:

```bash
python scripts/run_stateful_stress_suite.py --dry-run
MEMTRACE_MEMORY_WRITER_BACKEND=mlx MEMTRACE_PLANNER_BACKEND=mlx python scripts/run_stateful_stress_suite.py
```

Prepare or run the optional trusted-memory utility control:

```bash
python scripts/run_trusted_utility_suite.py --dry-run
MEMTRACE_MEMORY_WRITER_BACKEND=mlx MEMTRACE_PLANNER_BACKEND=mlx python scripts/run_trusted_utility_suite.py
```

Generate tables, figures, audit reports, and release bundle:

```bash
python scripts/make_tables.py
python scripts/make_figures.py
python scripts/attribute_failures.py
python scripts/make_audit.py
python scripts/make_audit_review.py
python scripts/export_release.py
```

## Current Limitations

- The audited pilot evaluates one actor/backend pair and does not make broad cross-model claims.
- The benchmark uses synthetic enterprise-assistant tasks with fixed retrieval and deterministic tools.
- The `profile` backend is a deterministic smoke-test fixture and is not an official result backend.
