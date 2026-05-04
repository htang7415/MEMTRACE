# Reproduce MEMTRACE Results

The release contains the audited v1.0 trace set and can regenerate tables from traces without running the actor model.
Full model generation requires Apple Silicon, `mlx-lm`, the referenced MLX actor model, and the dense retrieval model.

## Setup For No-Inference Validation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For full MLX-backed model generation from the source repository, also install:

```bash
python -m pip install -r requirements-inference.txt
```

## Validate Packaged Results

From the repository root:

```bash
python -m memtrace.validate_release
python -m memtrace.eval.metrics --main data/results --calibration data/calibration/oracle_memory --out artifacts/recomputed
python scripts/validate_release.py
python scripts/recompute_metrics.py --main data/results --calibration data/calibration/oracle_memory --out artifacts/recomputed
```

From the packaged artifact root, such as `release`:

```bash
python scripts/validate_artifact.py
python scripts/aggregate_metrics.py --traces traces/v1_main_324 --out tables/main_metrics.json
python scripts/aggregate_metrics.py --traces traces/calibration_oracle_memory_72 --out tables/calibration_metrics.json
python scripts/compute_confidence_intervals.py --metrics tables/main_metrics.json --out tables/confidence_intervals.json
python scripts/run_smoke_test.py --config configs/scoring.yaml
python scripts/build_croissant.py --validate
```

## Rebuild the Full Harness Outputs

```bash
python github_harness/scripts/build_corpus.py
python github_harness/scripts/build_episodes.py
python github_harness/scripts/build_index.py
python github_harness/scripts/verify_retrieval.py
MEMTRACE_MEMORY_WRITER_BACKEND=mlx MEMTRACE_PLANNER_BACKEND=mlx python github_harness/scripts/run_experiments.py
python github_harness/scripts/score_runs.py
python github_harness/scripts/make_tables.py
```

The deterministic `profile` backend is a smoke-test fixture and is not an official result backend.
The oracle-retrieved-memory calibration runner is `github_harness/scripts/run_oracle_memory_calibration.py`; the packaged calibration metrics are generated from the complete 72-episode MLX calibration trace set.
