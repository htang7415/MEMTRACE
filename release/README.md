# MEMTRACE Anonymous Artifact

This artifact supports the MEMTRACE NeurIPS Evaluations & Datasets submission.
It contains the v1.0 audited pilot traces, a forced-memory calibration packet, generated metrics, documentation, and a no-model validation harness.

## Contents

- `traces/v1_main_324/`: 324 main traces, with 108 traces each for `S0`, `S1`, and `S2`.
- `traces/calibration_oracle_memory_72/`: 72 forced-memory calibration traces for `S1-ORACLE-RETRIEVED-MEMORY`.
- `results/`: packaged run summaries, episode scores, main metrics, calibration metrics, and attribution reports.
- `tables/`: regenerated JSON metric tables used by the paper.
- `data/`: synthetic corpus, allowlist, episode specifications, and gold labels.
- `github_harness/`: source harness for rebuilding assets and rerunning experiments.
- `scripts/`: artifact-local validation and metric regeneration commands.
- `paper/`: anonymous manuscript source and PDF.
- `RELEASE_MANIFEST.md` and `TRACE_SCHEMA.md`: release counts and trace-row contract.

## Quick Validation

```bash
python scripts/validate_release.py
python scripts/recompute_metrics.py --main data/results --calibration data/calibration --out artifacts/recomputed
python scripts/make_figures.py --metrics artifacts/recomputed --out paper/figures
python scripts/validate_artifact.py
python scripts/aggregate_metrics.py --traces traces/v1_main_324 --out tables/main_metrics.json
python scripts/aggregate_metrics.py --traces traces/calibration_oracle_memory_72 --out tables/calibration_metrics.json
python scripts/build_croissant.py --validate
python scripts/run_smoke_test.py --config configs/scoring.yaml
```

These commands regenerate metrics from packaged traces and do not run the actor model.
Full model generation requires Apple Silicon, `mlx-lm`, `sentence-transformers`, the referenced MLX actor model, and the dense retrieval model.

## Scope

The main paper-facing result evaluates one actor/backend pair: `mlx-community/Qwen2.5-7B-Instruct-4bit` with MLX writer/planner backends.
The artifact contains 396 trace files: 324 main S0/S1/S2 traces and 72 calibration traces.
The main S0/S1/S2 metrics exclude calibration traces.
The calibration packet tests benchmark sensitivity when poisoned memory is inserted and forced into the trigger context.
