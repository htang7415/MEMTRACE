# MEMTRACE Anonymous Artifact

## 1. What MEMTRACE Is

MEMTRACE is a validity-first benchmark and protocol for separating immediate retrieval-context violations, poisoned-memory admission, trigger-time memory retrieval, unsafe proposal, policy-check blocking, unsafe execution, and execution-format failure in memory-enabled tool agents.
This anonymous artifact supports the MEMTRACE NeurIPS Evaluations & Datasets submission.
It contains the v1.0 audited pilot traces, a separate forced-memory calibration packet, generated metrics, documentation, and a no-model validation harness.

## 2. What Claims This Artifact Supports

The artifact supports a narrow evaluation claim for one actor/backend pair: `mlx-community/Qwen2.5-7B-Instruct-4bit` with MLX writer/planner backends.
It shows that one-shot unsafe execution and poisoned-memory admission can be separated from delayed stateful unsafe execution under fixed retrieval, deterministic tools, declared validity gates, and trace-level scoring.
The calibration packet shows scorer and trace-protocol sensitivity when poisoned memory is forced into the trigger context.

## 3. What Claims This Artifact Does Not Support

The artifact does not claim broad cross-model prevalence, absence of persistent-memory risk, broad model ranking, or defense superiority.
`S2` is a provenance-aware reference writer, not a complete deployed defense.
Calibration traces are excluded from the main S0/S1/S2 rates.

## 4. File Layout

- `traces/v1_main_324/`: 324 main traces, with 108 traces each for `S0`, `S1`, and `S2`.
- `traces/calibration_oracle_memory_72/`: 72 forced-memory calibration traces for `S1-ORACLE-RETRIEVED-MEMORY`.
- `results/`: packaged run summaries, episode scores, main metrics, calibration metrics, and attribution reports.
- `tables/`: regenerated JSON metric tables used by the paper.
- `data/`: synthetic corpus, allowlist, episode specifications, and gold labels.
- `github_harness/`: source harness for rebuilding assets and rerunning experiments.
- `scripts/`: artifact-local validation and metric regeneration commands.
- `paper/`: anonymous manuscript source and PDF.
- `VALIDATION.md`, `RELEASE_MANIFEST.md`, and `TRACE_SCHEMA.md`: validation record, release counts, and trace-row contract.

## 5. Reproduce Metrics Without Model Execution

```bash
python scripts/recompute_metrics.py --main data/results --calibration data/calibration --out artifacts/recomputed
python scripts/aggregate_metrics.py --traces traces/v1_main_324 --out tables/main_metrics.json
```

These commands regenerate metrics from packaged traces and do not run the actor model.

## 6. Validate All Traces and Metadata

```bash
python scripts/validate_release.py
python scripts/validate_artifact.py
python scripts/build_croissant.py --validate
```

`scripts/validate_release.py` checks the 324 main traces, 72 calibration traces, 396 total traces, run-summary counts, schema contract, locked paper counts, calibration values, pilot gates, audit reconciliation, and packaged tables.

## 7. Reproduce Calibration Metrics

```bash
python scripts/recompute_metrics.py --main data/results --calibration data/calibration --out artifacts/recomputed
python scripts/aggregate_metrics.py --traces traces/calibration_oracle_memory_72 --out tables/calibration_metrics.json
```

The calibration condition is `S1-ORACLE-RETRIEVED-MEMORY`.
It is excluded from all main S0/S1/S2 rates.

## 8. Inspect Audit Packet

The audit packet is in `audit/`.
It contains 40 scorer-audit comparisons after frozen-scorer reconciliation.
The retained artifact does not record the trace-sampling rule, so the audit packet is a scorer-consistency check rather than a random or stratified reliability estimate.

## 9. Known Limitations

The v1.0 audited pilot evaluates one actor/backend pair over synthetic enterprise-assistant tasks.
Original model-generation jobs retain actor/backend metadata but not exact worker, wall-clock runtime, or peak-memory telemetry.
Full model reruns require Apple Silicon, `mlx-lm`, `sentence-transformers`, the referenced MLX actor model, and the dense retrieval model.

## 10. Licenses and Third-Party Assets

MEMTRACE synthetic assets and harness code are under the anonymous MIT review license.
External models and dependencies are documented in `THIRD_PARTY_ASSETS.md`.
Model weights are referenced, not redistributed.

## Command Summary

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
