# MEMTRACE Validation Record

## One-Command Validation

```bash
python scripts/validate_release.py
```

This command validates the anonymous release without model execution.

## One-Command Metric Regeneration

```bash
python scripts/recompute_metrics.py --main data/results --calibration data/calibration --out artifacts/recomputed
```

This command regenerates main and calibration metrics from the packaged run summaries and traces.

## Required Status Table

| Check | Required result | Status |
|---|---:|---|
| Main trace count | 324 | pass |
| Calibration trace count | 72 | pass |
| Total trace count | 396 | pass |
| Main summary rows | 324 | pass |
| Calibration summary rows | 72 | pass |
| Schema validation | 396/396 | pass |
| Retrieval verification | all required records present | pass |
| Metrics regeneration | Tables 2-9 reproduced | pass |
| Figures regeneration | Figures 2-3 reproduced | pass |
| Croissant metadata | validates with RAI fields | pass |
| Dataset card | present | pass |
| Evaluation card | present | pass |
| Third-party assets | documented | pass |
| Anonymity scan | no identifying strings | pass |
| PDF render inspection | no unreadable main tables | pass |
| Page/style compliance | official NeurIPS style | pass |

## Additional Checks

- `scripts/validate_release.py` validates all 396 traces against the committed schema contract.
- `scripts/validate_artifact.py` checks required files, trace uniqueness, stateful causal diagnostics, metric-table equality, and anonymity markers.
- `scripts/build_croissant.py --validate` validates the Croissant metadata, including Responsible AI fields.
- `scripts/make_figures.py --metrics artifacts/recomputed --out figures` regenerates the result figures from committed metrics.
- `THIRD_PARTY_ASSETS.md` documents the referenced model, backend, dependency, and MEMTRACE asset rows.
