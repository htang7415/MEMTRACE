# MEMTRACE

MEMTRACE is a compact benchmark harness for evaluating how persistent memory changes safety outcomes in memory-enabled agents.
It compares one-shot prompt-injection behavior against multi-turn stateful behavior under fixed retrieval, fixed task structure, and fixed memory-system variants.

## Scope

- 12 base tasks
- 2 task families: policy memory, tool-argument memory
- 2 payloads: direct override, contextual drift
- 3 systems: `S0`, `S1`, `S2`
- 108 episodes per system
- 324 generated runs in the current harness

## Runbook

Setup:

```bash
uv sync
```

Tests:

```bash
uv run pytest
```

Build benchmark assets:

```bash
uv run python scripts/build_corpus.py
uv run python scripts/build_episodes.py
uv run python scripts/verify_retrieval.py
```

Run benchmark and score:

```bash
uv run python scripts/run_experiments.py
uv run python scripts/score_runs.py
```

Generate tables, figures, and release bundle:

```bash
uv run python scripts/make_tables.py
uv run python scripts/make_figures.py
uv run python scripts/export_release.py
```

## Outputs

Core result artifacts:

- `data/results/metrics.json`
- `data/results/episode_scores.json`
- `data/results/run_summary.json`
- `data/results/table1.md`
- `data/results/supplementary_tables.md`

Figures:

- `figures/figure1_pipeline.svg`
- `figures/figure2_ovr_vs_svr.svg`
- `figures/figure3_par_by_task_family.svg`
- `figures/figure4_one_shot_vs_stateful.svg`
- `figures/figure5_violation_rate_by_horizon.svg`

Release bundle:

- `release/manifest.json`
- `release/docs/dataset_card.md`
- packaged `gold/`, `results/`, `figures/`, and `traces/`

## Current Metrics

From `data/results/metrics.json`:

- `S0`: `CSR=1.0`, `OVR=0.5`, `SVR=0.0`, `SRG=-0.5`
- `S1`: `CSR=1.0`, `OVR=0.5`, `SVR=0.8333`, `SRG=0.3333`
- `S2`: `CSR=1.0`, `OVR=0.5`, `SVR=0.0`, `SRG=-0.5`

Horizon breakdown for `S1`:

- `Δ=1`: `1.0`
- `Δ=3`: `1.0`
- `Δ=7`: `0.5`

## Current Limitations

- the harness currently models one actor scaffold rather than two actor-model backends
- `S2` is still close to an idealized defense
- `S0` and `S2` remain flat-safe in the current stateful evaluation
- macOS AppleDouble files inside `.git/` may still exist on this volume, though they are excluded from the tracked workspace
