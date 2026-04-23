# MEMTRACE

MEMTRACE is a compact benchmark harness for evaluating how persistent memory changes safety outcomes in memory-enabled agents.
It compares one-shot prompt-injection behavior against multi-turn stateful behavior under fixed retrieval, fixed task structure, and fixed memory-system variants.

## Scope

- 12 base tasks
- 2 task families: policy memory, tool-argument memory
- 2 payloads: direct override, contextual drift
- 3 systems: `S0`, `S1`, `S2`
- 108 episodes per system
- 648 generated runs across 2 actor models

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

Protocol requirements for official benchmark runs:

- build the corpus, episodes, and dense retrieval index first
- use the MLX writer and planner backends declared in `project.md`
- do not reuse prior traces generated under a different backend or protocol version

Generate tables, figures, and release bundle:

```bash
uv run python scripts/make_tables.py
uv run python scripts/make_figures.py
uv run python scripts/export_release.py
```

## Current Limitations

- official experiments require Apple Silicon plus `mlx-lm` and a built dense index
- the `profile` backend is a deterministic test fixture and is not valid for paper results
- macOS AppleDouble files inside `.git/` may still exist on this volume, though they are excluded from the tracked workspace
