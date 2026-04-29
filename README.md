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
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
```

Tests:

```bash
python -m pytest
```

Build benchmark assets:

```bash
python scripts/build_corpus.py
python scripts/build_episodes.py
python scripts/build_index.py
python scripts/verify_retrieval.py
```

Run benchmark and score:

```bash
MEMTRACE_MEMORY_WRITER_BACKEND=mlx MEMTRACE_PLANNER_BACKEND=mlx python scripts/run_experiments.py
python scripts/score_runs.py
python scripts/validate_run.py
```

Protocol requirements for official benchmark runs:

- build the corpus, episodes, and dense retrieval index first
- use the MLX writer and planner backends declared in `Project.md`
- do not reuse prior traces generated under a different backend or protocol version

Generate tables, figures, and release bundle:

```bash
python scripts/make_tables.py
python scripts/make_figures.py
python scripts/attribute_failures.py
python scripts/make_audit.py
python scripts/make_audit_review.py
python scripts/make_paper_brief.py
python scripts/make_paper_tables.py
python scripts/make_paper_figures.py
python scripts/export_release.py
python scripts/check_neurips_readiness.py
```

For submission, place the official `neurips_2026.sty` file in `paper/`; `paper/main.tex` uses it automatically when present and otherwise falls back to local compile geometry.

Promote a validated result directory into the paper-facing result paths:

```bash
python scripts/promote_results.py --source-dir data/results_qwen_toolfix_v1
```

## Current Limitations

- official experiments require Apple Silicon plus `mlx-lm` and a built dense index
- the `profile` backend is a deterministic test fixture and is not valid for paper results
- macOS AppleDouble files inside `.git/` may still exist on this volume, though they are excluded from the tracked workspace
