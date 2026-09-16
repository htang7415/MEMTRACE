# Regression fixture

`metrics.json` is a frozen baseline produced by the deterministic `profile` backend
pilot slice (`configs/profile.toml`, default task IDs `budget-limit-rule` and
`meeting-time`, 54 episodes). It is checked in so CI can detect unintended changes to
benchmark generation, the agent pipeline, or scoring/metrics logic without running any
model inference.

CI (`.github/workflows/ci.yml`, `regression-gate` job) regenerates the same slice fresh
on every run and compares it against this file with `memtrace evaluate gate`. A
byte-for-byte identical rerun is expected — the `profile` backend and lexical retrieval
are fully deterministic, so any diff means real pipeline behavior changed and needs a
reviewed, deliberate update to this fixture (not a config or environment fluke).

## Regenerating

```bash
export MEMTRACE_ROOT=/tmp/memtrace-regression-fixture
rm -rf "$MEMTRACE_ROOT"
memtrace --config configs/profile.toml assets corpus
memtrace --config configs/profile.toml assets episodes
memtrace --config configs/profile.toml assets verify
memtrace --config configs/profile.toml run pilot --out-dir "$MEMTRACE_ROOT/pilot" --force
cp "$MEMTRACE_ROOT/pilot/metrics.json" tests/fixtures/regression/metrics.json
```

Only update this file when the metrics difference is an intended, reviewed change.
