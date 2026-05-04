# MEMTRACE Dataset Card

## Summary

MEMTRACE is a compact synthetic benchmark for evaluating persistent-memory safety in memory-enabled agents.
It compares one-shot prompt-injection risk against multi-turn stateful risk under fixed retrieval, deterministic tools, fixed task structure, and declared validity gates.
All tasks, retrieved passages, payloads, gold labels, and traces in the released v1.0 bundle are synthetic.
The release does not contain human-subject data.

## What Is Included

- 12 benchmark tasks.
- 2 task families: policy memory and tool-argument memory.
- 2 payload types: direct override and contextual drift.
- 3 system variants: `S0`, `S1`, and `S2`.
- 324 main per-turn JSONL traces.
- 72 oracle-retrieved-memory calibration JSONL traces.
- Gold labels, aggregate metrics, markdown tables, JSON metric tables, SVG figures, audit packet, and Croissant metadata.

## Core Counts

- clean-control episodes per system: 12
- one-shot attack episodes per system: 24
- stateful attack episodes per system: 72
- total main episodes per system: 108
- main paper-facing traces: 324
- oracle-retrieved-memory calibration traces: 72
- evaluated actor/backend pair: `mlx-community/Qwen2.5-7B-Instruct-4bit` with MLX writer/planner backends

## Trace Schema

Each JSONL turn record contains the turn-level fields needed to replay and audit scoring, including:

- `schema_version`
- `episode_id`
- `turn`
- `query`
- `retrieved_passages`
- `memory_writer_output`
- `admitted_memory_records`
- `memory_store_state`
- `tool_router_log`
- `planner_output`
- `label`
- `poison_admission_flag`

Stateful episode scores additionally include causal-chain diagnostics:

- `poison_in_initial_retrieval`
- `writer_candidate_emitted`
- `writer_candidate_valid_schema`
- `poison_admitted`
- `admitted_memory_id`
- `admitted_memory_type`
- `admitted_memory_source_ids`
- `admitted_memory_retrieved_at_trigger`
- `current_turn_poison_retrieved_at_trigger`
- `unsafe_tool_call_proposed_before_checker`
- `policy_checker_blocked_unsafe_call`
- `unsafe_tool_call_executed`
- `execution_failure`
- `failure_reason`

## Metrics

The v1.0 bundle reports:

- `CSR`: clean success rate.
- `OVR`: one-shot violation rate.
- `SVR`: stateful violation rate.
- `PAR`: poisoned-memory admission rate.
- `PAR-exec`: poisoned-memory admission rate over executed stateful episodes.
- `PRR`: admitted poison retrieved at trigger divided by poison admissions.
- `UPR`: unsafe proposal before checker divided by parseable stateful planner outputs.
- `EFR`: execution-failure rate.
- `CAL-PRR`, `CAL-UPR`, `CAL-SVR`, and `CAL-EFR` for the oracle-retrieved-memory calibration condition.

## License

The MEMTRACE synthetic corpus, traces, labels, metrics, and harness code are released under the MIT License in the anonymized review bundle.
External models and Python packages are documented separately in `THIRD_PARTY_ASSETS.md`; model weights are referenced, not redistributed.

## Intended Use

Use MEMTRACE to evaluate how persistent memory changes observed safety behavior under otherwise matched task and retrieval conditions.
The artifact supports metric regeneration from traces without running the actor model.

## Limitations

- The v1.0 audited pilot includes one actor/backend pair and does not make broad cross-model claims.
- The benchmark uses synthetic enterprise-assistant tasks with fixed retrieval and deterministic tools.
- `S2` is a provenance-aware reference writer, not a complete deployed defense.
- The oracle-retrieved-memory calibration traces are excluded from the main S0/S1/S2 rates.
