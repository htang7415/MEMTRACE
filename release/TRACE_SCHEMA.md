# MEMTRACE Trace Schema

Every released JSONL trace row includes the fields below. Missing fields are schema errors in `scripts/validate_release.py`.

- `trace_id`
- `protocol_version`
- `actor_model`
- `actor_revision`
- `tokenizer_id`
- `tokenizer_revision`
- `writer_backend`
- `planner_backend`
- `system_id`
- `episode_kind`
- `task_id`
- `payload_id`
- `payload_type`
- `horizon`
- `retrieved_passage_ids_initial`
- `poison_retrieved_initial`
- `writer_candidates`
- `accepted_memory_ids`
- `admitted_poison`
- `retrieved_memory_ids_trigger`
- `admitted_poison_retrieved_at_trigger`
- `unsafe_proposal_before_checker`
- `unsafe_blocked_by_checker`
- `unsafe_executed`
- `execution_failure`
- `violation_attribution`
- `final_tool_call`
- `policy_checker_verdict`
- `scorer_version`
- `scorer_hash`
- `trace_created_at`

Clean episodes use `payload_id = null`, `payload_type = null`, and `horizon = null`.
One-shot episodes use `horizon = null`.
Stateful episodes use `horizon` in `{1, 3, 7}`.
