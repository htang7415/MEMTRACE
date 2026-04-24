from memtrace.eval.metrics import aggregate_metrics, rate


def test_rate_zero_denominator() -> None:
    assert rate(1, 0) == 0.0


def test_aggregate_metrics_includes_pilot_validation_fields() -> None:
    episode_scores = [
        {
            "actor_model": "mlx-community/Qwen2.5-3B-Instruct-4bit",
            "system": "S0",
            "episode_kind": "stateful_attack",
            "payload_type": "direct_override",
            "safe": True,
            "unsafe": False,
            "poison_admission_flag": False,
            "horizon": 1,
            "tool_required": True,
            "planner_structured_output": True,
            "planner_explicit_null": False,
            "planner_malformed_output": False,
            "tool_call_present": True,
            "writer_turn_count": 1,
            "writer_structured_turns": 1,
            "writer_malformed_turns": 0,
            "writer_candidates_presented": 1,
            "writer_valid_memory_type_candidates": 1,
            "admitted_candidates": 1,
            "clean_candidates_presented": 0,
            "admitted_clean_candidates": 0,
        },
        {
            "actor_model": "mlx-community/Qwen2.5-3B-Instruct-4bit",
            "system": "S1",
            "episode_kind": "stateful_attack",
            "payload_type": "direct_override",
            "safe": True,
            "unsafe": False,
            "poison_admission_flag": False,
            "horizon": 1,
            "tool_required": True,
            "planner_structured_output": True,
            "planner_explicit_null": False,
            "planner_malformed_output": False,
            "tool_call_present": True,
            "writer_turn_count": 1,
            "writer_structured_turns": 1,
            "writer_malformed_turns": 0,
            "writer_candidates_presented": 1,
            "writer_valid_memory_type_candidates": 1,
            "admitted_candidates": 1,
            "clean_candidates_presented": 0,
            "admitted_clean_candidates": 0,
        },
        {
            "actor_model": "mlx-community/Qwen2.5-3B-Instruct-4bit",
            "system": "S2",
            "episode_kind": "stateful_attack",
            "payload_type": "direct_override",
            "safe": True,
            "unsafe": False,
            "poison_admission_flag": False,
            "horizon": 1,
            "tool_required": True,
            "planner_structured_output": True,
            "planner_explicit_null": False,
            "planner_malformed_output": False,
            "tool_call_present": True,
            "writer_turn_count": 1,
            "writer_structured_turns": 1,
            "writer_malformed_turns": 0,
            "writer_candidates_presented": 1,
            "writer_valid_memory_type_candidates": 1,
            "admitted_candidates": 1,
            "clean_candidates_presented": 0,
            "admitted_clean_candidates": 0,
        },
    ]
    metrics = aggregate_metrics(episode_scores)
    s1 = metrics["by_configuration"]["mlx-community/Qwen2.5-3B-Instruct-4bit"]["S1"]
    assert s1["required_tool_call_rate"] == 1.0
    assert s1["writer_structured_turn_rate"] == 1.0
    assert s1["writer_valid_memory_type_rate"] == 1.0
    assert metrics["pilot_validation_by_configuration"]["mlx-community/Qwen2.5-3B-Instruct-4bit"]["S1"]["official_pilot_valid"] is True
