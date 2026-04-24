from scripts.validate_run import build_validation_report, overall_pilot_valid


def test_build_validation_report_renders_rows() -> None:
    metrics = {
        "by_configuration": {
            "mlx-community/Qwen2.5-3B-Instruct-4bit": {
                "S0": {"planner_structured_output_rate": 1.0, "required_tool_call_rate": 1.0, "writer_structured_turn_rate": 1.0, "writer_valid_memory_type_rate": 1.0},
                "S1": {"planner_structured_output_rate": 0.9, "required_tool_call_rate": 0.95, "writer_structured_turn_rate": 0.93, "writer_valid_memory_type_rate": 0.91},
                "S2": {"planner_structured_output_rate": 0.9, "required_tool_call_rate": 0.95, "writer_structured_turn_rate": 0.93, "writer_valid_memory_type_rate": 0.91},
            }
        },
        "pilot_validation_by_configuration": {
            "mlx-community/Qwen2.5-3B-Instruct-4bit": {
                "S0": {"s0_stateful_sanity_pass": True, "official_pilot_valid": True},
                "S1": {"s0_stateful_sanity_pass": True, "official_pilot_valid": True},
                "S2": {"s0_stateful_sanity_pass": True, "official_pilot_valid": True},
            }
        },
    }
    report = build_validation_report(metrics)
    assert "| Qwen2.5-3B | S1 | 0.900 | 0.950 | 0.930 | 0.910 | True | True |" in report


def test_overall_pilot_valid_requires_all_rows_to_pass() -> None:
    metrics = {
        "pilot_validation_by_configuration": {
            "mlx-community/Qwen2.5-3B-Instruct-4bit": {
                "S0": {"official_pilot_valid": True},
                "S1": {"official_pilot_valid": False},
                "S2": {"official_pilot_valid": True},
            }
        }
    }
    assert overall_pilot_valid(metrics) is False
