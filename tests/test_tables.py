from scripts.make_tables import build_supplementary_tables, build_table1


def test_build_table1_renders_system_rows() -> None:
    metrics = {
        "by_configuration": {
            "mlx-community/Qwen2.5-3B-Instruct-4bit": {
                "S0": {"CSR": 1.0, "OVR": 0.5, "OVR_ci95": [0.2, 0.8], "SVR": 0.5, "SVR_ci95": [0.2, 0.8], "PAR": 0.0, "PAR_ci95": [0.0, 0.1], "PAR_executed_only": 0.0, "PAR_executed_only_ci95": [0.0, 0.1], "WR": 0.0, "WR_ci95": [0.0, 0.1], "HDR": 0.0, "CWRR": 1.0, "phi": None, "phi_ci95": None, "phi_status": "undefined_par_zero", "planner_structured_output_rate": 1.0, "planner_explicit_null_rate": 0.0, "planner_malformed_output_rate": 0.0, "execution_failure_rate": 0.0, "required_tool_call_rate": 1.0, "writer_structured_turn_rate": 1.0, "writer_malformed_turn_rate": 0.0, "writer_valid_memory_type_rate": 1.0},
                "S1": {"CSR": 1.0, "OVR": 0.5, "OVR_ci95": [0.2, 0.8], "SVR": 0.8, "SVR_ci95": [0.5, 0.95], "PAR": 1.0, "PAR_ci95": [0.9, 1.0], "PAR_executed_only": 0.75, "PAR_executed_only_ci95": [0.4, 0.95], "WR": 1.0, "WR_ci95": [0.9, 1.0], "HDR": 0.1, "CWRR": 0.0, "phi": 1.0, "phi_ci95": [1.0, 1.0], "phi_status": "ok", "planner_structured_output_rate": 1.0, "planner_explicit_null_rate": 0.0, "planner_malformed_output_rate": 0.0, "execution_failure_rate": 0.2, "required_tool_call_rate": 1.0, "writer_structured_turn_rate": 1.0, "writer_malformed_turn_rate": 0.0, "writer_valid_memory_type_rate": 1.0},
                "S2": {"CSR": 1.0, "OVR": 0.5, "OVR_ci95": [0.2, 0.8], "SVR": 0.5, "SVR_ci95": [0.2, 0.8], "PAR": 0.0, "PAR_ci95": [0.0, 0.1], "PAR_executed_only": 0.0, "PAR_executed_only_ci95": [0.0, 0.1], "WR": 0.0, "WR_ci95": [0.0, 0.1], "HDR": 0.0, "CWRR": 0.08, "phi": None, "phi_ci95": None, "phi_status": "undefined_par_zero", "planner_structured_output_rate": 1.0, "planner_explicit_null_rate": 0.0, "planner_malformed_output_rate": 0.0, "execution_failure_rate": 0.0, "required_tool_call_rate": 1.0, "writer_structured_turn_rate": 1.0, "writer_malformed_turn_rate": 0.0, "writer_valid_memory_type_rate": 1.0},
            }
        },
        "system_average": {
            "S0": {"CSR": 1.0, "OVR": 0.5, "OVR_ci95": [0.2, 0.8], "SVR": 0.5, "SVR_ci95": [0.2, 0.8], "PAR": 0.0, "PAR_ci95": [0.0, 0.1], "PAR_executed_only": 0.0, "PAR_executed_only_ci95": [0.0, 0.1], "WR": 0.0, "WR_ci95": [0.0, 0.1], "HDR": 0.0, "CWRR": 1.0, "planner_structured_output_rate": 1.0, "planner_explicit_null_rate": 0.0, "planner_malformed_output_rate": 0.0, "execution_failure_rate": 0.0, "required_tool_call_rate": 1.0, "writer_structured_turn_rate": 1.0, "writer_malformed_turn_rate": 0.0, "writer_valid_memory_type_rate": 1.0},
            "S1": {"CSR": 1.0, "OVR": 0.5, "OVR_ci95": [0.2, 0.8], "SVR": 0.8, "SVR_ci95": [0.5, 0.95], "PAR": 1.0, "PAR_ci95": [0.9, 1.0], "PAR_executed_only": 0.75, "PAR_executed_only_ci95": [0.4, 0.95], "WR": 1.0, "WR_ci95": [0.9, 1.0], "HDR": 0.1, "CWRR": 0.0, "planner_structured_output_rate": 1.0, "planner_explicit_null_rate": 0.0, "planner_malformed_output_rate": 0.0, "execution_failure_rate": 0.2, "required_tool_call_rate": 1.0, "writer_structured_turn_rate": 1.0, "writer_malformed_turn_rate": 0.0, "writer_valid_memory_type_rate": 1.0},
            "S2": {"CSR": 1.0, "OVR": 0.5, "OVR_ci95": [0.2, 0.8], "SVR": 0.5, "SVR_ci95": [0.2, 0.8], "PAR": 0.0, "PAR_ci95": [0.0, 0.1], "PAR_executed_only": 0.0, "PAR_executed_only_ci95": [0.0, 0.1], "WR": 0.0, "WR_ci95": [0.0, 0.1], "HDR": 0.0, "CWRR": 0.08, "planner_structured_output_rate": 1.0, "planner_explicit_null_rate": 0.0, "planner_malformed_output_rate": 0.0, "execution_failure_rate": 0.0, "required_tool_call_rate": 1.0, "writer_structured_turn_rate": 1.0, "writer_malformed_turn_rate": 0.0, "writer_valid_memory_type_rate": 1.0},
        },
        "ranking_reversal_by_actor_model": {"mlx-community/Qwen2.5-3B-Instruct-4bit": False},
        "pilot_validation_by_configuration": {
            "mlx-community/Qwen2.5-3B-Instruct-4bit": {
                "S0": {"provenance_writer_mechanism_check_pass": True, "official_pilot_valid": True},
                "S1": {"provenance_writer_mechanism_check_pass": True, "official_pilot_valid": True},
                "S2": {"provenance_writer_mechanism_check_pass": True, "official_pilot_valid": True},
            }
        },
    }
    table = build_table1(metrics)
    assert "| Qwen2.5-3B | S0 | 1.000 | 0.500 | [0.200, 0.800] | 0.500 | [0.200, 0.800] | 0.000 | [0.000, 0.100] | 0.000 | [0.000, 0.100] | N/A |" in table
    assert "| Qwen2.5-3B | S1 | 1.000 | 0.500 | [0.200, 0.800] | 0.800 | [0.500, 0.950] | 1.000 | [0.900, 1.000] | 0.750 | [0.400, 0.950] | False |" in table
    assert "| Qwen2.5-3B | S1 | 1.000 | [0.900, 1.000] | 0.100 | N/A | 1.000 | [1.000, 1.000] | ok |" in table
    assert "| Qwen2.5-3B | S1 | 1.000 | 1.000 | 0.200 | 1.000 | 1.000 | True | True |" in table


def test_build_supplementary_tables_groups_by_task() -> None:
    episode_scores = [
        {"episode_id": "qwen25_3b:S1:task-a:stateful:d1:direct_override", "actor_model": "mlx-community/Qwen2.5-3B-Instruct-4bit", "task_id": "task-a", "system": "S1", "episode_kind": "stateful_attack", "unsafe": True, "poison_admission_flag": True, "execution_failure": False},
        {"episode_id": "qwen25_3b:S1:task-a:stateful:d3:direct_override", "actor_model": "mlx-community/Qwen2.5-3B-Instruct-4bit", "task_id": "task-a", "system": "S1", "episode_kind": "stateful_attack", "unsafe": False, "poison_admission_flag": True, "execution_failure": True},
    ]
    tables = build_supplementary_tables(episode_scores, None)
    assert "| task-a | 1.000 | 1.000 | 0.500 | 2 | 1 |" in tables


def test_build_supplementary_tables_renders_na_for_zero_executed_par_exec_and_horizon_ef() -> None:
    episode_scores = [
        {"episode_id": "llama:S1:task-a:stateful:d1:direct_override", "actor_model": "mlx-community/Llama-3.2-3B-Instruct-4bit", "task_id": "task-a", "system": "S1", "episode_kind": "stateful_attack", "unsafe": False, "poison_admission_flag": True, "execution_failure": True},
    ]
    metrics = {
        "by_configuration": {
            "mlx-community/Llama-3.2-3B-Instruct-4bit": {
                "S1": {"stateful_execution_failure_by_horizon": {"1": 1.0, "3": 0.5, "7": 0.25}},
            }
        }
    }
    tables = build_supplementary_tables(episode_scores, metrics)
    assert "| task-a | 1.000 | N/A | 0.000 | 1 | 0 |" in tables
    assert "### Llama-3.2-3B S1 Stateful Execution Failure by Horizon" in tables
    assert "| Δ=1 | 1.000 |" in tables
