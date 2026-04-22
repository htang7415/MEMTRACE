from scripts.make_tables import build_supplementary_tables, build_table1


def test_build_table1_renders_system_rows() -> None:
    metrics = {
        "by_configuration": {
            "mlx-community/Qwen2.5-3B-Instruct-4bit": {
                "S0": {"CSR": 1.0, "OVR": 0.5, "SVR": 0.5, "SRG": 0.0, "PAR": 0.0, "WR": 0.0, "HDR": 0.0, "CWRR": 1.0, "phi": None, "phi_ci95": None, "phi_status": "undefined_par_zero"},
                "S1": {"CSR": 1.0, "OVR": 0.5, "SVR": 0.8, "SRG": 0.3, "PAR": 1.0, "WR": 1.0, "HDR": 0.1, "CWRR": 0.0, "phi": 1.0, "phi_ci95": [1.0, 1.0], "phi_status": "ok"},
                "S2": {"CSR": 1.0, "OVR": 0.5, "SVR": 0.5, "SRG": 0.0, "PAR": 0.0, "WR": 0.0, "HDR": 0.0, "CWRR": 0.08, "phi": None, "phi_ci95": None, "phi_status": "undefined_par_zero"},
            }
        },
        "system_average": {
            "S0": {"CSR": 1.0, "OVR": 0.5, "SVR": 0.5, "SRG": 0.0, "PAR": 0.0, "WR": 0.0, "HDR": 0.0, "CWRR": 1.0},
            "S1": {"CSR": 1.0, "OVR": 0.5, "SVR": 0.8, "SRG": 0.3, "PAR": 1.0, "WR": 1.0, "HDR": 0.1, "CWRR": 0.0},
            "S2": {"CSR": 1.0, "OVR": 0.5, "SVR": 0.5, "SRG": 0.0, "PAR": 0.0, "WR": 0.0, "HDR": 0.0, "CWRR": 0.08},
        },
        "ranking_reversal_by_actor_model": {"mlx-community/Qwen2.5-3B-Instruct-4bit": False},
    }
    table = build_table1(metrics)
    assert "| Qwen2.5-3B | S0 | 1.000 | 0.500 | 0.500 | 0.000 | 0.000 | N/A |" in table
    assert "| Qwen2.5-3B | S1 | 1.000 | 0.500 | 0.800 | 0.300 | 1.000 | False |" in table


def test_build_supplementary_tables_groups_by_task() -> None:
    episode_scores = [
        {"episode_id": "qwen25_3b:S1:task-a:stateful:d1:direct_override", "actor_model": "mlx-community/Qwen2.5-3B-Instruct-4bit", "task_id": "task-a", "system": "S1", "episode_kind": "stateful_attack", "unsafe": True, "poison_admission_flag": True},
        {"episode_id": "qwen25_3b:S1:task-a:stateful:d3:direct_override", "actor_model": "mlx-community/Qwen2.5-3B-Instruct-4bit", "task_id": "task-a", "system": "S1", "episode_kind": "stateful_attack", "unsafe": False, "poison_admission_flag": True},
    ]
    tables = build_supplementary_tables(episode_scores, None)
    assert "| task-a | 1.000 | 0.500 | 2 |" in tables
