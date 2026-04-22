from scripts.make_tables import build_supplementary_tables, build_table1


def test_build_table1_renders_system_rows() -> None:
    metrics = {
        "S0": {"CSR": 1.0, "OVR": 0.5, "SVR": 0.0, "SRG": -0.5, "PAR": 0.0, "WR": 0.0, "HDR": 0.0, "CWRR": 1.0, "stateful_by_horizon": {"1": 0.0, "3": 0.0, "7": 0.0}},
        "S1": {"CSR": 1.0, "OVR": 0.5, "SVR": 0.8, "SRG": 0.3, "PAR": 1.0, "WR": 1.0, "HDR": 0.1, "CWRR": 0.0, "stateful_by_horizon": {"1": 1.0, "3": 1.0, "7": 0.5}},
        "S2": {"CSR": 1.0, "OVR": 0.5, "SVR": 0.0, "SRG": -0.5, "PAR": 0.0, "WR": 0.0, "HDR": 0.0, "CWRR": 0.08, "stateful_by_horizon": {"1": 0.0, "3": 0.0, "7": 0.0}},
    }
    table = build_table1(metrics)
    assert "| S0 | 1.000 | 0.500 | 0.000 | -0.500 | 0.000 | 0.000 | 0.000 | 1.000 |" in table
    assert "| S1 | 1.000 | 0.500 | 0.800 | 0.300 | 1.000 | 1.000 | 0.100 | 0.000 |" in table


def test_build_supplementary_tables_groups_by_task() -> None:
    episode_scores = [
        {"episode_id": "S1:task-a:stateful:d1:direct_override", "system": "S1", "episode_kind": "stateful_attack", "unsafe": True, "poison_admission_flag": True},
        {"episode_id": "S1:task-a:stateful:d3:direct_override", "system": "S1", "episode_kind": "stateful_attack", "unsafe": False, "poison_admission_flag": True},
    ]
    tables = build_supplementary_tables(episode_scores)
    assert "| task-a | 1.000 | 0.500 | 2 |" in tables
