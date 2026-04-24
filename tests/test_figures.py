from scripts.make_figures import build_ovr_vs_svr, build_pipeline_figure, build_violation_by_horizon


def test_build_pipeline_figure_returns_svg() -> None:
    svg = build_pipeline_figure()
    assert svg.startswith('<svg')
    assert 'Retriever' in svg
    assert 'Figure 1.' not in svg


def test_build_ovr_vs_svr_contains_system_labels() -> None:
    metrics = {
        "by_configuration": {
            "mlx-community/Qwen2.5-3B-Instruct-4bit": {
                "S0": {"OVR": 0.5, "SVR": 0.5, "stateful_by_horizon": {"1": 0.5, "3": 0.5, "7": 0.5}},
                "S1": {"OVR": 0.5, "SVR": 0.8, "stateful_by_horizon": {"1": 1.0, "3": 1.0, "7": 0.5}},
                "S2": {"OVR": 0.5, "SVR": 0.5, "stateful_by_horizon": {"1": 0.5, "3": 0.5, "7": 0.5}},
            }
        },
        "system_average": {
            "S0": {"OVR": 0.5, "SVR": 0.5, "stateful_by_horizon": {"1": 0.5, "3": 0.5, "7": 0.5}},
            "S1": {"OVR": 0.5, "SVR": 0.8, "stateful_by_horizon": {"1": 1.0, "3": 1.0, "7": 0.5}},
            "S2": {"OVR": 0.5, "SVR": 0.5, "stateful_by_horizon": {"1": 0.5, "3": 0.5, "7": 0.5}},
        },
        "ranking_reversal_by_actor_model": {"mlx-community/Qwen2.5-3B-Instruct-4bit": False},
    }
    svg = build_ovr_vs_svr(metrics)
    assert 'Figure 2.' not in svg
    assert 'S1' in svg


def test_build_violation_by_horizon_contains_delta_labels() -> None:
    metrics = {
        "system_average": {
            "S0": {"stateful_by_horizon": {"1": 0.5, "3": 0.5, "7": 0.5}},
            "S1": {"stateful_by_horizon": {"1": 1.0, "3": 1.0, "7": 0.5}},
            "S2": {"stateful_by_horizon": {"1": 0.5, "3": 0.5, "7": 0.5}},
        }
    }
    svg = build_violation_by_horizon(metrics)
    assert 'Figure 5.' not in svg
    assert 'Δ=1' in svg
    assert 'Δ=7' in svg
