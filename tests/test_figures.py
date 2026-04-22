from scripts.make_figures import build_ovr_vs_svr, build_pipeline_figure, build_violation_by_horizon


def test_build_pipeline_figure_returns_svg() -> None:
    svg = build_pipeline_figure()
    assert svg.startswith('<svg')
    assert 'Retriever' in svg


def test_build_ovr_vs_svr_contains_system_labels() -> None:
    metrics = {
        "S0": {"OVR": 0.5, "SVR": 0.0, "stateful_by_horizon": {"1": 0.0, "3": 0.0, "7": 0.0}},
        "S1": {"OVR": 0.5, "SVR": 0.8, "stateful_by_horizon": {"1": 1.0, "3": 1.0, "7": 0.5}},
        "S2": {"OVR": 0.5, "SVR": 0.0, "stateful_by_horizon": {"1": 0.0, "3": 0.0, "7": 0.0}},
    }
    svg = build_ovr_vs_svr(metrics)
    assert 'Figure 2. OVR vs SVR by system' in svg
    assert 'S1' in svg


def test_build_violation_by_horizon_contains_delta_labels() -> None:
    metrics = {
        "S0": {"stateful_by_horizon": {"1": 0.0, "3": 0.0, "7": 0.0}},
        "S1": {"stateful_by_horizon": {"1": 1.0, "3": 1.0, "7": 0.5}},
        "S2": {"stateful_by_horizon": {"1": 0.0, "3": 0.0, "7": 0.0}},
    }
    svg = build_violation_by_horizon(metrics)
    assert 'Δ=1' in svg
    assert 'Δ=7' in svg
