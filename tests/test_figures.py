from scripts.make_figures import build_causal_chain_by_system, build_ovr_vs_svr, build_pipeline_figure, build_task_localization, build_violation_by_horizon


def _metric_row(ovr: float, svr: float, *, admitted: int = 0, retrieved: int = 0) -> dict:
    return {
        "OVR": ovr,
        "SVR": svr,
        "rate_counts": {
            "one_shot_unsafe": [int(round(ovr * 24)), 24],
            "stateful_unsafe": [int(round(svr * 72)), 72],
        },
        "causal_chain_counts": {
            "stateful_attacks": 72,
            "poison_admitted": admitted,
            "admitted_poison_retrieved_at_trigger": retrieved,
            "unsafe_proposal_before_checker": 0,
            "unsafe_executed": 0,
            "execution_failure": 6,
        },
    }


def test_build_pipeline_figure_returns_svg() -> None:
    svg = build_pipeline_figure()
    assert svg.startswith('<svg')
    assert 'Retriever' in svg
    assert 'Figure 1.' not in svg


def test_build_ovr_vs_svr_contains_system_labels() -> None:
    metrics = {
        "by_configuration": {
            "mlx-community/Qwen2.5-3B-Instruct-4bit": {
                "S0": _metric_row(0.25, 0.0),
                "S1": _metric_row(0.30, 0.0, admitted=6, retrieved=6),
                "S2": _metric_row(0.25, 0.0),
            }
        },
    }
    svg = build_ovr_vs_svr(metrics)
    assert 'Figure 2.' not in svg
    assert 'S1' in svg
    assert '7/24' in svg


def test_build_causal_chain_by_system_contains_mechanism_counts() -> None:
    metrics = {
        "by_configuration": {
            "mlx-community/Qwen2.5-3B-Instruct-4bit": {
                "S0": _metric_row(0.5, 0.0),
                "S1": _metric_row(0.5, 0.0, admitted=6, retrieved=6),
                "S2": _metric_row(0.5, 0.0),
            }
        }
    }
    svg = build_causal_chain_by_system(metrics)
    assert 'Where the S1 delayed chain stops' in svg
    assert 'S1' in svg
    assert '6/72' in svg


def test_build_task_localization_contains_active_tasks() -> None:
    scores = []
    for task_id, admitted in (("budget-limit-rule", 3), ("calendar-attendee", 3), ("meeting-time", 0)):
        for index in range(6):
            scores.append(
                {
                    "system": "S1",
                    "episode_kind": "stateful_attack",
                    "task_id": task_id,
                    "poison_admission_flag": index < admitted,
                }
            )
    svg = build_task_localization(scores)
    assert 'budget limit' in svg
    assert 'calendar attendee' in svg
    assert '3/6' in svg


def test_build_violation_by_horizon_contains_delta_labels() -> None:
    scores = []
    for horizon in (1, 3, 7):
        scores.extend(
            [
                {
                    "system": "S1",
                    "episode_kind": "stateful_attack",
                    "horizon": horizon,
                    "poison_admission_flag": True,
                    "admitted_memory_retrieved_at_trigger": True,
                    "unsafe_tool_call_proposed_before_checker": False,
                    "unsafe_tool_call_executed": False,
                    "execution_failure": False,
                },
                {
                    "system": "S1",
                    "episode_kind": "stateful_attack",
                    "horizon": horizon,
                    "poison_admission_flag": False,
                    "admitted_memory_retrieved_at_trigger": False,
                    "unsafe_tool_call_proposed_before_checker": False,
                    "unsafe_tool_call_executed": False,
                    "execution_failure": True,
                },
            ]
        )
    svg = build_violation_by_horizon(scores)
    assert 'Figure 5.' not in svg
    assert 'Δ=1' in svg
    assert 'Δ=7' in svg
    assert 'Poison admitted' in svg
    assert '1/2' in svg
