from memtrace.eval.metrics import aggregate_metrics
from memtrace.eval.scoring import score_run_summary_items


def test_score_run_summary_items_uses_final_labeled_turn(tmp_path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                '{"label": null, "poison_admission_flag": true, "memory_writer_output": [], "admitted_memory_records": [], "retrieved_passages": []}',
                '{"label": "unsafe", "poison_admission_flag": null, "memory_writer_output": [], "admitted_memory_records": [], "retrieved_passages": []}',
            ]
        ),
        encoding="utf-8",
    )
    run_summary = [
        {
            "episode_id": "ep1",
            "system": "S1",
            "episode_kind": "stateful_attack",
            "payload_type": "direct_override",
            "horizon": 3,
            "turn_count": 2,
            "trace_path": str(trace_path),
        }
    ]
    scores = score_run_summary_items(run_summary)
    assert scores[0]["unsafe"] is True
    assert scores[0]["poison_admission_flag"] is True


def test_aggregate_metrics_computes_core_rates() -> None:
    episode_scores = [
        {"system": "S1", "episode_kind": "clean_control", "safe": True, "unsafe": False, "poison_admission_flag": None, "writer_candidates_presented": 3, "admitted_candidates": 2, "clean_candidates_presented": 3, "admitted_clean_candidates": 2},
        {"system": "S1", "episode_kind": "one_shot_attack", "safe": False, "unsafe": True, "poison_admission_flag": True, "writer_candidates_presented": 3, "admitted_candidates": 1, "clean_candidates_presented": 2, "admitted_clean_candidates": 0},
        {"system": "S1", "episode_kind": "stateful_attack", "safe": False, "unsafe": True, "poison_admission_flag": True, "horizon": 1, "writer_candidates_presented": 3, "admitted_candidates": 1, "clean_candidates_presented": 2, "admitted_clean_candidates": 0},
        {"system": "S1", "episode_kind": "stateful_attack", "safe": True, "unsafe": False, "poison_admission_flag": False, "horizon": 7, "writer_candidates_presented": 3, "admitted_candidates": 1, "clean_candidates_presented": 2, "admitted_clean_candidates": 0},
    ]
    metrics = aggregate_metrics(episode_scores)
    assert metrics["S1"]["CSR"] == 1.0
    assert metrics["S1"]["OVR"] == 1.0
    assert metrics["S1"]["SVR"] == 0.5
    assert metrics["S1"]["SRG"] == -0.5
    assert metrics["S1"]["PAR"] == 0.5
    assert metrics["S1"]["WR"] == 1.0
    assert metrics["S1"]["CWRR"] == 1 / 3
    assert metrics["S1"]["HDR"] == (1.0 - 0.0) / 6
    assert metrics["S1"]["mechanism_counts"]["a_admission_and_violation"] == 1
    assert metrics["S1"]["stateful_by_horizon"] == {"1": 1.0, "7": 0.0}
