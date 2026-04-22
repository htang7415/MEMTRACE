from memtrace.eval.audit import labeler_audit_report, stratified_audit_sample, stratified_sample_size


def test_stratified_sample_size() -> None:
    assert stratified_sample_size() == 40


def test_stratified_audit_sample_is_deterministic_and_bounded() -> None:
    scores = [
        {
            "episode_id": f"ep-{index}",
            "actor_model": "model-a",
            "system": "S1",
            "episode_kind": "stateful_attack",
            "horizon": 1,
            "unsafe": bool(index % 2),
        }
        for index in range(5)
    ]
    sample = stratified_audit_sample(scores, sample_size=3)
    assert [item["episode_id"] for item in sample] == ["ep-0", "ep-1", "ep-2"]


def test_stratified_audit_sample_covers_actor_models() -> None:
    scores = [
        {"episode_id": "a-1", "actor_model": "model-a", "system": "S1", "episode_kind": "clean_control", "horizon": 1, "unsafe": False},
        {"episode_id": "a-2", "actor_model": "model-a", "system": "S1", "episode_kind": "clean_control", "horizon": 1, "unsafe": False},
        {"episode_id": "b-1", "actor_model": "model-b", "system": "S1", "episode_kind": "clean_control", "horizon": 1, "unsafe": False},
    ]
    sample = stratified_audit_sample(scores, sample_size=2)
    assert {item["actor_model"] for item in sample} == {"model-a", "model-b"}


def test_labeler_audit_report_computes_agreement() -> None:
    scores = [
        {"episode_id": "ep-safe", "unsafe": False},
        {"episode_id": "ep-unsafe", "unsafe": True},
    ]
    audit = [
        {"episode_id": "ep-safe", "human_label": "safe"},
        {"episode_id": "ep-unsafe", "human_label": "safe"},
    ]
    report = labeler_audit_report(scores, audit)
    assert report["agreement_rate"] == 0.5
