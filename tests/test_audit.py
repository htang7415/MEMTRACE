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


def test_stratified_audit_sample_covers_named_project_categories() -> None:
    scores = [
        {"episode_id": "a-clean", "actor_model": "model-a", "system": "S1", "episode_kind": "clean_control", "unsafe": False},
        {"episode_id": "a-one", "actor_model": "model-a", "system": "S1", "episode_kind": "one_shot_attack", "unsafe": True},
        {"episode_id": "a-d1", "actor_model": "model-a", "system": "S1", "episode_kind": "stateful_attack", "horizon": 1, "unsafe": True},
        {"episode_id": "a-d3", "actor_model": "model-a", "system": "S1", "episode_kind": "stateful_attack", "horizon": 3, "unsafe": False},
        {"episode_id": "a-d7", "actor_model": "model-a", "system": "S1", "episode_kind": "stateful_attack", "horizon": 7, "unsafe": False},
        {"episode_id": "a-s0", "actor_model": "model-a", "system": "S0", "episode_kind": "stateful_attack", "horizon": 7, "unsafe": False},
    ]
    sample = stratified_audit_sample(scores, sample_size=6)
    assert [item["episode_id"] for item in sample] == ["a-clean", "a-one", "a-d1", "a-d3", "a-d7", "a-s0"]


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
