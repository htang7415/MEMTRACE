from memtrace.eval.audit import (
    audit_template_records,
    labeler_audit_report,
    render_audit_report_markdown,
    reviewed_audit_records,
    stratified_audit_sample,
    stratified_sample_size,
)


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


def test_audit_template_is_blind_and_pending() -> None:
    sample = [
        {
            "episode_id": "ep-fail",
            "actor_model": "model-a",
            "system": "S1",
            "task_id": "task-a",
            "episode_kind": "stateful_attack",
            "payload_type": "direct_override",
            "horizon": 1,
            "trace_path": "traces/ep-fail.jsonl",
            "execution_failure": True,
        }
    ]
    template = audit_template_records(sample)
    assert "rule_label" not in template[0]
    assert template[0]["human_label"] is None
    assert reviewed_audit_records(template) == []


def test_render_audit_report_markdown_includes_pending_count() -> None:
    report = {
        "n": 1,
        "agreements": 1,
        "agreement_rate": 1.0,
        "comparisons": [
            {
                "episode_id": "ep-safe",
                "rule_label": "safe",
                "human_label": "safe",
                "agreement": True,
            }
        ],
    }
    markdown = render_audit_report_markdown(report, pending_count=39)
    assert "- reviewed episodes: 1" in markdown
    assert "- pending episodes: 39" in markdown
