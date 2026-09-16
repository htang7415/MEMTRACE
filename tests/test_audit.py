from memtrace.evaluation.audit import (
    ambiguity_targeted_audit_sample,
    audit_template_records,
    cohens_kappa,
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
        {
            "episode_id": "a-1",
            "actor_model": "model-a",
            "system": "S1",
            "episode_kind": "clean_control",
            "horizon": 1,
            "unsafe": False,
        },
        {
            "episode_id": "a-2",
            "actor_model": "model-a",
            "system": "S1",
            "episode_kind": "clean_control",
            "horizon": 1,
            "unsafe": False,
        },
        {
            "episode_id": "b-1",
            "actor_model": "model-b",
            "system": "S1",
            "episode_kind": "clean_control",
            "horizon": 1,
            "unsafe": False,
        },
    ]
    sample = stratified_audit_sample(scores, sample_size=2)
    assert {item["actor_model"] for item in sample} == {"model-a", "model-b"}


def test_stratified_audit_sample_covers_named_project_categories() -> None:
    scores = [
        {
            "episode_id": "a-clean",
            "actor_model": "model-a",
            "system": "S1",
            "episode_kind": "clean_control",
            "unsafe": False,
        },
        {
            "episode_id": "a-one",
            "actor_model": "model-a",
            "system": "S1",
            "episode_kind": "one_shot_attack",
            "unsafe": True,
        },
        {
            "episode_id": "a-d1",
            "actor_model": "model-a",
            "system": "S1",
            "episode_kind": "stateful_attack",
            "horizon": 1,
            "unsafe": True,
        },
        {
            "episode_id": "a-d3",
            "actor_model": "model-a",
            "system": "S1",
            "episode_kind": "stateful_attack",
            "horizon": 3,
            "unsafe": False,
        },
        {
            "episode_id": "a-d7",
            "actor_model": "model-a",
            "system": "S1",
            "episode_kind": "stateful_attack",
            "horizon": 7,
            "unsafe": False,
        },
        {
            "episode_id": "a-s0",
            "actor_model": "model-a",
            "system": "S0",
            "episode_kind": "stateful_attack",
            "horizon": 7,
            "unsafe": False,
        },
    ]
    sample = stratified_audit_sample(scores, sample_size=6)
    assert [item["episode_id"] for item in sample] == ["a-clean", "a-one", "a-d1", "a-d3", "a-d7", "a-s0"]


def test_ambiguity_targeted_sample_prioritizes_flagged_episodes() -> None:
    scores = [
        {
            "episode_id": "clear-1",
            "actor_model": "model-a",
            "system": "S1",
            "episode_kind": "clean_control",
            "horizon": 1,
            "ambiguity_reason": None,
        },
        {
            "episode_id": "ambiguous-1",
            "actor_model": "model-a",
            "system": "S1",
            "episode_kind": "stateful_attack",
            "horizon": 1,
            "ambiguity_reason": "partial_argument_match",
        },
        {
            "episode_id": "ambiguous-2",
            "actor_model": "model-a",
            "system": "S1",
            "episode_kind": "stateful_attack",
            "horizon": 3,
            "ambiguity_reason": "policy_decision_at_threshold",
        },
    ]
    sample = ambiguity_targeted_audit_sample(scores, sample_size=2)
    assert [item["episode_id"] for item in sample] == ["ambiguous-1", "ambiguous-2"]


def test_ambiguity_targeted_sample_fills_remaining_slots_from_stratified_pool() -> None:
    scores = [
        {
            "episode_id": "ambiguous-1",
            "actor_model": "model-a",
            "system": "S1",
            "episode_kind": "stateful_attack",
            "horizon": 1,
            "ambiguity_reason": "partial_argument_match",
        },
        {
            "episode_id": "clear-1",
            "actor_model": "model-a",
            "system": "S1",
            "episode_kind": "clean_control",
            "horizon": 1,
            "ambiguity_reason": None,
        },
        {
            "episode_id": "clear-2",
            "actor_model": "model-a",
            "system": "S1",
            "episode_kind": "one_shot_attack",
            "horizon": 1,
            "ambiguity_reason": None,
        },
    ]
    sample = ambiguity_targeted_audit_sample(scores, sample_size=2)
    episode_ids = [item["episode_id"] for item in sample]
    assert episode_ids[0] == "ambiguous-1"
    assert episode_ids[1] in {"clear-1", "clear-2"}
    assert len(sample) == 2


def test_cohens_kappa_is_higher_than_raw_agreement_under_class_imbalance() -> None:
    # 9/10 episodes are trivially "safe" on both sides; one genuine disagreement.
    comparisons = [{"rule_label": "safe", "human_label": "safe", "agreement": True} for _ in range(9)]
    comparisons.append({"rule_label": "unsafe", "human_label": "safe", "agreement": False})
    kappa = cohens_kappa(comparisons)
    agreement_rate = 9 / 10
    assert kappa is not None
    assert kappa < agreement_rate


def test_cohens_kappa_is_none_for_empty_or_single_category() -> None:
    assert cohens_kappa([]) is None
    single_category = [{"rule_label": "safe", "human_label": "safe", "agreement": True} for _ in range(5)]
    assert cohens_kappa(single_category) is None


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


def test_labeler_audit_report_includes_ambiguity_reason_and_kappa() -> None:
    scores = [
        {"episode_id": "ep-safe", "unsafe": False, "ambiguity_reason": None},
        {"episode_id": "ep-flagged", "unsafe": True, "ambiguity_reason": "partial_argument_match"},
    ]
    audit = [
        {"episode_id": "ep-safe", "human_label": "safe"},
        {"episode_id": "ep-flagged", "human_label": "unsafe"},
    ]
    report = labeler_audit_report(scores, audit)
    assert report["agreement_rate"] == 1.0
    assert report["cohens_kappa"] == 1.0
    comparisons_by_id = {item["episode_id"]: item for item in report["comparisons"]}
    assert comparisons_by_id["ep-flagged"]["ambiguity_reason"] == "partial_argument_match"
    assert comparisons_by_id["ep-safe"]["ambiguity_reason"] is None


def test_render_audit_report_markdown_includes_pending_count() -> None:
    report = {
        "n": 1,
        "agreements": 1,
        "agreement_rate": 1.0,
        "cohens_kappa": None,
        "comparisons": [
            {
                "episode_id": "ep-safe",
                "rule_label": "safe",
                "human_label": "safe",
                "agreement": True,
                "ambiguity_reason": None,
            }
        ],
    }
    markdown = render_audit_report_markdown(report, pending_count=39)
    assert "- reviewed episodes: 1" in markdown
    assert "- pending episodes: 39" in markdown
    assert "Cohen's kappa: undefined" in markdown
