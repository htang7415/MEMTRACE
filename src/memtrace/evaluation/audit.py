"""Audit helpers for validating rule-based labels against human review."""

from __future__ import annotations

from collections import Counter, defaultdict


def stratified_sample_size() -> int:
    return 40


def stratified_audit_sample(episode_scores: list[dict], sample_size: int = 40) -> list[dict]:
    """Select a deterministic sample across the project.md audit categories."""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for item in sorted(episode_scores, key=lambda row: row["episode_id"]):
        grouped[_audit_stratum(item)].append(item)

    sample: list[dict] = []
    actor_models = sorted({stratum[0] for stratum in grouped})
    category_order = ("clean", "one_shot", "delta_1", "delta_3", "delta_7", "s0_sanity")
    strata = [(actor_model, category) for actor_model in actor_models for category in category_order]
    while len(sample) < sample_size and any(grouped.values()):
        for stratum in strata:
            if grouped[stratum]:
                sample.append(grouped[stratum].pop(0))
                if len(sample) == sample_size:
                    break
    return sample


def ambiguity_targeted_audit_sample(episode_scores: list[dict], sample_size: int = 40) -> list[dict]:
    """Sample the audit packet so it actually exercises the rule labeler's judgment calls.

    Category-stratified sampling alone can reach `sample_size` on trivially obvious
    episodes and report high agreement without ever reviewing a genuinely contestable
    case. This samples every episode the labeler itself flagged as a judgment call
    (`ambiguity_reason_for_labeling`, e.g. a partial argument match or a policy value
    sitting on its own threshold) first, deterministically ordered by episode_id, then
    fills any remaining slots via the existing category-stratified sample over the
    remaining (unambiguous) episodes for category coverage.
    """
    ambiguous = sorted(
        (item for item in episode_scores if item.get("ambiguity_reason")),
        key=lambda item: item["episode_id"],
    )[:sample_size]
    if len(ambiguous) >= sample_size:
        return ambiguous
    ambiguous_ids = {item["episode_id"] for item in ambiguous}
    remaining_pool = [item for item in episode_scores if item["episode_id"] not in ambiguous_ids]
    fill = stratified_audit_sample(remaining_pool, sample_size=sample_size - len(ambiguous))
    return ambiguous + fill


def labeler_audit_report(episode_scores: list[dict], audit_records: list[dict]) -> dict:
    """Compare human audit labels with rule-based labels for sampled episodes."""
    score_by_id = {item["episode_id"]: item for item in episode_scores}
    comparisons = []
    for record in audit_records:
        episode_id = record["episode_id"]
        score = score_by_id[episode_id]
        human_label = record["human_label"]
        rule_label = _rule_label(score)
        comparisons.append(
            {
                "episode_id": episode_id,
                "rule_label": rule_label,
                "human_label": human_label,
                "agreement": rule_label == human_label,
                "ambiguity_reason": score.get("ambiguity_reason"),
            }
        )

    agreements = sum(1 for item in comparisons if item["agreement"])
    return {
        "n": len(comparisons),
        "agreements": agreements,
        "agreement_rate": agreements / len(comparisons) if comparisons else 0.0,
        "cohens_kappa": cohens_kappa(comparisons),
        "comparisons": comparisons,
    }


def cohens_kappa(comparisons: list[dict]) -> float | None:
    """Chance-corrected agreement between rule and human labels.

    Plain agreement rate can look strong purely from class imbalance (most episodes are
    "safe"), so it is not evidence of real label reliability on its own. Returns `None`
    when there are no comparisons or when expected agreement is already 1.0 (a single
    label category on both sides).
    """
    n = len(comparisons)
    if n == 0:
        return None
    rule_counts = Counter(item["rule_label"] for item in comparisons)
    human_counts = Counter(item["human_label"] for item in comparisons)
    categories = set(rule_counts) | set(human_counts)
    observed_agreement = sum(1 for item in comparisons if item["agreement"]) / n
    expected_agreement = sum((rule_counts.get(c, 0) / n) * (human_counts.get(c, 0) / n) for c in categories)
    if expected_agreement >= 1.0:
        return None
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


def audit_template_records(sample: list[dict]) -> list[dict]:
    """Build the review packet humans fill before computing agreement."""
    return [
        {
            "episode_id": item["episode_id"],
            "actor_model": item.get("actor_model"),
            "system": item.get("system"),
            "task_id": item.get("task_id"),
            "episode_kind": item.get("episode_kind"),
            "payload_type": item.get("payload_type"),
            "horizon": item.get("horizon"),
            "trace_path": item.get("trace_path"),
            "ambiguity_reason": item.get("ambiguity_reason"),
            "human_label": None,
            "notes": "",
        }
        for item in sample
    ]


def reviewed_audit_records(records: list[dict]) -> list[dict]:
    """Return only records with completed human labels."""
    valid_labels = {"safe", "unsafe", "execution_failure"}
    return [record for record in records if record.get("human_label") in valid_labels]


def render_audit_report_markdown(report: dict, pending_count: int = 0) -> str:
    kappa = report.get("cohens_kappa")
    ambiguous_reviewed = sum(1 for item in report["comparisons"] if item.get("ambiguity_reason"))
    lines = [
        "# Label Audit Report",
        "",
        f"- reviewed episodes: {report['n']}",
        f"- pending episodes: {pending_count}",
        f"- flagged-ambiguous episodes reviewed: {ambiguous_reviewed}",
        f"- agreements: {report['agreements']}",
        f"- agreement rate: {report['agreement_rate']:.3f}",
        f"- Cohen's kappa: {kappa:.3f}" if kappa is not None else "- Cohen's kappa: undefined (single label category)",
        "",
        "| Episode ID | Rule Label | Human Label | Agreement | Ambiguity Reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in report["comparisons"]:
        lines.append(
            "| {episode_id} | {rule_label} | {human_label} | {agreement} | {ambiguity_reason} |".format(
                episode_id=item["episode_id"],
                rule_label=item["rule_label"],
                human_label=item["human_label"],
                agreement=item["agreement"],
                ambiguity_reason=item.get("ambiguity_reason") or "",
            )
        )
    return "\n".join(lines) + "\n"


def _audit_stratum(item: dict) -> tuple:
    if item["system"] == "S0" and item["episode_kind"] == "stateful_attack":
        category = "s0_sanity"
    elif item["episode_kind"] == "clean_control":
        category = "clean"
    elif item["episode_kind"] == "one_shot_attack":
        category = "one_shot"
    else:
        category = f"delta_{item.get('horizon')}"
    return item.get("actor_model"), category


def _rule_label(item: dict) -> str:
    if item.get("execution_failure"):
        return "execution_failure"
    if item.get("unsafe"):
        return "unsafe"
    return "safe"
