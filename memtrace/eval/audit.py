"""Audit helpers for validating rule-based labels against human review."""

from __future__ import annotations

from collections import defaultdict


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


def labeler_audit_report(episode_scores: list[dict], audit_records: list[dict]) -> dict:
    """Compare human audit labels with rule-based labels for sampled episodes."""
    score_by_id = {item["episode_id"]: item for item in episode_scores}
    comparisons = []
    for record in audit_records:
        episode_id = record["episode_id"]
        score = score_by_id[episode_id]
        rule_label = "unsafe" if score["unsafe"] else "safe"
        human_label = record["human_label"]
        comparisons.append(
            {
                "episode_id": episode_id,
                "rule_label": rule_label,
                "human_label": human_label,
                "agreement": rule_label == human_label,
            }
        )

    agreements = sum(1 for item in comparisons if item["agreement"])
    return {
        "n": len(comparisons),
        "agreements": agreements,
        "agreement_rate": agreements / len(comparisons) if comparisons else 0.0,
        "comparisons": comparisons,
    }


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
