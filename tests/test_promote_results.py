import json
from pathlib import Path

import pytest

from scripts.promote_results import promote_results


def test_promote_results_copies_validated_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("{}\n", encoding="utf-8")
    for filename in (
        "run_summary.json",
        "episode_scores.json",
        "table1.md",
        "supplementary_tables.md",
    ):
        (source / filename).write_text("[]\n", encoding="utf-8")
    (source / "run_summary.json").write_text(json.dumps([{"episode_id": "ep1", "trace_path": str(trace_path)}]), encoding="utf-8")
    (source / "metrics.json").write_text(
        json.dumps(
            {
                "pilot_validation_by_configuration": {
                    "model-a": {
                        "S0": {"official_pilot_valid": True},
                        "S1": {"official_pilot_valid": True},
                        "S2": {"official_pilot_valid": True},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (source / "figures").mkdir()
    (source / "figures" / "figure1.svg").write_text("<svg/>", encoding="utf-8")

    count = promote_results(source, tmp_path / "results", tmp_path / "figures", expected_episodes=1)

    assert count == 1
    assert (tmp_path / "results" / "metrics.json").exists()
    assert (tmp_path / "figures" / "figure1.svg").exists()


def test_promote_results_rejects_invalid_rows(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("{}\n", encoding="utf-8")
    for filename in (
        "run_summary.json",
        "episode_scores.json",
        "table1.md",
        "supplementary_tables.md",
    ):
        (source / filename).write_text("[]\n", encoding="utf-8")
    (source / "run_summary.json").write_text(json.dumps([{"episode_id": "ep1", "trace_path": str(trace_path)}]), encoding="utf-8")
    (source / "metrics.json").write_text(
        json.dumps(
            {
                "pilot_validation_by_configuration": {
                    "model-a": {
                        "S0": {"official_pilot_valid": False},
                        "S1": {"official_pilot_valid": True},
                        "S2": {"official_pilot_valid": True},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        promote_results(source, tmp_path / "results", tmp_path / "figures", expected_episodes=1)


def test_promote_results_rejects_wrong_episode_count(tmp_path: Path) -> None:
    source = _valid_source(tmp_path)

    with pytest.raises(SystemExit, match="1/2 episodes"):
        promote_results(source, tmp_path / "results", tmp_path / "figures", expected_episodes=2)


def test_promote_results_rejects_incomplete_system_rows(tmp_path: Path) -> None:
    source = _valid_source(tmp_path)
    (source / "metrics.json").write_text(
        json.dumps(
            {
                "pilot_validation_by_configuration": {
                    "model-a": {
                        "S0": {"official_pilot_valid": True},
                        "S1": {"official_pilot_valid": True},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="incomplete pilot validation"):
        promote_results(source, tmp_path / "results", tmp_path / "figures", expected_episodes=1)


def test_promote_results_rejects_missing_trace(tmp_path: Path) -> None:
    source = _valid_source(tmp_path)
    (source / "run_summary.json").write_text(
        json.dumps([{"episode_id": "ep1", "trace_path": str(tmp_path / "missing.jsonl")}]),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="missing traces"):
        promote_results(source, tmp_path / "results", tmp_path / "figures", expected_episodes=1)


def test_promote_results_rejects_low_audit_agreement(tmp_path: Path) -> None:
    source = _valid_source(tmp_path)
    audit_template = tmp_path / "audit_template.jsonl"
    audit_template.write_text(json.dumps({"episode_id": "ep1", "human_label": "safe"}) + "\n", encoding="utf-8")
    audit_report = tmp_path / "audit_report.json"
    audit_report.write_text(json.dumps({"agreement_rate": 0.5}), encoding="utf-8")

    with pytest.raises(SystemExit, match="audit agreement"):
        promote_results(
            source,
            tmp_path / "results",
            tmp_path / "figures",
            expected_episodes=1,
            audit_template_path=audit_template,
            audit_report_path=audit_report,
        )


def test_promote_results_rejects_incomplete_audit_labels(tmp_path: Path) -> None:
    source = _valid_source(tmp_path)
    audit_template = tmp_path / "audit_template.jsonl"
    audit_template.write_text(
        "\n".join(
            [
                json.dumps({"episode_id": "ep1", "human_label": "safe"}),
                json.dumps({"episode_id": "ep2", "human_label": None}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    audit_report = tmp_path / "audit_report.json"
    audit_report.write_text(json.dumps({"agreement_rate": 1.0}), encoding="utf-8")

    with pytest.raises(SystemExit, match="incomplete audit labels: 1/2 reviewed"):
        promote_results(
            source,
            tmp_path / "results",
            tmp_path / "figures",
            expected_episodes=1,
            audit_template_path=audit_template,
            audit_report_path=audit_report,
        )


def test_promote_results_publishes_validated_audit_outputs(tmp_path: Path) -> None:
    source = _valid_source(tmp_path)
    for filename in ("audit_sample.json", "audit_report.md", "audit_review.md"):
        (source / filename).write_text(f"{filename}\n", encoding="utf-8")
    audit_template = source / "audit_template.jsonl"
    audit_template.write_text(json.dumps({"episode_id": "ep1", "human_label": "safe"}) + "\n", encoding="utf-8")
    audit_report = source / "audit_report.json"
    audit_report.write_text(json.dumps({"agreement_rate": 1.0}), encoding="utf-8")
    audit_dir = tmp_path / "audit"

    promote_results(
        source,
        tmp_path / "results",
        tmp_path / "figures",
        expected_episodes=1,
        audit_template_path=audit_template,
        audit_report_path=audit_report,
        audit_dir=audit_dir,
    )

    assert (audit_dir / "audit_template.jsonl").read_text(encoding="utf-8") == audit_template.read_text(encoding="utf-8")
    assert json.loads((audit_dir / "audit_report.json").read_text(encoding="utf-8"))["agreement_rate"] == 1.0
    assert (audit_dir / "audit_review.md").read_text(encoding="utf-8") == "audit_review.md\n"


def _valid_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("{}\n", encoding="utf-8")
    for filename in (
        "episode_scores.json",
        "table1.md",
        "supplementary_tables.md",
    ):
        (source / filename).write_text("[]\n", encoding="utf-8")
    (source / "run_summary.json").write_text(json.dumps([{"episode_id": "ep1", "trace_path": str(trace_path)}]), encoding="utf-8")
    (source / "metrics.json").write_text(
        json.dumps(
            {
                "pilot_validation_by_configuration": {
                    "model-a": {
                        "S0": {"official_pilot_valid": True},
                        "S1": {"official_pilot_valid": True},
                        "S2": {"official_pilot_valid": True},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return source
