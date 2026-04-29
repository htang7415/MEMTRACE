import json
from pathlib import Path

import pytest

from scripts.promote_results import promote_results


def test_promote_results_copies_validated_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for filename in (
        "run_summary.json",
        "episode_scores.json",
        "table1.md",
        "supplementary_tables.md",
    ):
        (source / filename).write_text("[]\n", encoding="utf-8")
    (source / "run_summary.json").write_text(json.dumps([{"episode_id": "ep1"}]), encoding="utf-8")
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
    (source / "figures").mkdir()
    (source / "figures" / "figure1.svg").write_text("<svg/>", encoding="utf-8")

    count = promote_results(source, tmp_path / "results", tmp_path / "figures")

    assert count == 1
    assert (tmp_path / "results" / "metrics.json").exists()
    assert (tmp_path / "figures" / "figure1.svg").exists()


def test_promote_results_rejects_invalid_rows(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for filename in (
        "run_summary.json",
        "episode_scores.json",
        "table1.md",
        "supplementary_tables.md",
    ):
        (source / filename).write_text("[]\n", encoding="utf-8")
    (source / "metrics.json").write_text(
        json.dumps(
            {
                "pilot_validation_by_configuration": {
                    "model-a": {
                        "S0": {"official_pilot_valid": False},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        promote_results(source, tmp_path / "results", tmp_path / "figures")
