import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from memtrace.constants import TRUSTED_UTILITY_PAYLOAD_TYPE
from scripts import run_trusted_utility_suite


def test_trusted_utility_suite_dry_run_lists_default_episode_set(capsys) -> None:
    run_trusted_utility_suite.main(["--dry-run"])

    output = capsys.readouterr().out
    assert "utility_episodes=72" in output
    assert TRUSTED_UTILITY_PAYLOAD_TYPE in output


def test_trusted_utility_suite_prepare_only_writes_assets(tmp_path, capsys) -> None:
    run_trusted_utility_suite.main(["--prepare-only", "--out-dir", str(tmp_path)])

    output = capsys.readouterr().out
    episodes = json.loads((tmp_path / "episodes.json").read_text(encoding="utf-8"))
    assert "utility_episodes=72" in output
    assert (tmp_path / "corpus" / "passages.jsonl").exists()
    assert (tmp_path / "corpus" / "allowlist.json").exists()
    assert len(episodes) == 72
    assert {episode["payload_type"] for episode in episodes} == {TRUSTED_UTILITY_PAYLOAD_TYPE}


def test_trusted_utility_suite_profile_smoke_scores_memory_utility(tmp_path) -> None:
    run_trusted_utility_suite.main(
        [
            "--limit",
            "2",
            "--writer-backend",
            "profile",
            "--planner-backend",
            "profile",
            "--out-dir",
            str(tmp_path),
            "--force",
        ]
    )

    utility_scores = json.loads((tmp_path / "utility_scores.json").read_text(encoding="utf-8"))
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert len(utility_scores) == 2
    assert all(row["trusted_memory_admitted"] for row in utility_scores)
    assert all(row["trusted_memory_retrieved_at_trigger"] for row in utility_scores)
    assert all(row["correct_delayed_use"] for row in utility_scores)
    assert metrics["by_system"]["S1"]["correct_delayed_use"] == [2, 2, 1.0]
