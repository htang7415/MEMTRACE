import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from memtrace.constants import STATEFUL_STRESS_PAYLOAD_TYPE
from scripts import run_stateful_stress_suite


def test_stateful_stress_suite_dry_run_lists_default_episode_set(capsys) -> None:
    run_stateful_stress_suite.main(["--dry-run"])

    output = capsys.readouterr().out
    assert "stress_episodes=72" in output
    assert STATEFUL_STRESS_PAYLOAD_TYPE in output


def test_stateful_stress_suite_prepare_only_writes_assets(tmp_path, capsys) -> None:
    run_stateful_stress_suite.main(["--prepare-only", "--out-dir", str(tmp_path)])

    output = capsys.readouterr().out
    episodes = json.loads((tmp_path / "episodes.json").read_text(encoding="utf-8"))
    assert "stress_episodes=72" in output
    assert (tmp_path / "corpus" / "passages.jsonl").exists()
    assert (tmp_path / "corpus" / "allowlist.json").exists()
    assert len(episodes) == 72
    assert {episode["payload_type"] for episode in episodes} == {STATEFUL_STRESS_PAYLOAD_TYPE}
