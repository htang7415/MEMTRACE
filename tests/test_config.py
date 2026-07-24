from pathlib import Path

import pytest

from memtrace.config import DEFAULT_ACTOR_MODELS, Settings


def test_settings_defaults_are_rooted_at_working_directory(tmp_path: Path) -> None:
    settings = Settings.load(environ={}, cwd=tmp_path)

    assert settings.root == tmp_path
    assert settings.data_dir == tmp_path / "data"
    assert settings.passages_path == tmp_path / "data" / "corpus" / "passages.jsonl"
    assert settings.actor_models == DEFAULT_ACTOR_MODELS


def test_settings_load_toml_with_environment_override(tmp_path: Path) -> None:
    config_path = tmp_path / "memtrace.toml"
    config_path.write_text(
        "\n".join(
            [
                "[memtrace]",
                'data_dir = "runtime-data"',
                'memory_writer_backend = "profile"',
                'planner_backend = "profile"',
                "top_k = 3",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings.load(
        config_path,
        environ={"MEMTRACE_TOP_K": "7"},
        cwd=tmp_path,
    )

    assert settings.data_dir == tmp_path / "runtime-data"
    assert settings.memory_writer_backend == "profile"
    assert settings.planner_backend == "profile"
    assert settings.top_k == 7


def test_settings_reject_unknown_toml_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "memtrace.toml"
    config_path.write_text("[memtrace]\nunknown = true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown MEMTRACE settings"):
        Settings.load(config_path, environ={}, cwd=tmp_path)
