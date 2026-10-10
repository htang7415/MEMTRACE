from __future__ import annotations

import importlib

import pytest

import memtrace.cli as cli


def _calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[str]]]:
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(cli, "_invoke", lambda module_name, arguments: calls.append((module_name, arguments)))
    return calls


def test_cli_dispatches_arguments_to_the_command_module(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _calls(monkeypatch)
    assert cli.main(["run", "resume", "experiments/r0_resume_baseline.yaml", "--out", "x"]) == 0
    assert calls == [("memtrace.kv.resume", ["experiments/r0_resume_baseline.yaml", "--out", "x"])]


def test_cli_adds_the_modules_own_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _calls(monkeypatch)
    cli.main(["run", "experiment", "spec.yaml"])
    cli.main(["components", "check", "--static"])
    assert calls == [
        ("memtrace.harness.__main__", ["run", "spec.yaml"]),
        ("memtrace.components", ["check", "--static"]),
    ]


def test_every_command_module_imports_and_has_main() -> None:
    for module_name in set(cli._COMMANDS.values()):
        assert callable(getattr(importlib.import_module(module_name), "main", None)), module_name


def test_unknown_command_is_rejected() -> None:
    with pytest.raises(SystemExit):
        cli.main(["run", "pilot"])
