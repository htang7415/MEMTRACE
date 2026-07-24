import memtrace.cli as cli


def test_cli_dispatches_run_arguments(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "_invoke", lambda module_name, arguments: calls.append((module_name, arguments)))

    result = cli.main(["run", "pilot", "--dry-run", "--limit", "2"])

    assert result == 0
    assert calls == [("memtrace.commands.run_pilot", ["--dry-run", "--limit", "2"])]


def test_cli_builds_assets_in_order(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "_invoke", lambda module_name, arguments: calls.append((module_name, arguments)))

    result = cli.main(["assets", "build"])

    assert result == 0
    assert calls == [(module_name, []) for module_name in cli._ASSET_BUILD_ORDER]
