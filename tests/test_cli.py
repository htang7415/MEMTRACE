import memtrace.cli as cli


def test_cli_dispatches_evaluate_gate(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "_invoke", lambda module_name, arguments: calls.append((module_name, arguments)))

    result = cli.main(["evaluate", "gate", "--baseline", "a.json", "--candidate", "b.json"])

    assert result == 0
    assert calls == [("memtrace.memrisk.evaluation.gate", ["--baseline", "a.json", "--candidate", "b.json"])]


def test_cli_dispatches_report_explore(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "_invoke", lambda module_name, arguments: calls.append((module_name, arguments)))

    result = cli.main(["report", "explore", "--system", "S1"])

    assert result == 0
    assert calls == [("memtrace.memrisk.evaluation.trace_explorer", ["--system", "S1"])]


def test_cli_dispatches_adversarial_mutation_run(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "_invoke", lambda module_name, arguments: calls.append((module_name, arguments)))

    result = cli.main(["run", "adversarial-mutation", "--dry-run"])

    assert result == 0
    assert calls == [("memtrace.memrisk.commands.run_adversarial_mutation_suite", ["--dry-run"])]


def test_cli_dispatches_run_arguments(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "_invoke", lambda module_name, arguments: calls.append((module_name, arguments)))

    result = cli.main(["run", "pilot", "--dry-run", "--limit", "2"])

    assert result == 0
    assert calls == [("memtrace.memrisk.commands.run_pilot", ["--dry-run", "--limit", "2"])]


def test_cli_builds_assets_in_order(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "_invoke", lambda module_name, arguments: calls.append((module_name, arguments)))

    result = cli.main(["assets", "build"])

    assert result == 0
    assert calls == [(module_name, []) for module_name in cli._ASSET_BUILD_ORDER]


def test_cli_dispatches_components_check(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "_invoke", lambda module_name, arguments: calls.append((module_name, arguments)))

    assert cli.main(["components", "check", "--static"]) == 0
    assert calls == [("memtrace.components", ["check", "--static"])]
