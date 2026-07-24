"""Public command-line interface for MEMTRACE."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from memtrace import __version__
from memtrace.config import Settings, activate


_COMMANDS = {
    ("assets", "corpus"): "memtrace.commands.build_corpus",
    ("assets", "episodes"): "memtrace.commands.build_episodes",
    ("assets", "index"): "memtrace.commands.build_index",
    ("assets", "verify"): "memtrace.commands.verify_retrieval",
    ("run", "benchmark"): "memtrace.commands.run_experiments",
    ("run", "pilot"): "memtrace.commands.run_pilot",
    ("run", "calibration"): "memtrace.commands.run_oracle_memory_calibration",
    ("run", "stateful-stress"): "memtrace.commands.run_stateful_stress_suite",
    ("run", "trusted-utility"): "memtrace.commands.run_trusted_utility_suite",
    ("evaluate", "score"): "memtrace.evaluation.score",
    ("evaluate", "recompute"): "memtrace.evaluation.recompute",
    ("evaluate", "validate"): "memtrace.evaluation.validate",
    ("evaluate", "attribute"): "memtrace.evaluation.attribute_failures",
    ("evaluate", "audit"): "memtrace.evaluation.audit_report",
    ("report", "tables"): "memtrace.evaluation.tables",
    ("report", "figures"): "memtrace.evaluation.figures",
}

_ASSET_BUILD_ORDER = (
    "memtrace.commands.build_corpus",
    "memtrace.commands.build_episodes",
    "memtrace.commands.build_index",
    "memtrace.commands.verify_retrieval",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memtrace", description="Evaluate persistent-memory risk in tool-using agents."
    )
    parser.add_argument("--config", type=Path, help="TOML file containing a [memtrace] settings table.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    groups = parser.add_subparsers(dest="group", required=True)

    assets = groups.add_parser("assets", help="Build and verify benchmark assets.")
    assets.add_argument("action", choices=("build", "corpus", "episodes", "index", "verify"))
    assets.add_argument("arguments", nargs=argparse.REMAINDER)

    run = groups.add_parser("run", help="Run benchmark and diagnostic workloads.")
    run.add_argument(
        "action",
        choices=("benchmark", "pilot", "calibration", "stateful-stress", "trusted-utility"),
    )
    run.add_argument("arguments", nargs=argparse.REMAINDER)

    evaluate = groups.add_parser("evaluate", help="Score, validate, and audit results.")
    evaluate.add_argument("action", choices=("score", "recompute", "validate", "attribute", "audit"))
    evaluate.add_argument("arguments", nargs=argparse.REMAINDER)

    report = groups.add_parser("report", help="Generate tables and figures.")
    report.add_argument("action", choices=("tables", "figures"))
    report.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    activate(Settings.load(args.config))

    if args.group == "assets" and args.action == "build":
        if args.arguments:
            raise SystemExit("memtrace assets build does not accept additional arguments")
        for module_name in _ASSET_BUILD_ORDER:
            _invoke(module_name, [])
        return 0

    module_name = _COMMANDS[(args.group, args.action)]
    _invoke(module_name, args.arguments)
    return 0


def _invoke(module_name: str, arguments: list[str]) -> None:
    module = importlib.import_module(module_name)
    original_argv = sys.argv
    sys.argv = [module_name, *arguments]
    try:
        module.main()
    finally:
        sys.argv = original_argv
