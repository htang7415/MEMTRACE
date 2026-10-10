"""Public command-line interface for MEMTRACE."""

from __future__ import annotations

import argparse
import importlib
import sys

from memtrace import __version__

_COMMANDS = {
    ("data", "fetch"): "memtrace.datasets.sources",
    ("data", "verify"): "memtrace.datasets.sources",
    ("run", "experiment"): "memtrace.harness.__main__",
    ("run", "kv-sim"): "memtrace.kv.__main__",
    ("run", "resume"): "memtrace.kv.resume",
    ("run", "restore-bench"): "memtrace.kv.restore_bench",
    ("evaluate", "bfcl"): "memtrace.serving.quality",
    ("report", "retention"): "memtrace.kv.retention",
    ("report", "kv-validate"): "memtrace.kv.validate",
    ("report", "dashboard"): "memtrace.harness.dashboard_export",
    ("components", "check"): "memtrace.components",
}

# Commands whose module takes a subcommand of its own
_LEADING_ARGS = {
    ("data", "fetch"): ["fetch"],
    ("data", "verify"): ["verify"],
    ("run", "experiment"): ["run"],
    ("components", "check"): ["check"],
}


_HELP = {
    "data": "Download and verify the pinned public datasets (data/public/).",
    "run": "Run experiments: harness specs, the KV-cache simulator, the resume scenario, the restore benchmark.",
    "evaluate": "Gate engine changes on task accuracy (BFCL).",
    "report": "Write analyses and the dashboard data.",
    "components": "Check installed engines, images, models and tools against components.lock.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memtrace", description="Serving LLM agents efficiently: engines, routing, gateway, and KV-cache memory."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    groups = parser.add_subparsers(dest="group", required=True)
    for group, help_text in _HELP.items():
        sub = groups.add_parser(group, help=help_text)
        sub.add_argument("action", choices=[action for g, action in _COMMANDS if g == group])
        sub.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    module_name = _COMMANDS[(args.group, args.action)]
    _invoke(module_name, [*_LEADING_ARGS.get((args.group, args.action), []), *args.arguments])
    return 0


def _invoke(module_name: str, arguments: list[str]) -> None:
    module = importlib.import_module(module_name)
    original_argv = sys.argv
    sys.argv = [module_name, *arguments]
    try:
        module.main()
    finally:
        sys.argv = original_argv
