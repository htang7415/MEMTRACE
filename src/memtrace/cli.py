"""Public command-line interface for MEMTRACE."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from memtrace import __version__
from memtrace.memrisk.config import Settings, activate


_COMMANDS = {
    ("assets", "corpus"): "memtrace.memrisk.commands.build_corpus",
    ("assets", "episodes"): "memtrace.memrisk.commands.build_episodes",
    ("assets", "index"): "memtrace.memrisk.commands.build_index",
    ("assets", "verify"): "memtrace.memrisk.commands.verify_retrieval",
    ("run", "benchmark"): "memtrace.memrisk.commands.run_experiments",
    ("run", "pilot"): "memtrace.memrisk.commands.run_pilot",
    ("run", "calibration"): "memtrace.memrisk.commands.run_oracle_memory_calibration",
    ("run", "stateful-stress"): "memtrace.memrisk.commands.run_stateful_stress_suite",
    ("run", "trusted-utility"): "memtrace.memrisk.commands.run_trusted_utility_suite",
    ("run", "adversarial-mutation"): "memtrace.memrisk.commands.run_adversarial_mutation_suite",
    ("run", "engine-bench"): "memtrace.serving.engine_bench",
    ("evaluate", "score"): "memtrace.memrisk.evaluation.score",
    ("evaluate", "recompute"): "memtrace.memrisk.evaluation.recompute",
    ("evaluate", "validate"): "memtrace.memrisk.evaluation.validate",
    ("evaluate", "gate"): "memtrace.memrisk.evaluation.gate",
    ("evaluate", "attribute"): "memtrace.memrisk.evaluation.attribute_failures",
    ("evaluate", "audit"): "memtrace.memrisk.evaluation.audit_report",
    ("evaluate", "bfcl"): "memtrace.serving.quality",
    ("report", "tables"): "memtrace.memrisk.evaluation.tables",
    ("report", "figures"): "memtrace.memrisk.evaluation.figures",
    ("report", "explore"): "memtrace.memrisk.evaluation.trace_explorer",
    ("report", "engines"): "memtrace.serving.report",
    ("report", "engine-parity"): "memtrace.serving.parity",
    ("report", "retention"): "memtrace.kv.retention",
    ("report", "kv-sim"): "memtrace.kv.prefix_sim.sweep",
    ("report", "kv-validate"): "memtrace.kv.validate",
}

_ASSET_BUILD_ORDER = (
    "memtrace.memrisk.commands.build_corpus",
    "memtrace.memrisk.commands.build_episodes",
    "memtrace.memrisk.commands.build_index",
    "memtrace.memrisk.commands.verify_retrieval",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memtrace", description="LLM serving and agent KV-cache research, and a memory-risk benchmark for agents."
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
        choices=(
            "benchmark",
            "pilot",
            "calibration",
            "stateful-stress",
            "trusted-utility",
            "adversarial-mutation",
            "engine-bench",
        ),
    )
    run.add_argument("arguments", nargs=argparse.REMAINDER)

    evaluate = groups.add_parser("evaluate", help="Score, validate, and audit results.")
    evaluate.add_argument("action", choices=("score", "recompute", "validate", "gate", "attribute", "audit", "bfcl"))
    evaluate.add_argument("arguments", nargs=argparse.REMAINDER)

    report = groups.add_parser("report", help="Generate tables and figures.")
    report.add_argument(
        "action",
        choices=("tables", "figures", "explore", "engines", "engine-parity", "retention", "kv-sim", "kv-validate"),
    )
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
