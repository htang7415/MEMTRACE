try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass

import argparse
import json
from pathlib import Path

from memtrace.config import METRICS_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MEMTRACE pilot metrics.")
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH)
    args = parser.parse_args()

    with args.metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    report = build_validation_report(metrics)
    print(report)
    raise SystemExit(0 if overall_pilot_valid(metrics) else 1)


def build_validation_report(metrics: dict) -> str:
    lines = [
        "# Pilot Validation",
        "",
        "| Actor Model | System | Planner Structured | Required Tool Call | Execution Failure | Writer Structured | Writer Valid Type | S0 Sanity | Mechanism Check | Official Pilot Valid |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    pilot_validation = metrics.get("pilot_validation_by_configuration", {})
    for actor_model, rows in metrics["by_configuration"].items():
        for system in ("S0", "S1", "S2"):
            row = rows.get(system)
            if row is None:
                lines.append(
                    f"| {_short_actor_model(actor_model)} | {system} | n/a | n/a | n/a | n/a | n/a | False | False | False |"
                )
                continue
            gate_row = pilot_validation.get(actor_model, {}).get(system, {})
            lines.append(
                "| {actor_model} | {system} | {planner:.3f} | {required_tool:.3f} | {execution_failure:.3f} | {writer_structured:.3f} | {writer_valid:.3f} | {s0_sanity} | {par_signal} | {pilot_valid} |".format(
                    actor_model=_short_actor_model(actor_model),
                    system=system,
                    planner=row["planner_structured_output_rate"],
                    required_tool=row["required_tool_call_rate"],
                    execution_failure=row["execution_failure_rate"],
                    writer_structured=row["writer_structured_turn_rate"],
                    writer_valid=row["writer_valid_memory_type_rate"],
                    s0_sanity=str(bool(gate_row.get("s0_stateful_sanity_pass", False))),
                    par_signal=str(bool(gate_row.get("provenance_writer_mechanism_check_pass", False))),
                    pilot_valid=str(bool(gate_row.get("official_pilot_valid", False))),
                )
            )
    return "\n".join(lines) + "\n"


def overall_pilot_valid(metrics: dict) -> bool:
    pilot_validation = metrics.get("pilot_validation_by_configuration", {})
    if not pilot_validation:
        return False
    for actor_rows in pilot_validation.values():
        if any(system not in actor_rows for system in ("S0", "S1", "S2")):
            return False
        if not all(actor_rows[system].get("official_pilot_valid", False) for system in ("S0", "S1", "S2")):
            return False
    return True


def _short_actor_model(actor_model: str) -> str:
    if "Qwen2.5-3B" in actor_model:
        return "Qwen2.5-3B"
    if "Qwen2.5-7B" in actor_model:
        return "Qwen2.5-7B"
    if "Llama-3.2-3B" in actor_model:
        return "Llama-3.2-3B"
    if "Llama-3.1-8B" in actor_model:
        return "Llama-3.1-8B"
    return actor_model


if __name__ == "__main__":
    main()
