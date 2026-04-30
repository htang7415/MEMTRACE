try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass

import json
from collections import defaultdict

from memtrace.config import EPISODE_SCORES_PATH, METRICS_PATH, SUPPLEMENTARY_TABLES_MD_PATH, TABLE1_MD_PATH


def main() -> None:
    with METRICS_PATH.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with EPISODE_SCORES_PATH.open("r", encoding="utf-8") as handle:
        episode_scores = json.load(handle)

    table1 = build_table1(metrics)
    supplementary = build_supplementary_tables(episode_scores, metrics)

    TABLE1_MD_PATH.write_text(table1, encoding="utf-8")
    SUPPLEMENTARY_TABLES_MD_PATH.write_text(supplementary, encoding="utf-8")

    print(f"table1_path={TABLE1_MD_PATH}")
    print(f"supplementary_tables_path={SUPPLEMENTARY_TABLES_MD_PATH}")


def build_table1(metrics: dict) -> str:
    lines = [
        "Table 1. Main deterministic benchmark results.",
        "",
        "Panel A. Primary outcomes.",
        "",
        "| Actor Model | System | CSR | OVR | OVR 95% CI | SVR | SVR 95% CI | PAR | PAR 95% CI | PAR-exec | PAR-exec 95% CI | Ranking Reversal |",
        "| --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- | ---: | --- | --- |",
    ]
    ranking = metrics["ranking_reversal_by_actor_model"]
    for actor_model, rows in metrics["by_configuration"].items():
        for system in ("S0", "S1", "S2"):
            row = rows[system]
            ranking_cell = "N/A" if system == "S0" else str(bool(ranking.get(actor_model, False)))
            lines.append(
                "| {actor_model} | {system} | {CSR:.3f} | {OVR:.3f} | {OVR_ci} | {SVR:.3f} | {SVR_ci} | {PAR:.3f} | {PAR_ci} | {PAR_exec:.3f} | {PAR_exec_ci} | {ranking_cell} |".format(
                    actor_model=_short_actor_model(actor_model),
                    system=system,
                    CSR=row["CSR"],
                    OVR=row["OVR"],
                    OVR_ci=_format_ci(row["OVR_ci95"]),
                    SVR=row["SVR"],
                    SVR_ci=_format_ci(row["SVR_ci95"]),
                    PAR=row["PAR"],
                    PAR_ci=_format_ci(row["PAR_ci95"]),
                    PAR_exec=row["PAR_executed_only"],
                    PAR_exec_ci=_format_ci(row["PAR_executed_only_ci95"]),
                    ranking_cell=ranking_cell,
                )
            )
    lines.extend(
        [
            "",
            "Panel B. Secondary diagnostics.",
            "",
            "| Actor Model | System | WR | WR 95% CI | HDR | CWRR | φ | 95% CI | φ Status |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for actor_model, rows in metrics["by_configuration"].items():
        for system in ("S0", "S1", "S2"):
            row = rows[system]
            wr = "N/A" if system == "S0" else f"{row['WR']:.3f}"
            cwrr = "N/A" if system != "S2" else f"{row['CWRR']:.3f}"
            phi = "N/A" if row["phi"] is None else f"{row['phi']:.3f}"
            phi_ci = "N/A" if row["phi_ci95"] is None else f"[{row['phi_ci95'][0]:.3f}, {row['phi_ci95'][1]:.3f}]"
            lines.append(
                "| {actor_model} | {system} | {WR} | {WR_ci} | {HDR:.3f} | {CWRR} | {phi} | {phi_ci} | {status} |".format(
                    actor_model=_short_actor_model(actor_model),
                    system=system,
                    WR=wr,
                    WR_ci="N/A" if system == "S0" else _format_ci(row["WR_ci95"]),
                    HDR=row["HDR"],
                    CWRR=cwrr,
                    phi=phi,
                    phi_ci=phi_ci,
                    status=row["phi_status"],
                )
            )
    lines.extend(
        [
            "",
            "Panel C. System averages over actor models.",
            "",
            "| System | CSR | OVR | SVR | PAR | PAR-exec | WR | HDR | CWRR |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for system in ("S0", "S1", "S2"):
        row = metrics["system_average"][system]
        lines.append(
            "| {system} | {CSR:.3f} | {OVR:.3f} | {SVR:.3f} | {PAR:.3f} | {PAR_exec:.3f} | {WR:.3f} | {HDR:.3f} | {CWRR:.3f} |".format(
                system=system,
                CSR=row["CSR"],
                OVR=row["OVR"],
                SVR=row["SVR"],
                PAR=row["PAR"],
                PAR_exec=row["PAR_executed_only"],
                WR=row["WR"],
                HDR=row["HDR"],
                CWRR=row["CWRR"],
            )
        )
    lines.extend(
        [
            "",
            "Panel D. Validity diagnostics for pilot promotion.",
            "",
            "| Actor Model | System | Planner Structured | Required Tool Call | Execution Failure | Writer Structured | Writer Valid Type | Mechanism Check | Official Pilot Valid |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    pilot_validation = metrics.get("pilot_validation_by_configuration", {})
    for actor_model, rows in metrics["by_configuration"].items():
        for system in ("S0", "S1", "S2"):
            row = rows[system]
            pilot_row = pilot_validation.get(actor_model, {}).get(system, {})
            lines.append(
                "| {actor_model} | {system} | {planner:.3f} | {required_tool:.3f} | {execution_failure:.3f} | {writer_structured:.3f} | {writer_valid:.3f} | {par_signal} | {pilot_valid} |".format(
                    actor_model=_short_actor_model(actor_model),
                    system=system,
                    planner=row["planner_structured_output_rate"],
                    required_tool=row["required_tool_call_rate"],
                    execution_failure=row["execution_failure_rate"],
                    writer_structured=row["writer_structured_turn_rate"],
                    writer_valid=row["writer_valid_memory_type_rate"],
                    par_signal=str(bool(pilot_row.get("provenance_writer_mechanism_check_pass", False))),
                    pilot_valid=str(bool(pilot_row.get("official_pilot_valid", False))),
                )
            )
    lines.extend(
        [
            "",
            "Notes: `PAR-exec` conditions on non-execution-failure stateful episodes. Confidence intervals are 95% Wilson intervals under the deterministic `temperature=0` protocol.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_supplementary_tables(episode_scores: list[dict], metrics: dict | None = None) -> str:
    lines = ["Supplementary tables.", ""]
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for item in episode_scores:
        grouped[(item["actor_model"], item["system"], item["task_id"])].append(item)

    for actor_model in sorted({item["actor_model"] for item in episode_scores}):
        for system in ("S0", "S1", "S2"):
            lines.extend(
                [
                    f"## {_short_actor_model(actor_model)} {system} Per-Task PAR and SVR",
                    "",
                    "| Task | PAR | PAR-exec | SVR | Episodes | Executed Episodes |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            rows = []
            for (row_actor, row_system, task_id), items in grouped.items():
                if row_actor != actor_model or row_system != system:
                    continue
                stateful = [item for item in items if item["episode_kind"] == "stateful_attack"]
                if not stateful:
                    continue
                executed_stateful = [item for item in stateful if not item.get("execution_failure")]
                par = sum(1 for item in stateful if item["poison_admission_flag"] is True) / len(stateful)
                par_exec = sum(1 for item in executed_stateful if item["poison_admission_flag"] is True) / len(executed_stateful) if executed_stateful else 0.0
                svr = sum(1 for item in stateful if item["unsafe"]) / len(stateful)
                rows.append((task_id, par, par_exec, svr, len(stateful), len(executed_stateful)))
            for task_id, par, par_exec, svr, count, executed_count in sorted(rows):
                par_exec_str = "N/A" if executed_count == 0 else f"{par_exec:.3f}"
                lines.append(f"| {task_id} | {par:.3f} | {par_exec_str} | {svr:.3f} | {count} | {executed_count} |")
            lines.append("")
            if metrics is not None:
                horizon_row = metrics.get("by_configuration", {}).get(actor_model, {}).get(system, {})
                ef_by_horizon = horizon_row.get("stateful_execution_failure_by_horizon", {})
                if ef_by_horizon:
                    lines.extend(
                        [
                            f"### {_short_actor_model(actor_model)} {system} Stateful Execution Failure by Horizon",
                            "",
                            "| Horizon | Execution Failure |",
                            "| --- | ---: |",
                        ]
                    )
                    for horizon in ("1", "3", "7"):
                        if horizon in ef_by_horizon:
                            lines.append(f"| Δ={horizon} | {ef_by_horizon[horizon]:.3f} |")
            lines.append("")
    return "\n".join(lines) + "\n"


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


def _format_ci(values: list[float]) -> str:
    return f"[{values[0]:.3f}, {values[1]:.3f}]"


if __name__ == "__main__":
    main()
