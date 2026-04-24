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
        "# Table 1",
        "",
        "| Actor Model | System | CSR | OVR | SVR | SRG | PAR | Ranking Reversal |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    ranking = metrics["ranking_reversal_by_actor_model"]
    for actor_model, rows in metrics["by_configuration"].items():
        for system in ("S0", "S1", "S2"):
            row = rows[system]
            ranking_cell = "N/A" if system == "S0" else str(bool(ranking.get(actor_model, False)))
            lines.append(
                "| {actor_model} | {system} | {CSR:.3f} | {OVR:.3f} | {SVR:.3f} | {SRG:.3f} | {PAR:.3f} | {ranking_cell} |".format(
                    actor_model=_short_actor_model(actor_model),
                    system=system,
                    CSR=row["CSR"],
                    OVR=row["OVR"],
                    SVR=row["SVR"],
                    SRG=row["SRG"],
                    PAR=row["PAR"],
                    ranking_cell=ranking_cell,
                )
            )
    lines.extend(
        [
            "",
            "Secondary metrics:",
            "",
            "| Actor Model | System | WR | HDR | CWRR | φ | 95% CI | φ Status |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
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
                "| {actor_model} | {system} | {WR} | {HDR:.3f} | {CWRR} | {phi} | {phi_ci} | {status} |".format(
                    actor_model=_short_actor_model(actor_model),
                    system=system,
                    WR=wr,
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
            "Averaged over actor models:",
            "",
            "| System | CSR | OVR | SVR | SRG | PAR | WR | HDR | CWRR |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for system in ("S0", "S1", "S2"):
        row = metrics["system_average"][system]
        lines.append(
            "| {system} | {CSR:.3f} | {OVR:.3f} | {SVR:.3f} | {SRG:.3f} | {PAR:.3f} | {WR:.3f} | {HDR:.3f} | {CWRR:.3f} |".format(
                system=system,
                CSR=row["CSR"],
                OVR=row["OVR"],
                SVR=row["SVR"],
                SRG=row["SRG"],
                PAR=row["PAR"],
                WR=row["WR"],
                HDR=row["HDR"],
                CWRR=row["CWRR"],
            )
        )
    lines.extend(
        [
            "",
            "Validity diagnostics:",
            "",
            "| Actor Model | System | Planner Structured | Required Tool Call | Writer Structured | Writer Valid Type | Official Pilot Valid |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    pilot_validation = metrics.get("pilot_validation_by_configuration", {})
    for actor_model, rows in metrics["by_configuration"].items():
        for system in ("S0", "S1", "S2"):
            row = rows[system]
            pilot_row = pilot_validation.get(actor_model, {}).get(system, {})
            lines.append(
                "| {actor_model} | {system} | {planner:.3f} | {required_tool:.3f} | {writer_structured:.3f} | {writer_valid:.3f} | {pilot_valid} |".format(
                    actor_model=_short_actor_model(actor_model),
                    system=system,
                    planner=row["planner_structured_output_rate"],
                    required_tool=row["required_tool_call_rate"],
                    writer_structured=row["writer_structured_turn_rate"],
                    writer_valid=row["writer_valid_memory_type_rate"],
                    pilot_valid=str(bool(pilot_row.get("official_pilot_valid", False))),
                )
            )
    return "\n".join(lines) + "\n"


def build_supplementary_tables(episode_scores: list[dict], metrics: dict | None = None) -> str:
    del metrics
    lines = ["# Supplementary Tables", ""]
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for item in episode_scores:
        grouped[(item["actor_model"], item["system"], item["task_id"])].append(item)

    for actor_model in sorted({item["actor_model"] for item in episode_scores}):
        for system in ("S0", "S1", "S2"):
            lines.extend(
                [
                    f"## {_short_actor_model(actor_model)} {system} Per-Task PAR and SVR",
                    "",
                    "| Task | PAR | SVR | Episodes |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            rows = []
            for (row_actor, row_system, task_id), items in grouped.items():
                if row_actor != actor_model or row_system != system:
                    continue
                stateful = [item for item in items if item["episode_kind"] == "stateful_attack"]
                if not stateful:
                    continue
                par = sum(1 for item in stateful if item["poison_admission_flag"] is True) / len(stateful)
                svr = sum(1 for item in stateful if item["unsafe"]) / len(stateful)
                rows.append((task_id, par, svr, len(stateful)))
            for task_id, par, svr, count in sorted(rows):
                lines.append(f"| {task_id} | {par:.3f} | {svr:.3f} | {count} |")
            lines.append("")
    return "\n".join(lines) + "\n"


def _short_actor_model(actor_model: str) -> str:
    if "Qwen2.5-3B" in actor_model:
        return "Qwen2.5-3B"
    if "Llama-3.2-3B" in actor_model:
        return "Llama-3.2-3B"
    return actor_model


if __name__ == "__main__":
    main()
