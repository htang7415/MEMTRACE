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
    supplementary = build_supplementary_tables(episode_scores)

    TABLE1_MD_PATH.write_text(table1, encoding="utf-8")
    SUPPLEMENTARY_TABLES_MD_PATH.write_text(supplementary, encoding="utf-8")

    print(f"table1_path={TABLE1_MD_PATH}")
    print(f"supplementary_tables_path={SUPPLEMENTARY_TABLES_MD_PATH}")


def build_table1(metrics: dict) -> str:
    lines = [
        "# Table 1",
        "",
        "| System | CSR | OVR | SVR | SRG | PAR | WR | HDR | CWRR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system in ("S0", "S1", "S2"):
        row = metrics[system]
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
            "Stateful by horizon:",
            "",
            "| System | VR@Δ1 | VR@Δ3 | VR@Δ7 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for system in ("S0", "S1", "S2"):
        horizon = metrics[system]["stateful_by_horizon"]
        lines.append(
            "| {system} | {d1:.3f} | {d3:.3f} | {d7:.3f} |".format(
                system=system,
                d1=horizon.get("1", 0.0),
                d3=horizon.get("3", 0.0),
                d7=horizon.get("7", 0.0),
            )
        )
    return "\n".join(lines) + "\n"


def build_supplementary_tables(episode_scores: list[dict]) -> str:
    lines = ["# Supplementary Tables", ""]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in episode_scores:
        task_id = _task_id_from_episode_id(item["episode_id"])
        grouped[(item["system"], task_id)].append(item)

    for system in ("S0", "S1", "S2"):
        lines.extend(
            [
                f"## {system} Per-Task PAR and SVR",
                "",
                "| Task | PAR | SVR | Episodes |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        rows = []
        for (row_system, task_id), items in grouped.items():
            if row_system != system:
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


def _task_id_from_episode_id(episode_id: str) -> str:
    parts = episode_id.split(":")
    if len(parts) < 2:
        return episode_id
    return parts[1]


if __name__ == "__main__":
    main()
