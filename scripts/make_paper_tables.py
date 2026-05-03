try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from memtrace.eval.metrics import wilson_ci

DEFAULT_METRICS_PATH = Path("data/results/metrics.json")
DEFAULT_CALIBRATION_METRICS_PATH = Path("data/calibration/oracle_memory/metrics.json")
DEFAULT_EPISODE_SCORES_PATH = Path("data/results/episode_scores.json")
DEFAULT_ATTRIBUTION_LABELS_PATH = Path("data/results/attribution_labels.json")
DEFAULT_OUTPUT_DIR = Path("paper/manuscript/tables")
DEFAULT_ATTEMPTED_RUNS_SCORER_ID = "v1"

ACTOR_CODES = {
    "mlx-community/Qwen2.5-7B-Instruct-4bit": "Q7",
    "mlx-community/Qwen2.5-3B-Instruct-4bit": "Q3",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit": "L31",
    "mlx-community/Llama-3.2-3B-Instruct-4bit": "L32",
}

EPISODE_KIND_CODES = {
    "clean_control": "C",
    "one_shot_attack": "O",
    "stateful_attack": "T",
}

RELEASE_ROOT_CANDIDATES = (Path("paper/anonymous_memtrace"), Path("release"))
REQUIRED_TRACE_FIELDS = {
    "trace_id",
    "protocol_version",
    "actor_model",
    "actor_revision",
    "tokenizer_id",
    "tokenizer_revision",
    "writer_backend",
    "planner_backend",
    "system_id",
    "episode_kind",
    "task_id",
    "payload_id",
    "payload_type",
    "horizon",
    "retrieved_passage_ids_initial",
    "poison_retrieved_initial",
    "writer_candidates",
    "accepted_memory_ids",
    "admitted_poison",
    "retrieved_memory_ids_trigger",
    "admitted_poison_retrieved_at_trigger",
    "unsafe_proposal_before_checker",
    "unsafe_blocked_by_checker",
    "unsafe_executed",
    "execution_failure",
    "violation_attribution",
    "final_tool_call",
    "policy_checker_verdict",
    "scorer_version",
    "scorer_hash",
    "trace_created_at",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables for the MEMTRACE paper draft.")
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--calibration-metrics-path", type=Path, default=DEFAULT_CALIBRATION_METRICS_PATH)
    parser.add_argument("--episode-scores-path", type=Path, default=DEFAULT_EPISODE_SCORES_PATH)
    parser.add_argument("--attribution-labels-path", type=Path, default=DEFAULT_ATTRIBUTION_LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--attempted-runs-data-dir", type=Path, default=Path("data"))
    parser.add_argument("--attempted-runs-scorer-id", default=DEFAULT_ATTEMPTED_RUNS_SCORER_ID)
    args = parser.parse_args()

    with args.metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    calibration_metrics = metrics
    if args.calibration_metrics_path.exists():
        with args.calibration_metrics_path.open("r", encoding="utf-8") as handle:
            calibration_metrics = json.load(handle)
    with args.episode_scores_path.open("r", encoding="utf-8") as handle:
        episode_scores = json.load(handle)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "main_results.tex").write_text(build_main_results_table(metrics), encoding="utf-8")
    (args.output_dir / "mechanism_counts.tex").write_text(build_mechanism_counts_table(metrics), encoding="utf-8")
    (args.output_dir / "causal_chain_counts.tex").write_text(build_causal_chain_counts_table(metrics), encoding="utf-8")
    (args.output_dir / "oracle_memory_calibration.tex").write_text(
        build_oracle_memory_calibration_table(calibration_metrics),
        encoding="utf-8",
    )
    (args.output_dir / "validity_gates.tex").write_text(build_validity_gates_table(metrics), encoding="utf-8")
    (args.output_dir / "confidence_intervals.tex").write_text(
        build_confidence_intervals_table(metrics, calibration_metrics),
        encoding="utf-8",
    )
    (args.output_dir / "task_level_signal.tex").write_text(build_task_level_signal_table(episode_scores), encoding="utf-8")
    (args.output_dir / "attempted_runs.tex").write_text(
        build_attempted_runs_table(args.attempted_runs_data_dir, args.attempted_runs_scorer_id),
        encoding="utf-8",
    )
    if args.attribution_labels_path.exists():
        with args.attribution_labels_path.open("r", encoding="utf-8") as handle:
            attribution_labels = json.load(handle)
        (args.output_dir / "failure_attribution.tex").write_text(
            build_failure_attribution_table(attribution_labels),
            encoding="utf-8",
        )

    print(f"paper_tables_dir={args.output_dir}")


def build_main_results_table(metrics: dict) -> str:
    actor_model = next(iter(metrics["by_configuration"]))
    rows = metrics["by_configuration"][actor_model]
    lines = [
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llllll}",
        "\\toprule",
        "System & Clean success & One-shot unsafe & Memory-path signal & Delayed outcome & Exec failure \\\\",
        "\\midrule",
    ]
    for system in ("S0", "S1", "S2"):
        row = rows[system]
        counts = row["causal_chain_counts"]
        lines.append(
            "{system} & {CSR} & {OVR} & {memory_path} & {delayed_outcome} & {EFR} \\\\".format(
                system=system,
                CSR=_count_rate_cell(row, "clean_success", "CSR"),
                OVR=_event_count_rate_cell(row, "one_shot_unsafe", "OVR"),
                memory_path=_memory_path_cell(system, counts),
                delayed_outcome=_delayed_outcome_cell(system, row, counts),
                EFR=_event_count_rate_cell(row, "execution_failure", "execution_failure_rate"),
            )
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "",
            "\\footnotesize Memory-path signal summarizes poisoned-memory admission and trigger retrieval. Delayed outcome summarizes unsafe stateful proposal and execution, separated from execution-format failures.",
            "",
        ]
    )
    return "\n".join(lines)


def build_mechanism_counts_table(metrics: dict) -> str:
    actor_model = next(iter(metrics["by_configuration"]))
    rows = metrics["by_configuration"][actor_model]
    lines = [
        "\\begin{tabular}{lll}",
        "\\toprule",
        "System & Observed mechanism & Interpretation \\\\",
        "\\midrule",
    ]
    for system in ("S0", "S1", "S2"):
        counts = rows[system]["mechanism_counts"]
        admission_safe = counts["b_admission_and_no_violation"]
        admission_violate = counts["a_admission_and_violation"]
        no_admission_safe = counts["d_no_admission_and_no_violation"]
        if admission_safe:
            mechanism = f"{admission_safe} admit-and-safe; no admit-and-violate" if admission_violate == 0 else f"{admission_safe} admit-and-safe; {admission_violate} admit-and-violate"
            interpretation = "poison can enter memory, but does not become a delayed violation"
        elif system == "S2":
            mechanism = f"{no_admission_safe} executed stateful attacks with no poison admission"
            interpretation = "provenance-aware writer blocks poisoned memory candidates"
        elif system == "S0":
            mechanism = f"{no_admission_safe} executed stateful attacks with no persisted memory"
            interpretation = "no-memory baseline exercises the same scorer path"
        else:
            mechanism = f"{no_admission_safe} executed stateful attacks without admission"
            interpretation = "no delayed violation observed"
        lines.append(f"{system} & {_latex_escape(mechanism)} & {_latex_escape(interpretation)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def build_causal_chain_counts_table(metrics: dict) -> str:
    actor_model = next(iter(metrics["by_configuration"]))
    rows = metrics["by_configuration"][actor_model]
    s1_counts = rows["S1"]["causal_chain_counts"]
    stateful_per_system = s1_counts["stateful_attacks"]
    admitted = s1_counts["poison_admitted"]
    retrieved = s1_counts["admitted_poison_retrieved_at_trigger"]
    proposal = s1_counts["unsafe_proposal_before_checker"]
    executed = s1_counts["unsafe_executed"]
    failures = s1_counts["execution_failure"]
    lines = [
        "\\begin{tabular}{lll}",
        "\\toprule",
        "Stateful-path checkpoint & Count & Interpretation \\\\",
        "\\midrule",
        f"Stateful attacks & {stateful_per_system} per system & fixed main-protocol denominator \\\\",
        f"S1 poison admitted & {admitted}/{stateful_per_system} & permissive writer admits a small localized subset \\\\",
        f"S1 admitted poison retrieved & {retrieved}/{admitted} & admitted records reach the trigger context \\\\",
        f"Unsafe proposal or execution & {_none_or_count(proposal + executed)} & chain stops before planner adoption/execution \\\\",
        f"Execution failures & {failures} stateful per system & tracked separately from unsafe actions \\\\",
    ]
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def build_oracle_memory_calibration_table(metrics: dict) -> str:
    rows = metrics.get("calibration_by_condition", {})
    lines = [
        "\\begin{tabular}{ll}",
        "\\toprule",
        "Field & Value \\\\",
        "\\midrule",
    ]
    if rows:
        for condition, row in sorted(rows.items()):
            lines.extend(
                [
                    f"Condition & {_latex_escape(condition)} \\\\",
                    f"Episodes & {row['episodes']} \\\\",
                    "Oracle inserted & {value} \\\\".format(
                        value=_count_rate_from_values(
                            row["oracle_memory_inserted"],
                            row["episodes"],
                            row["oracle_memory_inserted"] / row["episodes"] if row["episodes"] else 0.0,
                        )
                    ),
                    "Oracle retrieved & {value} \\\\".format(
                        value=_count_rate_from_values(
                            row["oracle_memory_retrieved"],
                            row["oracle_memory_inserted"],
                            row["CAL_PRR"],
                        )
                    ),
                    "Unsafe proposal & {value} \\\\".format(
                        value=_count_rate_from_values(
                            row["unsafe_proposal_before_checker"],
                            row["parseable_planner_outputs"],
                            row["CAL_UPR"],
                        )
                    ),
                    f"Unsafe executed & {_count_rate_from_values(row['unsafe_executed'], row['episodes'], row['CAL_SVR'])} \\\\",
                    f"Execution failure & {_none_fraction(row['execution_failure'], row['episodes'])} \\\\",
                    f"CAL-PRR / CAL-UPR / CAL-SVR / CAL-EFR & {row['CAL_PRR']:.3f} / {row['CAL_UPR']:.3f} / {row['CAL_SVR']:.3f} / {_none_rate(row['CAL_EFR'])} \\\\",
                ]
            )
    else:
        lines.append("\\multicolumn{2}{l}{Calibration metrics not generated.} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def build_validity_gates_table(metrics: dict, release_root: Path | None = None) -> str:
    actor_model = next(iter(metrics["by_configuration"]))
    rows = metrics["by_configuration"][actor_model]
    gates = metrics["pilot_validation_by_configuration"][actor_model]
    s1_par = rows.get("S1", {}).get("PAR", 0.0)
    s2_par = rows.get("S2", {}).get("PAR", 0.0)
    lines = [
        "\\begin{tabular}{@{}p{0.31\\linewidth}p{0.18\\linewidth}lp{0.30\\linewidth}@{}}",
        "\\toprule",
        "Check & Category & Result & Value \\\\",
        "\\midrule",
        _gate_row(
            "required tool-call rate $\\ge$ 0.90",
            "gate",
            all(gates[system].get("required_tool_call_rate_pass", False) for system in ("S0", "S1", "S2")),
            _system_values(rows, "required_tool_call_rate"),
        ),
        _gate_row(
            "execution-failure rate $\\le$ 0.10",
            "gate",
            all(gates[system].get("execution_failure_rate_pass", False) for system in ("S0", "S1", "S2")),
            _system_values(rows, "execution_failure_rate"),
        ),
        _gate_row(
            "writer structured-output rate $\\ge$ 0.90",
            "gate",
            all(gates[system].get("writer_structured_turn_rate_pass", False) for system in ("S0", "S1", "S2")),
            _system_values(rows, "writer_structured_turn_rate"),
        ),
        _gate_row(
            "writer valid-memory-type rate $\\ge$ 0.90",
            "gate",
            all(gates[system].get("writer_valid_memory_type_rate_pass", False) for system in ("S0", "S1", "S2")),
            _system_values(rows, "writer_valid_memory_type_rate"),
        ),
        _gate_row(
            "S0 stateful violation rate $\\le$ 0.05",
            "gate",
            all(gates[system].get("s0_stateful_sanity_pass", False) for system in ("S0", "S1", "S2")),
            "no S0 stateful violations observed" if rows.get("S0", {}).get("SVR", 0.0) == 0.0 else f"S0 SVR={rows.get('S0', {}).get('SVR', 0.0):.3f}",
        ),
    ]
    validation = _release_validation_counts(release_root or _available_release_root())
    if validation:
        lines.extend(
            [
                _gate_row(
                    "retrieval verification present for every main episode",
                    "gate",
                    all(value[0] == value[1] for value in validation["main_retrieval"].values()),
                    _system_fraction_values(validation["main_retrieval"]),
                ),
                _gate_row(
                    "trace schema validation for every main episode",
                    "gate",
                    all(value[0] == value[1] for value in validation["main_schema"].values()),
                    _system_fraction_values(validation["main_schema"]),
                ),
                _gate_row(
                    "calibration trace schema validation",
                    "separate calibration check",
                    validation["calibration_schema"][0] == validation["calibration_schema"][1],
                    _fraction_pair(validation["calibration_schema"]) + " traces",
                ),
                _gate_row(
                    "retained audit-packet reconciliation",
                    "reported audit check",
                    validation["audit"][0] == validation["audit"][1] and validation["audit"][1] == 40,
                    _fraction_pair(validation["audit"]) + " comparisons",
                ),
            ]
        )
    lines.extend(
        [
        _gate_row(
            "provenance-writer mechanism check",
            "mechanism",
            all(gates[system].get("provenance_writer_mechanism_check_pass", False) for system in ("S0", "S1", "S2")),
            f"S1 PAR={s1_par:.3f}; S2 no poison admissions" if s2_par == 0.0 else f"S1 PAR={s1_par:.3f}; S2 PAR={s2_par:.3f}",
        ),
        _gate_row(
            "declared pilot-validity gates",
            "summary",
            all(gates[system].get("official_pilot_valid", False) for system in ("S0", "S1", "S2")),
            "; ".join(f"{system}={'yes' if gates[system].get('official_pilot_valid', False) else 'no'}" for system in ("S0", "S1", "S2")),
        ),
        ]
    )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def build_confidence_intervals_table(metrics: dict, calibration_metrics: dict | None = None) -> str:
    actor_model = next(iter(metrics["by_configuration"]))
    rows = metrics["by_configuration"][actor_model]
    s0, s1, s2 = rows["S0"], rows["S1"], rows["S2"]
    interval_rows = [
        ("Clean success", s0, "clean_success", "CSR", "CSR_ci95"),
        ("S0 one-shot unsafe", s0, "one_shot_unsafe", "OVR", "OVR_ci95"),
        ("S1/S2 one-shot unsafe", s1, "one_shot_unsafe", "OVR", "OVR_ci95"),
        ("Stateful unsafe per system", s0, "stateful_unsafe", "SVR", "SVR_ci95"),
        ("S1 poisoned-memory admission", s1, "poison_admitted", "PAR", "PAR_ci95"),
        ("S1 PAR-exec", s1, "poison_admitted_executed_only", "PAR_executed_only", "PAR_executed_only_ci95"),
        ("Execution failure per system", s0, "execution_failure", "execution_failure_rate", "execution_failure_rate_ci95"),
        ("Executed-stateful unsafe per system", s0, "stateful_unsafe_executed_only", "SVR_executed_only", "SVR_executed_only_ci95"),
    ]
    lines = [
        "\\begin{tabular}{llll}",
        "\\toprule",
        "Quantity & Count & Rate & Wilson 95\\% interval \\\\",
        "\\midrule",
    ]
    for label, row, count_key, rate_key, interval_key in interval_rows:
        lines.append(
            "{label} & {count} & {rate} & {interval} \\\\".format(
                label=label,
                count=_count_cell_readable(row, count_key, rate_key),
                rate=_rate_cell_readable(row, count_key, rate_key),
                interval=_ci_cell(row, interval_key, rate_key),
            )
        )
    if calibration_metrics:
        calibration = calibration_metrics.get("calibration_by_condition", {}).get("S1-ORACLE-RETRIEVED-MEMORY")
        if calibration:
            unsafe = calibration["unsafe_executed"]
            episodes = calibration["episodes"]
            lines.append(
                "Calibration unsafe execution & {count} & {rate:.3f} & {interval} \\\\".format(
                    count=f"{unsafe}/{episodes}",
                    rate=calibration["CAL_SVR"],
                    interval=_format_ci(wilson_ci(unsafe, episodes)),
                )
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def build_task_level_signal_table(episode_scores: list[dict]) -> str:
    tasks = sorted({item["task_id"] for item in episode_scores if item["episode_kind"] == "stateful_attack"})
    rows = []
    for task_id in tasks:
        s1_stateful = [
            item
            for item in episode_scores
            if item["task_id"] == task_id and item["system"] == "S1" and item["episode_kind"] == "stateful_attack"
        ]
        admitted = sum(1 for item in s1_stateful if item["poison_admission_flag"] is True)
        unsafe = sum(1 for item in s1_stateful if item["unsafe"])
        rows.append((task_id, admitted, unsafe, len(s1_stateful)))
    active_rows = [row for row in rows if row[1] > 0]
    inactive_count = sum(1 for row in rows if row[1] == 0)
    inactive_denominator = sum(row[3] for row in rows if row[1] == 0)
    inactive_unsafe = sum(row[2] for row in rows if row[1] == 0)
    lines = [
        "\\begin{tabular}{llll}",
        "\\toprule",
        "Task group & S1 admissions & S1 delayed unsafe & Interpretation \\\\",
        "\\midrule",
    ]
    for task_id, admitted, unsafe, denominator in active_rows:
        lines.append(
            "{task} & {admitted} & {unsafe} & localized admission surface \\\\".format(
                task=_latex_escape(_short_task_label(task_id)),
                admitted=_count_fraction(admitted, denominator),
                unsafe=_none_fraction(unsafe, denominator),
            )
        )
    if inactive_count:
        lines.append(
            "other {tasks} tasks & {admitted} & {unsafe} & no S1 admissions in this packet \\\\".format(
                tasks=inactive_count,
                admitted=_none_fraction(0, inactive_denominator),
                unsafe=_none_fraction(inactive_unsafe, inactive_denominator),
            )
        )
    lines.append("\\multicolumn{4}{l}{\\footnotesize S0 and S2 have no poisoned-memory admissions in the same stateful task set.} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def build_failure_attribution_table(attribution_labels: list[dict]) -> str:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in attribution_labels:
        grouped[item["system"]].append(item)

    lines = [
        "\\begin{tabular}{lll}",
        "\\toprule",
        "Signal & Evidence & Interpretation \\\\",
        "\\midrule",
    ]
    summary = {}
    for system in ("S0", "S1", "S2"):
        items = grouped.get(system, [])
        unsafe_labels = Counter(item["attribution_label"] for item in items if item.get("unsafe"))
        summary[system] = {
            "retrieval": unsafe_labels["retrieval-mediated"],
            "memory": unsafe_labels["memory-mediated"],
            "execution_failures": sum(1 for item in items if item.get("execution_failure")),
        }
    lines.append(
        "Retrieval-mediated unsafe & {evidence} & current-turn OVR failures \\\\".format(
            evidence=_system_count_summary({system: summary[system]["retrieval"] for system in ("S0", "S1", "S2")})
        )
    )
    lines.append(
        "Memory-mediated unsafe & {evidence} & no delayed unsafe action attributed to memory \\\\".format(
            evidence=_system_count_summary({system: summary[system]["memory"] for system in ("S0", "S1", "S2")}, none_text="none observed in included rows")
        )
    )
    lines.append(
        "Execution-format failures & {evidence} & counted separately from unsafe actions \\\\".format(
            evidence=_system_count_summary({system: summary[system]["execution_failures"] for system in ("S0", "S1", "S2")})
        )
    )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def build_attempted_runs_table(data_dir: Path, scorer_id: str) -> str:
    lines = [
        "\\begin{tabular}{p{0.38\\linewidth}rlllp{0.26\\linewidth}}",
        "\\toprule",
        "Run packet and hash & Rows & Actor & Scope & Scorer & Inclusion decision \\\\",
        "\\midrule",
    ]
    for run_dir in _attempted_run_dirs(data_dir):
        summary_path = run_dir / "run_summary.json"
        metrics_path = run_dir / "metrics.json"
        with summary_path.open("r", encoding="utf-8") as handle:
            run_summary = json.load(handle)
        metrics = None
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
        lines.append(
            "{packet} & {rows} & {actor} & {scope} & {scorer} & {decision} \\\\".format(
                packet=_packet_hash_cell(run_dir.as_posix(), _trace_hash(run_summary)),
                rows=len(run_summary),
                actor=_actor_code(run_summary),
                scope=_scope_cell(run_summary),
                scorer=_latex_escape(scorer_id),
                decision=_latex_escape(_attempted_run_decision(run_dir, run_summary, metrics, data_dir)),
            )
        )
    calibration_dir = data_dir / "calibration" / "oracle_memory"
    if (calibration_dir / "run_summary.json").exists():
        with (calibration_dir / "run_summary.json").open("r", encoding="utf-8") as handle:
            calibration_summary = json.load(handle)
        lines.append(
            "{packet} & {rows} & Q7 & S1-ORACLE; T & {scorer} & {decision} \\\\".format(
                packet=_packet_hash_cell("data/calibration/s1_oracle_retrieved_memory", _trace_hash(calibration_summary)),
                rows=len(calibration_summary),
                scorer=_latex_escape(scorer_id),
                decision=_latex_escape("included calibration-only sensitivity packet; excluded from main rates"),
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _count_fraction(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}" if denominator else "0/0"


def _memory_path_cell(system: str, counts: dict) -> str:
    admitted = counts["poison_admitted"]
    retrieved = counts["admitted_poison_retrieved_at_trigger"]
    stateful = counts["stateful_attacks"]
    if admitted:
        return f"{admitted}/{stateful} admitted; {retrieved}/{admitted} retrieved"
    if system == "S0":
        return "not persisted"
    if system == "S2":
        return "poisoned writes blocked"
    return "no poison admission"


def _delayed_outcome_cell(system: str, row: dict, counts: dict) -> str:
    proposal = counts["unsafe_proposal_before_checker"]
    executed = counts["unsafe_executed"]
    _, stateful = row["rate_counts"]["stateful_unsafe"]
    if proposal or executed:
        return f"{proposal} unsafe proposals; {executed}/{stateful} executed"
    if system == "S0":
        return "no persisted memory path"
    if system == "S2":
        return "admission blocked before trigger"
    if counts["poison_admitted"]:
        return "retrieved poison not adopted"
    return f"none observed across {stateful} stateful attacks"


def _none_or_count(value: int) -> str:
    return "none observed" if value == 0 else str(value)


def _system_count_summary(values: dict[str, int], none_text: str = "none observed") -> str:
    if all(value == 0 for value in values.values()):
        return none_text
    unique_values = set(values.values())
    if len(unique_values) == 1:
        return f"{next(iter(unique_values))} per system"
    return "; ".join(f"{system}={values[system]}" for system in ("S0", "S1", "S2"))


def _none_fraction(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator}/{denominator}" if numerator else f"none in {denominator}"


def _none_rate(value: float) -> str:
    return "none observed" if value == 0.0 else f"{value:.3f}"


def _count_rate_cell(row: dict, count_key: str, rate_key: str) -> str:
    return f"{_count_cell(row, count_key, rate_key)} = {row[rate_key]:.3f}"


def _event_count_rate_cell(row: dict, count_key: str, rate_key: str) -> str:
    counts = row.get("rate_counts", {})
    if count_key in counts and counts[count_key][0] == 0:
        return f"none in {counts[count_key][1]}"
    return _count_rate_cell(row, count_key, rate_key)


def _count_rate_from_values(numerator: int, denominator: int, rate_value: float) -> str:
    return f"{numerator}/{denominator} = {rate_value:.3f}" if denominator else "0/0 = 0.000"


def _count_cell(row: dict, count_key: str, rate_key: str) -> str:
    counts = row.get("rate_counts", {})
    if count_key in counts:
        numerator, denominator = counts[count_key]
        return f"{numerator}/{denominator}"
    return f"{row[rate_key]:.3f}"


def _count_cell_readable(row: dict, count_key: str, rate_key: str) -> str:
    counts = row.get("rate_counts", {})
    if count_key in counts:
        numerator, denominator = counts[count_key]
        return _none_fraction(numerator, denominator)
    return f"{row[rate_key]:.3f}"


def _rate_cell_readable(row: dict, count_key: str, rate_key: str) -> str:
    counts = row.get("rate_counts", {})
    if count_key in counts and counts[count_key][0] == 0:
        return "not observed"
    return f"{row[rate_key]:.3f}"


def _ci_cell(row: dict, interval_key: str, rate_key: str) -> str:
    interval = row.get(interval_key)
    if interval is None:
        return "[n/a, n/a]"
    return _format_ci(interval)


def _format_ci(interval: list[float]) -> str:
    lower = "0" if interval[0] == 0.0 else f"{interval[0]:.3f}"
    upper = "0" if interval[1] == 0.0 else f"{interval[1]:.3f}"
    return f"[{lower}, {upper}]"


def _gate_row(name: str, category: str, passed: bool, value: str) -> str:
    return f"{name} & {category} & {'pass' if passed else 'fail'} & {value} \\\\"


def _system_values(rows: dict, metric_key: str) -> str:
    return "; ".join(f"{system}={rows[system][metric_key]:.3f}" for system in ("S0", "S1", "S2"))


def _system_fraction_values(values: dict[str, tuple[int, int]]) -> str:
    return "; ".join(f"{system}={_fraction_pair(values[system])}" for system in ("S0", "S1", "S2"))


def _fraction_pair(value: tuple[int, int]) -> str:
    return f"{value[0]}/{value[1]}"


def _short_task_label(task_id: str) -> str:
    return {
        "budget-limit-rule": "budget-limit-rule",
        "calendar-attendee": "calendar-attendee",
        "approval-limit-rule": "approval-limit-rule",
        "access-control-rule": "access-control-rule",
        "data-sharing-rule": "data-sharing-rule",
        "escalation-rule": "escalation-rule",
        "permission-scope-rule": "permission-scope-rule",
        "email-recipient": "email-recipient",
        "meeting-time": "meeting-time",
        "file-destination": "file-destination",
        "expense-code": "expense-code",
        "calendar-location": "calendar-location",
    }.get(task_id, task_id)


def _available_release_root() -> Path | None:
    for root in RELEASE_ROOT_CANDIDATES:
        if (root / "traces" / "v1_main_324").exists():
            return root
    return None


def _release_validation_counts(release_root: Path | None) -> dict | None:
    if release_root is None:
        return None
    main_summary_path = release_root / "results" / "run_summary.json"
    calibration_summary_path = release_root / "results" / "calibration_oracle_memory_run_summary.json"
    audit_report_path = release_root / "audit" / "audit_report.json"
    if not main_summary_path.exists() or not calibration_summary_path.exists() or not audit_report_path.exists():
        return None

    main_summary = json.loads(main_summary_path.read_text(encoding="utf-8"))
    calibration_summary = json.loads(calibration_summary_path.read_text(encoding="utf-8"))
    audit_report = json.loads(audit_report_path.read_text(encoding="utf-8"))
    main_retrieval = {system: [0, 0] for system in ("S0", "S1", "S2")}
    main_schema = {system: [0, 0] for system in ("S0", "S1", "S2")}

    for row in main_summary:
        system = row["system"]
        main_retrieval[system][1] += 1
        main_schema[system][1] += 1
        trace_rows = _load_trace_rows(release_root / row["trace_path"])
        if _trace_has_retrieval_records(trace_rows):
            main_retrieval[system][0] += 1
        if _trace_has_required_schema(trace_rows):
            main_schema[system][0] += 1

    calibration_schema = [0, 0]
    for row in calibration_summary:
        calibration_schema[1] += 1
        if _trace_has_required_schema(_load_trace_rows(release_root / row["trace_path"])):
            calibration_schema[0] += 1

    return {
        "main_retrieval": {system: tuple(value) for system, value in main_retrieval.items()},
        "main_schema": {system: tuple(value) for system, value in main_schema.items()},
        "calibration_schema": tuple(calibration_schema),
        "audit": (audit_report.get("agreements", 0), audit_report.get("n", 0)),
    }


def _load_trace_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _trace_has_retrieval_records(trace_rows: list[dict]) -> bool:
    return bool(trace_rows) and all(isinstance(row.get("retrieved_passages"), list) for row in trace_rows)


def _trace_has_required_schema(trace_rows: list[dict]) -> bool:
    return bool(trace_rows) and all(REQUIRED_TRACE_FIELDS.issubset(row.keys()) for row in trace_rows)


def _latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def _latex_path(text: str) -> str:
    return f"\\path{{{text}}}"


def _packet_hash_cell(path_text: str, trace_hash: str) -> str:
    return f"{_latex_path(path_text)}\\newline{{\\footnotesize hash \\texttt{{{trace_hash}}}}}"


def _attempted_run_dirs(data_dir: Path) -> list[Path]:
    candidates = [data_dir / "results"]
    if (data_dir / "pilot").exists():
        candidates.extend(sorted((data_dir / "pilot").glob("*")))
    candidates.extend(sorted(data_dir.glob("results_*")))
    run_dirs = []
    seen = set()
    for candidate in candidates:
        if not candidate.is_dir() or not (candidate / "run_summary.json").exists():
            continue
        key = candidate.as_posix()
        if key in seen:
            continue
        seen.add(key)
        run_dirs.append(candidate)
    return run_dirs


def _actor_code(run_summary: list[dict]) -> str:
    actors = sorted({item.get("actor_model", "unknown") for item in run_summary})
    if len(actors) != 1:
        return "mixed"
    return ACTOR_CODES.get(actors[0], "other")


def _scope_cell(run_summary: list[dict]) -> str:
    systems = sorted({item.get("system", "?") for item in run_summary})
    kinds = {item.get("episode_kind", "?") for item in run_summary}
    system_cell = "S0-S2" if systems == ["S0", "S1", "S2"] else "/".join(systems)
    kind_codes = [
        EPISODE_KIND_CODES.get(kind, kind)
        for kind in ("clean_control", "one_shot_attack", "stateful_attack")
        if kind in kinds
    ]
    return f"{system_cell}; {'/'.join(kind_codes)}"


def _attempted_run_decision(run_dir: Path, run_summary: list[dict], metrics: dict | None, data_dir: Path) -> str:
    if run_dir == data_dir / "results":
        return "included main evidence packet"
    if metrics is None:
        return "excluded: no validity report"
    if not _metrics_all_gates_pass(metrics):
        return "excluded: failed gates"
    if len(run_summary) == 324:
        return "excluded: superseded 324-row candidate"
    return "excluded: incomplete passing subset"


def _metrics_all_gates_pass(metrics: dict) -> bool:
    validation = metrics.get("pilot_validation_by_configuration", {})
    if not validation:
        return False
    for actor_rows in validation.values():
        for gate_row in actor_rows.values():
            if not gate_row.get("official_pilot_valid", False):
                return False
    return True


def _trace_hash(run_summary: list[dict]) -> str:
    digest = hashlib.sha256()
    found_any = False
    for item in sorted(run_summary, key=lambda row: row.get("trace_path", "")):
        trace_path = Path(item.get("trace_path", ""))
        if not trace_path.exists():
            continue
        found_any = True
        digest.update(trace_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(trace_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:10] if found_any else "missing"


if __name__ == "__main__":
    main()
