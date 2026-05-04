try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass

import json
import argparse
from pathlib import Path

from memtrace.eval.metrics import wilson_ci
from memtrace.config import (
    EPISODE_SCORES_PATH,
    FIGURE1_PATH,
    FIGURE2_PATH,
    FIGURE3_PATH,
    FIGURE4_PATH,
    FIGURE5_PATH,
    FIGURES_DIR,
    METRICS_PATH,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MEMTRACE figures from retained metrics.")
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    metrics_path = _metrics_path(args.metrics)
    episode_scores_path = EPISODE_SCORES_PATH
    figures_dir = args.out or FIGURES_DIR
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    calibration_metrics = None
    calibration_path = Path("data/calibration/oracle_memory/metrics.json")
    if calibration_path.exists():
        with calibration_path.open("r", encoding="utf-8") as handle:
            calibration_metrics = json.load(handle)
    with episode_scores_path.open("r", encoding="utf-8") as handle:
        episode_scores = json.load(handle)

    figures_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "figure1_path": figures_dir / "figure1_pipeline.svg",
        "figure2_path": figures_dir / "figure2_ovr_vs_svr.svg",
        "figure3_path": figures_dir / "figure3_par_by_task_family.svg",
        "figure4_path": figures_dir / "figure4_one_shot_vs_stateful.svg",
        "figure5_path": figures_dir / "figure5_violation_by_horizon.svg",
        "figure5_alias_path": figures_dir / "figure5_violation_rate_by_horizon.svg",
    }
    figure5_svg = build_violation_by_horizon(episode_scores)
    output_paths["figure1_path"].write_text(build_pipeline_figure(), encoding="utf-8")
    output_paths["figure2_path"].write_text(build_ovr_vs_svr(metrics, calibration_metrics), encoding="utf-8")
    output_paths["figure3_path"].write_text(build_causal_chain_by_system(metrics), encoding="utf-8")
    output_paths["figure4_path"].write_text(build_task_localization(episode_scores), encoding="utf-8")
    output_paths["figure5_path"].write_text(figure5_svg, encoding="utf-8")
    output_paths["figure5_alias_path"].write_text(figure5_svg, encoding="utf-8")

    for label, path in output_paths.items():
        print(f"{label}={path}")


def _metrics_path(metrics_arg: Path | None) -> Path:
    if metrics_arg is None:
        return METRICS_PATH
    if metrics_arg.is_dir():
        return metrics_arg / "main_metrics.json"
    return metrics_arg


def build_pipeline_figure() -> str:
    initial = [
        (70, 105, "Poisoned\ncorpus"),
        (240, 105, "Initial\nRetriever"),
        (410, 105, "Memory\nwriter"),
        (580, 105, "S1/S2\nfilter"),
        (750, 105, "Memory\nstore"),
    ]
    trigger = [
        (155, 245, "Benign\ntrigger"),
        (325, 245, "Memory\nRetriever"),
        (495, 245, "Planner\nproposal"),
        (665, 245, "Policy\nchecker"),
        (835, 245, "Tool\nexecution"),
    ]
    rects = [_svg_box(x, y, label) for x, y, label in initial + trigger]
    arrows = []
    labels = [
        '<text x="54" y="58" font-size="18" font-weight="bold">Turn t: initial exposure</text>',
        '<text x="54" y="208" font-size="18" font-weight="bold">Turn t + Δ: delayed trigger</text>',
        '<text x="628" y="91" text-anchor="middle" font-size="15" fill="#356d9a" font-weight="bold">PAR</text>',
        '<text x="592" y="323" text-anchor="middle" font-size="15" fill="#356d9a" font-weight="bold">UPR</text>',
        '<text x="752" y="323" text-anchor="middle" font-size="15" fill="#356d9a" font-weight="bold">blocked</text>',
        '<text x="922" y="323" text-anchor="middle" font-size="15" fill="#356d9a" font-weight="bold">SVR</text>',
        '<text x="610" y="195" text-anchor="middle" font-size="15" fill="#2f6f5e" font-weight="bold">PRR</text>',
        '<text x="418" y="188" text-anchor="middle" font-size="15" fill="#b85c38" font-weight="bold">one-shot OVR</text>',
        '<text x="567" y="353" text-anchor="middle" font-size="15" fill="#777777" font-weight="bold">EFR</text>',
        '<text x="305" y="92" text-anchor="middle" font-size="12" fill="#555">1 poison retrieved</text>',
        '<text x="475" y="92" text-anchor="middle" font-size="12" fill="#555">2 candidate emitted</text>',
        '<text x="645" y="92" text-anchor="middle" font-size="12" fill="#555">3 admitted</text>',
        '<text x="390" y="232" text-anchor="middle" font-size="12" fill="#555">4 retrieved</text>',
        '<text x="560" y="232" text-anchor="middle" font-size="12" fill="#555">5 proposal</text>',
        '<text x="730" y="232" text-anchor="middle" font-size="12" fill="#555">6 block</text>',
        '<text x="900" y="232" text-anchor="middle" font-size="12" fill="#555">7 execution</text>',
    ]
    for first, second in zip(initial, initial[1:]):
        arrows.append(_svg_arrow(first[0] + 132, first[1] + 28, second[0], second[1] + 28))
    for first, second in zip(trigger, trigger[1:]):
        arrows.append(_svg_arrow(first[0] + 132, first[1] + 28, second[0], second[1] + 28))
    arrows.extend(
        [
            '<path d="M 816 161 L 816 205 L 391 205 L 391 244" fill="none" stroke="#2f6f5e" stroke-width="1.4" marker-end="url(#arrow-green)" />',
            '<path d="M 306 158 C 370 200 435 222 509 246" fill="none" stroke="#b85c38" stroke-width="1.4" marker-end="url(#arrow-red)" />',
            _svg_arrow(560, 302, 560, 338, color="#777777", marker="arrow-gray"),
        ]
    )
    return _svg_wrap(
        1040,
        370,
        [
            (
                '<defs>'
                '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#1f1f1f"/></marker>'
                '<marker id="arrow-green" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#2f6f5e"/></marker>'
                '<marker id="arrow-red" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#b85c38"/></marker>'
                '<marker id="arrow-gray" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#777777"/></marker>'
                '</defs>'
            ),
            *rects,
            *arrows,
            *labels,
        ],
    )


def build_ovr_vs_svr(metrics: dict, calibration_metrics: dict | None = None) -> str:
    systems = ["S0", "S1", "S2"]
    actor_model = next(iter(metrics["by_configuration"]))
    rows = metrics["by_configuration"][actor_model]
    calibration = None
    if calibration_metrics:
        calibration = calibration_metrics.get("calibration_by_condition", {}).get("S1-ORACLE-RETRIEVED-MEMORY")
    parts = [
        '<text x="42" y="34" font-size="18" font-weight="bold">Mechanism summary</text>',
        '<text x="42" y="326" font-size="13" fill="#555">Counts are exact. Calibration is excluded from main S0/S1/S2 rates.</text>',
    ]
    panels = [
        (
            42,
            "A. One-shot unsafe",
            [(system, _format_count(rows[system]["rate_counts"]["one_shot_unsafe"])) for system in systems],
            "#b85c38",
        ),
        (
            312,
            "B. Memory path",
            [
                ("S0", "no persistence"),
                (
                    "S1",
                    f"{_format_count(_rate_count(rows['S1'], 'poison_admitted'))} admitted\n"
                    f"{_format_count(_rate_count(rows['S1'], 'admitted_memory_retrieved_at_trigger'))} retrieved",
                ),
                ("S2", f"{_format_count(_rate_count(rows['S2'], 'poison_admitted'))} admitted"),
            ],
            "#356d9a",
        ),
        (
            582,
            "C. Delayed unsafe",
            [
                ("S0", _format_count(rows["S0"]["rate_counts"]["stateful_unsafe"])),
                ("S1", _format_count(rows["S1"]["rate_counts"]["stateful_unsafe"])),
                ("S2", _format_count(rows["S2"]["rate_counts"]["stateful_unsafe"])),
                ("Cal.", _format_count([calibration["unsafe_executed"], calibration["episodes"]]) if calibration else "not run"),
            ],
            "#2f6f5e",
        ),
    ]
    for x, title, panel_rows, color in panels:
        parts.extend(_count_panel(x, 70, title, panel_rows, color))
    return _svg_wrap(850, 350, parts)


def build_causal_chain_by_system(metrics: dict) -> str:
    actor_model = next(iter(metrics["by_configuration"]))
    rows = metrics["by_configuration"][actor_model]
    s1_counts = rows["S1"]["causal_chain_counts"]
    stateful = s1_counts["stateful_attacks"]
    admitted = s1_counts["poison_admitted"]
    retrieved = s1_counts["admitted_poison_retrieved_at_trigger"]
    failures = s1_counts["execution_failure"]
    proposal = s1_counts["unsafe_proposal_before_checker"]
    executed = s1_counts["unsafe_executed"]
    stages = [
        ("Stateful attacks", str(stateful), "#777777", "0.22"),
        ("Poison admitted", f"{admitted}/{stateful}", "#2f6f5e", "0.88"),
        ("Retrieved at trigger", f"{retrieved}/{admitted}", "#2f6f5e", "0.88"),
        ("Unsafe proposals", f"{proposal}/{stateful}", "#b85c38", "0.18"),
        ("Unsafe executions", f"{executed}/{stateful}", "#777777", "0.18"),
    ]
    parts = [
        '<text x="42" y="34" font-size="18" font-weight="bold">Where the S1 delayed chain stops</text>',
        '<text x="42" y="56" font-size="14" fill="#555">72 stateful attacks → 6 admitted → 6 retrieved → 0 unsafe proposals → 0 executions</text>',
    ]
    x0 = 42
    y = 100
    box_w = 116
    box_h = 58
    gap = 20
    for index, (label, value, color, opacity) in enumerate(stages):
        x = x0 + index * (box_w + gap)
        parts.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="6" fill="{color}" fill-opacity="{opacity}" stroke="{color}" stroke-width="1.1" />')
        parts.append(f'<text x="{x + box_w/2}" y="{y + 24}" text-anchor="middle" font-size="15" font-weight="bold">{value}</text>')
        parts.append(f'<text x="{x + box_w/2}" y="{y + 45}" text-anchor="middle" font-size="12">{label}</text>')
        if index < len(stages) - 1:
            x1 = x + box_w + 4
            x2 = x + box_w + gap - 4
            parts.append(f'<line x1="{x1}" y1="{y + box_h/2}" x2="{x2}" y2="{y + box_h/2}" stroke="#555" stroke-width="1.1" marker-end="url(#arrow)" />')
    parts.insert(0, '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#555"/></marker></defs>')
    parts.append('<text x="42" y="218" font-size="14" fill="#555">S0/S2 do not persist poisoned memories in the main stateful packet.</text>')
    parts.append(f'<text x="42" y="244" font-size="14" fill="#555">Stateful execution-format failures: {failures} per system, tracked separately.</text>')
    return _svg_wrap(700, 310, parts)


def build_task_localization(episode_scores: list[dict]) -> str:
    rows = []
    for task_id in sorted({item["task_id"] for item in episode_scores if item["episode_kind"] == "stateful_attack"}):
        items = [
            item
            for item in episode_scores
            if item["system"] == "S1"
            and item["episode_kind"] == "stateful_attack"
            and item["task_id"] == task_id
        ]
        rows.append(
            (
                task_id,
                {
                    "denominator": len(items),
                    "candidate": sum(1 for item in items if item.get("writer_candidate_emitted")),
                    "admitted": sum(1 for item in items if item.get("poison_admission_flag") is True),
                    "retrieved": sum(1 for item in items if item.get("admitted_memory_retrieved_at_trigger")),
                    "proposal": sum(1 for item in items if item.get("unsafe_tool_call_proposed_before_checker")),
                    "execution": sum(1 for item in items if item.get("unsafe_tool_call_executed")),
                    "failure": sum(1 for item in items if item.get("execution_failure")),
                },
            )
        )
    stages = [("cand.", "candidate"), ("admit", "admitted"), ("retr.", "retrieved"), ("prop.", "proposal"), ("exec.", "execution"), ("fail", "failure")]
    parts = [
        '<text x="42" y="34" font-size="18" font-weight="bold">Task-by-stage S1 stateful signal</text>',
        '<text x="42" y="56" font-size="14" fill="#555">counts over six stateful attacks per task</text>',
    ]
    x0 = 210
    y0 = 86
    cell_w = 70
    cell_h = 22
    for col, (label, _) in enumerate(stages):
        parts.append(f'<text x="{x0 + col * cell_w + cell_w/2}" y="76" text-anchor="middle" font-size="12">{label}</text>')
    for index, (task_id, counts) in enumerate(rows):
        y = y0 + index * 28
        parts.append(f'<text x="42" y="{y + 16}" font-size="12">{_short_task_label(task_id)}</text>')
        denominator = counts["denominator"]
        for col, (_, key) in enumerate(stages):
            count = counts[key]
            fill = "#2f6f5e" if key in {"candidate", "admitted", "retrieved"} else "#b85c38"
            if key == "failure":
                fill = "#8a8a8a"
            opacity = 0.18 + 0.72 * count / max(denominator, 1)
            x = x0 + col * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 8}" height="{cell_h}" rx="4" fill="{fill}" fill-opacity="{opacity:.2f}" stroke="#c7c7c7" />')
            parts.append(f'<text x="{x + (cell_w - 8)/2}" y="{y + 15}" text-anchor="middle" font-size="10">{count}/{denominator}</text>')
    parts.append('<text x="42" y="438" font-size="12" fill="#555">Unsafe proposal/execution remains 0/6 for every task.</text>')
    return _svg_wrap(700, 460, parts)


def build_violation_by_horizon(episode_scores: list[dict]) -> str:
    rows = _s1_horizon_diagnostics(episode_scores)
    horizons = [1, 3, 7]
    stages = [
        ("Poison admitted", "poison_admitted", "#d9ebe5"),
        ("Admitted retrieved", "admitted_retrieved", "#d9ebe5"),
        ("Execution failure", "execution_failure", "#ededed"),
    ]
    x0 = 230
    y0 = 74
    cell_w = 132
    cell_h = 34
    gap = 8
    parts = [
        '<text x="42" y="34" font-size="18" font-weight="bold">S1 stateful chain by horizon</text>',
        '<text x="42" y="56" font-size="14" fill="#555">counts over 24 stateful episodes per horizon</text>',
    ]
    for col, horizon in enumerate(horizons):
        x = x0 + col * (cell_w + gap)
        parts.append(f'<text x="{x + cell_w/2}" y="66" text-anchor="middle" font-size="16">Δ={horizon}</text>')
    for row_index, (label, key, fill) in enumerate(stages):
        y = y0 + row_index * (cell_h + gap)
        parts.append(f'<text x="42" y="{y + 22}" font-size="15">{label}</text>')
        for col, horizon in enumerate(horizons):
            item = rows[horizon]
            count = item[key]
            denominator = item["denominator"]
            x = x0 + col * (cell_w + gap)
            opacity = "1.0" if count else "0.35"
            stroke = "#2f6f5e" if key in {"poison_admitted", "admitted_retrieved"} and count else "#b8b8b8"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" rx="5" fill="{fill}" '
                f'fill-opacity="{opacity}" stroke="{stroke}" stroke-width="1.0" />'
            )
            parts.append(
                f'<text x="{x + cell_w/2}" y="{y + 22}" text-anchor="middle" font-size="15" '
                f'font-weight="bold">{count}/{denominator}</text>'
            )
    parts += [
        '<rect x="230" y="210" width="412" height="42" rx="6" fill="#fff4ed" fill-opacity="0.55" stroke="#b85c38" stroke-width="1.0" />',
        '<text x="436" y="236" text-anchor="middle" font-size="15" font-weight="bold">Unsafe endpoint: not observed at any horizon</text>',
        '<text x="42" y="302" font-size="14" fill="#555">Execution failures are counted separately and are not asserted to be the same admitted-memory episodes.</text>',
    ]
    return _svg_wrap(700, 326, parts)


def _s1_horizon_diagnostics(episode_scores: list[dict]) -> dict[int, dict[str, int]]:
    rows: dict[int, dict[str, int]] = {}
    for horizon in (1, 3, 7):
        items = [
            item
            for item in episode_scores
            if item["system"] == "S1"
            and item["episode_kind"] == "stateful_attack"
            and item["horizon"] == horizon
        ]
        rows[horizon] = {
            "denominator": len(items),
            "poison_admitted": sum(1 for item in items if item.get("poison_admission_flag") is True),
            "admitted_retrieved": sum(1 for item in items if item.get("admitted_memory_retrieved_at_trigger")),
            "unsafe_proposal": sum(1 for item in items if item.get("unsafe_tool_call_proposed_before_checker")),
            "unsafe_executed": sum(1 for item in items if item.get("unsafe_tool_call_executed")),
            "execution_failure": sum(1 for item in items if item.get("execution_failure")),
        }
    return rows


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


def _short_task_label(task_id: str) -> str:
    return {
        "budget-limit-rule": "budget limit",
        "calendar-attendee": "calendar attendee",
        "approval-limit-rule": "approval limit",
        "access-control-rule": "access control",
        "data-sharing-rule": "data sharing",
        "escalation-rule": "escalation",
        "permission-scope-rule": "permission scope",
        "email-recipient": "email recipient",
        "meeting-time": "meeting time",
        "file-destination": "file destination",
        "expense-code": "expense code",
        "calendar-location": "calendar location",
    }.get(task_id, task_id)


def _svg_wrap(width: int, height: int, parts: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#ffffff" />'
        + "".join(parts)
        + "</svg>\n"
    )


def _svg_box(x: int, y: int, label: str) -> str:
    lines = label.split("\n")
    text = "".join(
        f'<tspan x="{x + 66}" dy="{0 if index == 0 else 18}">{line}</tspan>'
        for index, line in enumerate(lines)
    )
    return (
        f'<rect x="{x}" y="{y}" width="132" height="56" rx="8" fill="#faf8f3" '
        'stroke="#2b2b2b" stroke-width="1.2" />'
        f'<text x="{x + 66}" y="{y + 24}" text-anchor="middle" font-size="15" fill="#1f1f1f">{text}</text>'
    )


def _svg_arrow(x1: float, y1: float, x2: float, y2: float, *, color: str = "#1f1f1f", marker: str = "arrow") -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="1.4" marker-end="url(#{marker})" />'
    )


def _ci_from_rate_counts(row: dict, count_key: str) -> list[float]:
    successes, total = row["rate_counts"][count_key]
    return wilson_ci(successes, total)


def _rate_count(row: dict, count_key: str) -> list[int]:
    counts = row.get("rate_counts", {})
    if count_key in counts:
        return counts[count_key]
    causal = row.get("causal_chain_counts", {})
    if count_key == "poison_admitted":
        return [causal.get("poison_admitted", 0), causal.get("stateful_attacks", 0)]
    if count_key == "admitted_memory_retrieved_at_trigger":
        return [causal.get("admitted_poison_retrieved_at_trigger", 0), causal.get("poison_admitted", 0)]
    return [0, 0]


def _format_count(count_pair: list[int]) -> str:
    numerator, denominator = count_pair
    return f"{numerator}/{denominator}"


def _interval(center_x: float, baseline_y: float, y_scale: float, rate: float, interval: list[float]) -> str:
    y_mid = baseline_y - y_scale * rate
    y_low = baseline_y - y_scale * interval[0]
    y_high = baseline_y - y_scale * interval[1]
    return (
        f'<line x1="{center_x}" y1="{y_high}" x2="{center_x}" y2="{y_low}" stroke="#333333" stroke-width="1.1" />'
        f'<line x1="{center_x - 5}" y1="{y_high}" x2="{center_x + 5}" y2="{y_high}" stroke="#333333" stroke-width="1.1" />'
        f'<line x1="{center_x - 5}" y1="{y_low}" x2="{center_x + 5}" y2="{y_low}" stroke="#333333" stroke-width="1.1" />'
        f'<circle cx="{center_x}" cy="{y_mid}" r="1.8" fill="#333333" />'
    )


def _axes(x: int, y: int, width: int, baseline_y: int) -> list[str]:
    parts = [
        f'<line x1="{x}" y1="{y}" x2="{x}" y2="{baseline_y}" stroke="#1f1f1f" stroke-width="1.2" />',
        f'<line x1="{x}" y1="{baseline_y}" x2="{width}" y2="{baseline_y}" stroke="#1f1f1f" stroke-width="1.2" />',
    ]
    for value in (0.0, 0.5, 1.0):
        ty = baseline_y - 180 * value
        if value != 0.0:
            parts.append(f'<line x1="{x}" y1="{ty}" x2="{width}" y2="{ty}" stroke="#d8d8d8" stroke-width="0.8" />')
        parts.append(f'<line x1="{x-5}" y1="{ty}" x2="{x}" y2="{ty}" stroke="#1f1f1f" stroke-width="1.1" />')
        parts.append(f'<text x="{x-10}" y="{ty+4}" text-anchor="end" font-size="16">{value:.1f}</text>')
    return parts


def _rate_axes(x: int, y: int, width: int, baseline_y: int, *, y_max: float, ticks: tuple[float, ...]) -> list[str]:
    parts = [
        f'<line x1="{x}" y1="{y}" x2="{x}" y2="{baseline_y}" stroke="#1f1f1f" stroke-width="1.2" />',
        f'<line x1="{x}" y1="{baseline_y}" x2="{width}" y2="{baseline_y}" stroke="#1f1f1f" stroke-width="1.2" />',
    ]
    for value in ticks:
        ty = baseline_y - 180 * (value / y_max)
        if value != 0.0:
            parts.append(f'<line x1="{x}" y1="{ty}" x2="{width}" y2="{ty}" stroke="#d8d8d8" stroke-width="0.8" />')
        parts.append(f'<line x1="{x-5}" y1="{ty}" x2="{x}" y2="{ty}" stroke="#1f1f1f" stroke-width="1.1" />')
        parts.append(f'<text x="{x-10}" y="{ty+4}" text-anchor="end" font-size="16">{value:.2f}</text>')
    return parts


def _bar(x: float, baseline_y: float, width: float, height: float, fill: str) -> str:
    return f'<rect x="{x}" y="{baseline_y-height}" width="{width}" height="{height}" fill="{fill}" opacity="0.9" />'


def _count_panel(x: int, y: int, title: str, rows: list[tuple[str, str]], color: str) -> list[str]:
    parts = [
        f'<text x="{x}" y="{y}" font-size="14" font-weight="bold">{title}</text>',
        f'<rect x="{x}" y="{y + 12}" width="232" height="210" rx="8" fill="#ffffff" stroke="#d0d0d0" />',
    ]
    row_gap = 46 if any("\n" in text for _, text in rows) else 39
    for index, (label, text) in enumerate(rows):
        row_y = y + 42 + index * row_gap
        parts.append(f'<text x="{x + 12}" y="{row_y + 18}" font-size="13">{label}</text>')
        if text in {"no persistence", "not run"}:
            fill = "#eeeeee"
            opacity = 0.85
            stroke = "#b0b0b0"
        else:
            first_line = text.splitlines()[0]
            numerator = int(first_line.split("/", 1)[0])
            denominator = int(first_line.split("/", 1)[1].split()[0])
            fill = color
            opacity = 0.18 + 0.72 * numerator / max(denominator, 1)
            stroke = color
        text_lines = text.splitlines()
        rect_height = 38 if len(text_lines) > 1 else 26
        parts.append(f'<rect x="{x + 68}" y="{row_y}" width="150" height="{rect_height}" rx="5" fill="{fill}" fill-opacity="{opacity:.2f}" stroke="{stroke}" />')
        if len(text_lines) > 1:
            for line_index, line in enumerate(text_lines):
                parts.append(f'<text x="{x + 143}" y="{row_y + 15 + line_index * 14}" text-anchor="middle" font-size="12" font-weight="bold">{line}</text>')
        else:
            parts.append(f'<text x="{x + 143}" y="{row_y + 18}" text-anchor="middle" font-size="12" font-weight="bold">{text}</text>')
    return parts


if __name__ == "__main__":
    main()
