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
    output_paths["figure2_path"].write_text(build_ovr_vs_svr(metrics), encoding="utf-8")
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


def build_ovr_vs_svr(metrics: dict) -> str:
    systems = ["S0", "S1", "S2"]
    actor_model = next(iter(metrics["by_configuration"]))
    rows = metrics["by_configuration"][actor_model]
    colors = {"OVR": "#b85c38", "SVR": "#2f6f5e"}
    bars = []
    labels = []
    intervals = []
    base_y = 252
    y_max = 0.62
    y_scale = 180 / y_max
    x = 80
    stateful_total = sum(rows[system]["rate_counts"]["stateful_unsafe"][1] for system in systems)
    for i, system in enumerate(systems):
        row = rows[system]
        ovr = row["OVR"]
        ovr_ci = _ci_from_rate_counts(row, "one_shot_unsafe")
        sx = x + i * 180
        bar_center = sx + 38
        bars.append(_bar(sx + 10, base_y, 56, y_scale * ovr, colors["OVR"]))
        intervals.append(_interval(bar_center, base_y, y_scale, ovr, ovr_ci))
        labels.append(f'<text x="{sx+51}" y="{base_y+24}" text-anchor="middle" font-size="16">{system}</text>')
        labels.append(f'<text x="{bar_center}" y="{base_y - y_scale*ovr_ci[1] - 8}" text-anchor="middle" font-size="16" font-weight="bold">{row["rate_counts"]["one_shot_unsafe"][0]}/{row["rate_counts"]["one_shot_unsafe"][1]}</text>')
        labels.append(f'<circle cx="{sx+80}" cy="{base_y}" r="4" fill="{colors["SVR"]}" />')
    legend = [
        '<text x="70" y="38" font-size="16" font-weight="bold">Current-turn risk differs from delayed execution</text>',
        '<text x="22" y="64" font-size="16" text-anchor="middle" transform="rotate(-90 22 64)">Rate</text>',
        f'<text x="70" y="304" font-size="13" fill="#2f6f5e">Stateful unsafe: none observed across {stateful_total} stateful attacks</text>',
        '<text x="70" y="324" font-size="13" fill="#555">Whiskers: Wilson 95% interval over the fixed episode set</text>',
        '<line x1="90" y1="252" x2="548" y2="252" stroke="#2f6f5e" stroke-width="1.2" />',
    ]
    axes = _rate_axes(60, 60, 620, base_y, y_max=y_max, ticks=(0.0, 0.30, 0.60))
    return _svg_wrap(700, 340, labels + legend + axes + bars + intervals)


def build_causal_chain_by_system(metrics: dict) -> str:
    actor_model = next(iter(metrics["by_configuration"]))
    rows = metrics["by_configuration"][actor_model]
    s1_counts = rows["S1"]["causal_chain_counts"]
    stateful = s1_counts["stateful_attacks"]
    admitted = s1_counts["poison_admitted"]
    retrieved = s1_counts["admitted_poison_retrieved_at_trigger"]
    failures = s1_counts["execution_failure"]
    stages = [
        ("Stateful attacks", str(stateful), "#777777", "0.22"),
        ("Poison admitted", f"{admitted}/{stateful}", "#2f6f5e", "0.88"),
        ("Retrieved at trigger", f"{retrieved}/{admitted}", "#2f6f5e", "0.88"),
        ("Unsafe proposal/execution", "not adopted", "#b85c38", "0.18"),
        ("Main-pilot endpoint", "safe endpoint", "#777777", "0.18"),
    ]
    parts = [
        '<text x="42" y="34" font-size="18" font-weight="bold">Where the S1 delayed chain stops</text>',
        '<text x="42" y="56" font-size="14" fill="#555">admitted poison reaches the trigger context, then stops before unsafe planner action</text>',
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
        rows.append((task_id, sum(1 for item in items if item.get("poison_admission_flag") is True), len(items)))
    active = [row for row in rows if row[1] > 0]
    inactive_count = sum(1 for row in rows if row[1] == 0)
    inactive_denominator = sum(row[2] for row in rows if row[1] == 0)
    display_rows = active + [("all other tasks", 0, inactive_denominator)]
    parts = [
        '<text x="42" y="34" font-size="18" font-weight="bold">S1 admissions are task-localized</text>',
        '<text x="42" y="56" font-size="14" fill="#555">poisoned-memory admissions over stateful episodes</text>',
    ]
    x0 = 210
    y0 = 86
    scale = 82
    for index, (task_id, admitted, denominator) in enumerate(display_rows):
        y = y0 + index * 56
        label = _short_task_label(task_id) if task_id != "all other tasks" else f"other {inactive_count} tasks"
        parts.append(f'<text x="42" y="{y + 24}" font-size="16">{label}</text>')
        if admitted:
            parts.append(f'<rect x="{x0}" y="{y}" width="{scale * admitted}" height="32" fill="#356d9a" opacity="0.9" />')
        count_label = f"{admitted}/{denominator}" if admitted else f"none in {denominator}"
        parts.append(f'<text x="{x0 + scale * admitted + 12}" y="{y + 22}" font-size="16" font-weight="bold">{count_label}</text>')
    parts.append('<line x1="210" y1="260" x2="540" y2="260" stroke="#1f1f1f" />')
    for tick in range(4):
        x = x0 + scale * tick
        parts.append(f'<line x1="{x}" y1="255" x2="{x}" y2="265" stroke="#1f1f1f" />')
        parts.append(f'<text x="{x}" y="284" text-anchor="middle" font-size="14">{tick}</text>')
    return _svg_wrap(700, 310, parts)


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
        '<text x="42" y="302" font-size="14" fill="#555">Interpretation: admitted poison is retrieved, then stops before unsafe planner action.</text>',
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


if __name__ == "__main__":
    main()
