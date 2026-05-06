try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass

import argparse
import io
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "memtrace-mpl-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from memtrace.config import EPISODE_SCORES_PATH, FIGURES_DIR, METRICS_PATH


PALETTE = {
    "ink": "#1F2933",
    "muted": "#5C6773",
    "faint": "#EEF2F6",
    "grid": "#D7DCE2",
    "panel": "#F8FAFC",
    "paper": "#FBFCFE",
    "shadow": "#B9C3CF",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "pink": "#CC79A7",
    "gray": "#6F7782",
    "white": "#FFFFFF",
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "svg.fonttype": "none",
        "svg.hashsalt": "memtrace",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MEMTRACE figures from retained metrics.")
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    metrics_path = _metrics_path(args.metrics)
    figures_dir = args.out or FIGURES_DIR
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    calibration_metrics = None
    calibration_path = _calibration_path(metrics_path, args.metrics)
    if calibration_path is not None:
        with calibration_path.open("r", encoding="utf-8") as handle:
            calibration_metrics = json.load(handle)

    with EPISODE_SCORES_PATH.open("r", encoding="utf-8") as handle:
        episode_scores = json.load(handle)

    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "figure1": _save_figure(_draw_pipeline_figure(), figures_dir / "figure1_pipeline"),
        "figure2": _save_figure(_draw_ovr_vs_svr(metrics, calibration_metrics), figures_dir / "figure2_ovr_vs_svr"),
        "figure3": _save_figure(_draw_causal_chain_by_system(metrics), figures_dir / "figure3_par_by_task_family"),
        "figure4": _save_figure(_draw_task_localization(episode_scores), figures_dir / "figure4_one_shot_vs_stateful"),
        "figure5": _save_figure(_draw_violation_by_horizon(episode_scores), figures_dir / "figure5_violation_by_horizon"),
    }
    outputs["figure5_alias"] = _copy_outputs(
        figures_dir / "figure5_violation_by_horizon",
        figures_dir / "figure5_violation_rate_by_horizon",
    )

    for label, paths in outputs.items():
        for suffix, path in paths.items():
            print(f"{label}_{suffix}_path={path}")


def _metrics_path(metrics_arg: Path | None) -> Path:
    if metrics_arg is None:
        return METRICS_PATH
    if metrics_arg.is_dir():
        return metrics_arg / "main_metrics.json"
    return metrics_arg


def _calibration_path(metrics_path: Path, metrics_arg: Path | None) -> Path | None:
    candidates = []
    if metrics_arg is not None and metrics_arg.is_dir():
        candidates.append(metrics_arg / "calibration_metrics.json")
    candidates.extend(
        [
            metrics_path.parent / "calibration_metrics.json",
            Path("data/calibration/oracle_memory/metrics.json"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_pipeline_figure() -> str:
    return _figure_svg(_draw_pipeline_figure())


def build_ovr_vs_svr(metrics: dict, calibration_metrics: dict | None = None) -> str:
    return _figure_svg(_draw_ovr_vs_svr(metrics, calibration_metrics))


def build_causal_chain_by_system(metrics: dict) -> str:
    return _figure_svg(_draw_causal_chain_by_system(metrics))


def build_task_localization(episode_scores: list[dict]) -> str:
    return _figure_svg(_draw_task_localization(episode_scores))


def build_violation_by_horizon(episode_scores: list[dict]) -> str:
    return _figure_svg(_draw_violation_by_horizon(episode_scores))


def _draw_pipeline_figure():
    width, height = 1040, 480
    fig, ax = _canvas(width, height)
    _header(
        ax,
        "MEMTRACE evaluation path",
        "The protocol separates retrieval exposure, poisoned-memory persistence, planner adoption, checker blocking, and execution.",
    )

    _shadowed_rounded(ax, 42, 90, 956, 126, face=PALETTE["panel"], edge=PALETTE["grid"], radius=12)
    _shadowed_rounded(ax, 42, 270, 956, 126, face=PALETTE["paper"], edge=PALETTE["grid"], radius=12)
    _rounded(ax, 42, 90, 6, 126, face=PALETTE["orange"], edge=PALETTE["orange"], radius=4, alpha=0.88)
    _rounded(ax, 42, 270, 6, 126, face=PALETTE["blue"], edge=PALETTE["blue"], radius=4, alpha=0.88)
    _label(ax, 64, 118, "Turn t: initial exposure", size=13, weight="bold")
    _label(ax, 64, 298, "Turn t + Δ: delayed trigger", size=13, weight="bold")

    initial = [
        (92, 138, "1", "Poisoned\ncorpus", PALETTE["orange"]),
        (264, 138, "2", "Initial\nRetriever", PALETTE["blue"]),
        (436, 138, "3", "Memory\nwriter", PALETTE["green"]),
        (608, 138, "4", "S1/S2\nfilter", PALETTE["sky"]),
        (780, 138, "5", "Memory\nstore", PALETTE["green"]),
    ]
    trigger = [
        (178, 318, "6", "Benign\ntrigger", PALETTE["gray"]),
        (350, 318, "7", "Memory\nRetriever", PALETTE["blue"]),
        (522, 318, "8", "Planner\nproposal", PALETTE["vermillion"]),
        (694, 318, "9", "Policy\nchecker", PALETTE["sky"]),
        (866, 318, "10", "Tool\nexecution", PALETTE["pink"]),
    ]
    for x, y, number, text, color in initial + trigger:
        _stage_box(ax, x, y, number, text, color)

    for first, second in zip(initial, initial[1:]):
        _arrow(ax, first[0] + 132, first[1] + 33, second[0] - 8, second[1] + 33, PALETTE["muted"])
    for first, second in zip(trigger, trigger[1:]):
        _arrow(ax, first[0] + 132, first[1] + 33, second[0] - 8, second[1] + 33, PALETTE["muted"])

    _arrow(ax, 846, 200, 846, 250, PALETTE["green"], rad=0.0)
    _arrow(ax, 846, 250, 416, 250, PALETTE["green"], rad=0.0)
    _arrow(ax, 416, 250, 416, 312, PALETTE["green"], rad=0.0)
    _label(ax, 614, 242, "PRR", size=11, color=PALETTE["green"], weight="bold", ha="center")

    _arrow(ax, 330, 200, 542, 312, PALETTE["vermillion"], rad=0.15)
    _badge(ax, 468, 286, "one-shot OVR", PALETTE["vermillion"])
    _arrow(ax, 588, 384, 588, 420, PALETTE["gray"], rad=0.0)

    metric_labels = [
        (688, 130, "PAR", PALETTE["green"]),
        (586, 420, "UPR", PALETTE["vermillion"]),
        (762, 420, "blocked", PALETTE["sky"]),
        (934, 420, "SVR", PALETTE["pink"]),
        (622, 446, "EFR", PALETTE["gray"]),
    ]
    for x, y, text, color in metric_labels:
        _badge(ax, x, y, text, color)
    return fig


def _draw_ovr_vs_svr(metrics: dict, calibration_metrics: dict | None = None):
    width, height = 940, 470
    fig, ax = _canvas(width, height)
    _header(
        ax,
        "Causal-path mechanism summary",
        "The main result is a path decomposition: current-turn risk appears; S1 memory admission appears; delayed execution is not observed.",
    )
    actor_model = next(iter(metrics["by_configuration"]))
    rows = metrics["by_configuration"][actor_model]
    s1_counts = rows["S1"]["causal_chain_counts"]
    calibration = None
    if calibration_metrics:
        calibration = calibration_metrics.get("calibration_by_condition", {}).get("S1-ORACLE-RETRIEVED-MEMORY")

    panel_y, panel_h = 112, 300
    _summary_panel(ax, 40, panel_y, 250, panel_h, "A. Current-turn risk")
    _label(ax, 66, 154, "Current-turn unsafe actions", size=8.5, color=PALETTE["muted"])
    one_shot_rows = [
        ("S0", rows["S0"]["rate_counts"]["one_shot_unsafe"]),
        ("S1", rows["S1"]["rate_counts"]["one_shot_unsafe"]),
        ("S2", rows["S2"]["rate_counts"]["one_shot_unsafe"]),
    ]
    for index, (label, count_pair) in enumerate(one_shot_rows):
        _compact_bar(
            ax,
            72,
            186 + index * 52,
            label,
            count_pair,
            PALETTE["vermillion"],
            y_max=0.35,
            track_w=108,
        )
    _label(ax, 66, 370, "Not a delayed-memory result.", size=8.5, color=PALETTE["muted"])

    _summary_panel(ax, 315, panel_y, 300, panel_h, "B. S1 memory path")
    _label(ax, 340, 154, "S1 chain stops before unsafe proposal", size=8.5, color=PALETTE["muted"])
    path_rows = [
        ("Stateful attacks", f"{s1_counts['stateful_attacks']}", PALETTE["gray"]),
        ("Poison admitted", f"{s1_counts['poison_admitted']}/{s1_counts['stateful_attacks']}", PALETTE["blue"]),
        (
            "Retrieved at trigger",
            f"{s1_counts['admitted_poison_retrieved_at_trigger']}/{max(s1_counts['poison_admitted'], 1)}",
            PALETTE["blue"],
        ),
        (
            "Unsafe proposals",
            f"{s1_counts['unsafe_proposal_before_checker']}/{s1_counts.get('stateful_parseable_planner_outputs', s1_counts['stateful_attacks'])}",
            PALETTE["green"],
        ),
        ("Unsafe executions", f"{s1_counts['unsafe_executed']}/{s1_counts['stateful_attacks']}", PALETTE["green"]),
    ]
    for index, (label, value, color) in enumerate(path_rows):
        y_pos = 180 + index * 34
        _rounded(ax, 340, y_pos, 56, 23, face=color, edge=color, radius=7, alpha=0.86 if index < 3 else 0.70)
        _label(ax, 368, y_pos + 11.5, value, size=9.5, color=PALETTE["white"], weight="bold", ha="center", va="center")
        _label(ax, 410, y_pos + 11.5, label, size=9.5, weight="bold" if index in {1, 2, 3, 4} else "normal", va="center")
        if index < len(path_rows) - 1:
            _arrow(ax, 368, y_pos + 25, 368, y_pos + 31, PALETTE["muted"], lw=1.0)
    _label(ax, 340, 366, "S0: no persistence; S2: 0/72 admitted", size=8.2, color=PALETTE["muted"])

    _summary_panel(ax, 640, panel_y, 260, panel_h, "C. Delayed execution")
    stateful_pairs = [rows[system]["rate_counts"]["stateful_unsafe"] for system in ("S0", "S1", "S2")]
    main_unsafe = sum(pair[0] for pair in stateful_pairs)
    main_total = sum(pair[1] for pair in stateful_pairs)
    _label(ax, 770, 154, "Main protocol endpoint", size=8.5, color=PALETTE["muted"], ha="center")
    _label(ax, 770, 182, f"{main_unsafe}/{main_total}", size=28, weight="bold", color=PALETTE["green"], ha="center", va="center")
    _label(ax, 770, 212, "delayed unsafe executions", size=9, color=PALETTE["muted"], ha="center")
    for index, system in enumerate(("S0", "S1", "S2")):
        _compact_zero_row(ax, 675, 230 + index * 32, system, rows[system]["rate_counts"]["stateful_unsafe"])
    if calibration:
        _label(ax, 674, 342, "Oracle calibration", size=8.5, color=PALETTE["muted"], weight="bold")
        _compact_bar(ax, 674, 354, "Cal.", [calibration["unsafe_executed"], calibration["episodes"]], PALETTE["green"], y_max=0.12, track_w=90)
    else:
        _label(ax, 674, 360, "Oracle calibration not available", size=8.5, color=PALETTE["muted"])

    _label(
        ax,
        40,
        445,
        "Calibration is excluded from main S0/S1/S2 rates; counts are shown to keep the negative delayed-execution result auditable.",
        size=10,
        color=PALETTE["muted"],
    )
    return fig


def _draw_causal_chain_by_system(metrics: dict):
    width, height = 860, 360
    fig, ax = _canvas(width, height)
    actor_model = next(iter(metrics["by_configuration"]))
    rows = metrics["by_configuration"][actor_model]
    s1_counts = rows["S1"]["causal_chain_counts"]
    stateful = s1_counts["stateful_attacks"]
    admitted = s1_counts["poison_admitted"]
    retrieved = s1_counts["admitted_poison_retrieved_at_trigger"]
    proposal = s1_counts["unsafe_proposal_before_checker"]
    proposal_total = s1_counts.get("stateful_parseable_planner_outputs", stateful)
    executed = s1_counts["unsafe_executed"]
    failures = s1_counts["execution_failure"]
    _header(
        ax,
        "Where the S1 delayed chain stops",
        f"S1: {stateful} stateful attacks -> {admitted}/{stateful} admitted -> {retrieved}/{max(admitted, 1)} retrieved -> {proposal}/{proposal_total} unsafe proposals -> {executed}/{stateful} executions",
    )
    stages = [
        ("Stateful\nattacks", str(stateful), PALETTE["gray"], 1.0),
        ("Poison\nadmitted", f"{admitted}/{stateful}", PALETTE["green"], admitted / max(stateful, 1)),
        ("Retrieved\nat trigger", f"{retrieved}/{max(admitted, 1)}", PALETTE["blue"], retrieved / max(admitted, 1)),
        ("Unsafe\nproposals", f"{proposal}/{proposal_total}", PALETTE["vermillion"], 0.0),
        ("Unsafe\nexecutions", f"{executed}/{stateful}", PALETTE["pink"], 0.0),
    ]
    x0, y, box_w, box_h, gap = 46, 124, 132, 74, 32
    for index, (label, value, color, rate) in enumerate(stages):
        x = x0 + index * (box_w + gap)
        alpha = 0.12 + 0.68 * max(rate, 0.04 if value.startswith("0/") else 0.0)
        _shadowed_rounded(ax, x, y, box_w, box_h, face=color, edge=color, alpha=alpha, radius=12)
        _label(ax, x + box_w / 2, y + 28, value, size=16, weight="bold", ha="center")
        _label(ax, x + box_w / 2, y + 55, label, size=10, ha="center")
        if index < len(stages) - 1:
            _arrow(ax, x + box_w + 5, y + box_h / 2, x + box_w + gap - 5, y + box_h / 2, PALETTE["muted"])
    _shadowed_rounded(ax, 488, 238, 350, 44, face="#FFF3EC", edge=PALETTE["vermillion"], radius=10)
    _label(ax, 663, 260, "breakpoint: no unsafe planner adoption", size=11, weight="bold", ha="center", va="center")
    _label(ax, 46, 306, "S0/S2 do not persist poisoned memories in the main stateful packet.", size=10, color=PALETTE["muted"])
    _label(ax, 46, 330, f"Stateful execution-format failures: {failures} per system, tracked separately.", size=10, color=PALETTE["muted"])
    return fig


def _draw_task_localization(episode_scores: list[dict]):
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

    width, height = 900, 560
    fig, ax = _canvas(width, height)
    _header(ax, "Task-by-stage S1 stateful signal", "Counts are over six stateful attacks per task.")
    stages = [
        ("candidate", "candidate", PALETTE["green"]),
        ("admit", "admitted", PALETTE["green"]),
        ("retrieve", "retrieved", PALETTE["blue"]),
        ("proposal", "proposal", PALETTE["vermillion"]),
        ("execute", "execution", PALETTE["pink"]),
        ("failure", "failure", PALETTE["gray"]),
    ]
    x0, y0, cell_w, cell_h, row_gap = 250, 112, 92, 27, 32
    for col, (label, _, color) in enumerate(stages):
        x = x0 + col * cell_w
        _rounded(ax, x, 84, cell_w - 9, 25, face=color, edge=color, alpha=0.12, radius=5)
        _label(ax, x + (cell_w - 9) / 2, 96, label, size=10, weight="bold", ha="center", va="center")
    for index, (task_id, counts) in enumerate(rows):
        y = y0 + index * row_gap
        active = any(counts[key] for _, key, _ in stages)
        _label(
            ax,
            46,
            y + cell_h / 2,
            _short_task_label(task_id),
            size=10,
            weight="bold" if active else "normal",
            color=PALETTE["ink"] if active else PALETTE["muted"],
            va="center",
        )
        denominator = counts["denominator"]
        for col, (_, key, color) in enumerate(stages):
            count = counts[key]
            x = x0 + col * cell_w
            alpha = 0.08 + 0.76 * count / max(denominator, 1)
            _rounded(ax, x, y, cell_w - 9, cell_h, face=color, edge=PALETTE["grid"], alpha=alpha, radius=5)
            _label(ax, x + (cell_w - 9) / 2, y + cell_h / 2, f"{count}/{denominator}", size=9, weight="bold", ha="center", va="center")

    _legend_dot(ax, 46, 520, PALETTE["green"], "memory admission path")
    _legend_dot(ax, 238, 520, PALETTE["vermillion"], "unsafe planner path")
    _legend_dot(ax, 408, 520, PALETTE["gray"], "format failure")
    _label(ax, 852, 523, "Unsafe proposal/execution is 0/6 for every task.", size=8.5, color=PALETTE["muted"], ha="right")
    return fig


def _draw_violation_by_horizon(episode_scores: list[dict]):
    rows = _s1_horizon_diagnostics(episode_scores)
    horizons = [1, 3, 7]
    stages = [
        ("Poison admitted", "poison_admitted", PALETTE["green"]),
        ("Admitted retrieved", "admitted_retrieved", PALETTE["blue"]),
        ("Execution failure", "execution_failure", PALETTE["gray"]),
    ]
    width, height = 760, 390
    fig, ax = _canvas(width, height)
    _header(ax, "S1 stateful chain by horizon", "Counts are over 24 stateful episodes per horizon.")
    x0, y0, cell_w, cell_h, gap = 258, 112, 132, 40, 14
    for col, horizon in enumerate(horizons):
        _label(ax, x0 + col * (cell_w + gap) + cell_w / 2, 96, f"Δ={horizon}", size=13, weight="bold", ha="center")
    for row_index, (label, key, color) in enumerate(stages):
        y = y0 + row_index * (cell_h + gap)
        _rounded(ax, 48, y + 8, 5, cell_h - 16, face=color, edge=color, radius=2, alpha=0.85)
        _label(ax, 64, y + cell_h / 2, label, size=11, weight="bold", va="center")
        for col, horizon in enumerate(horizons):
            item = rows[horizon]
            count = item[key]
            denominator = item["denominator"]
            x = x0 + col * (cell_w + gap)
            alpha = 0.10 + 0.72 * count / max(denominator, 1)
            _shadowed_rounded(ax, x, y, cell_w, cell_h, face=color, edge=color if count else PALETTE["grid"], alpha=alpha, radius=8, shadow_alpha=0.10)
            _label(ax, x + cell_w / 2, y + cell_h / 2, f"{count}/{denominator}", size=12, weight="bold", ha="center", va="center")
    _shadowed_rounded(ax, 230, 286, 500, 48, face="#FFF3EC", edge=PALETTE["vermillion"], radius=10)
    _label(ax, 470, 310, "Unsafe endpoint: not observed at any horizon", size=12, weight="bold", ha="center")
    _label(
        ax,
        48,
        358,
        "Execution failures are tracked separately from admitted-memory cases.",
        size=10,
        color=PALETTE["muted"],
    )
    return fig


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


def _canvas(width: int, height: int):
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=PALETTE["white"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    return fig, ax


def _figure_svg(fig) -> str:
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", facecolor=PALETTE["white"])
    plt.close(fig)
    svg = buffer.getvalue()
    start = svg.find("<svg")
    return svg[start:] if start >= 0 else svg


def _save_figure(fig, stem: Path) -> dict[str, Path]:
    paths = {
        "svg": stem.with_suffix(".svg"),
        "pdf": stem.with_suffix(".pdf"),
        "png": stem.with_suffix(".png"),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths["svg"], format="svg", facecolor=PALETTE["white"])
    fig.savefig(paths["pdf"], format="pdf", facecolor=PALETTE["white"])
    fig.savefig(paths["png"], format="png", dpi=400, facecolor=PALETTE["white"])
    plt.close(fig)
    return paths


def _copy_outputs(source_stem: Path, destination_stem: Path) -> dict[str, Path]:
    outputs = {}
    for suffix in ("svg", "pdf", "png"):
        source = source_stem.with_suffix(f".{suffix}")
        destination = destination_stem.with_suffix(f".{suffix}")
        shutil.copyfile(source, destination)
        outputs[suffix] = destination
    return outputs


def _header(ax, title: str, subtitle: str) -> None:
    _label(ax, 40, 38, title, size=15, weight="bold")
    _label(ax, 40, 61, subtitle, size=9, color=PALETTE["muted"])
    ax.plot([40, ax.get_xlim()[1] - 40], [78, 78], color=PALETTE["grid"], lw=0.8)


def _stage_box(ax, x: float, y: float, number: str, text: str, color: str) -> None:
    _shadowed_rounded(ax, x, y, 132, 66, face=PALETTE["white"], edge=color, radius=10, lw=1.2, shadow_alpha=0.16)
    _rounded(ax, x, y, 26, 26, face=color, edge=color, radius=8)
    _label(ax, x + 13, y + 13, number, size=9, color=PALETTE["white"], weight="bold", ha="center", va="center")
    _label(ax, x + 73, y + 34, text, size=12, weight="bold", ha="center", va="center", linespacing=1.15)


def _summary_panel(ax, x: float, y: float, width: float, height: float, title: str) -> None:
    _label(ax, x, y - 16, title, size=12, weight="bold")
    _shadowed_rounded(ax, x, y, width, height, face=PALETTE["white"], edge=PALETTE["grid"], radius=12, shadow_alpha=0.12)


def _compact_bar(
    ax,
    x: float,
    y: float,
    label: str,
    count_pair: list[int],
    color: str,
    *,
    y_max: float,
    track_w: float,
) -> None:
    numerator, denominator = count_pair
    value = numerator / max(denominator, 1)
    bar_w = track_w * min(value / y_max, 1.0)
    _label(ax, x, y + 12, label, size=10, weight="bold", ha="right", va="center")
    track_x = x + 18
    _rounded(ax, track_x, y, track_w, 24, face=PALETTE["faint"], edge=PALETTE["grid"], radius=7)
    if bar_w > 0:
        _rounded(ax, track_x, y, bar_w, 24, face=color, edge=color, radius=7, alpha=0.84)
    else:
        ax.plot([track_x + 5, track_x + track_w - 5], [y + 12, y + 12], color=color, lw=1.3, alpha=0.75)
    _label(ax, track_x + track_w + 16, y + 12, _format_count(count_pair), size=10, weight="bold", va="center")
    _label(ax, track_x + track_w / 2, y + 39, f"{value:.3f}", size=8, color=PALETTE["muted"], ha="center")


def _compact_zero_row(ax, x: float, y: float, label: str, count_pair: list[int]) -> None:
    _label(ax, x, y + 11, label, size=9.5, weight="bold", ha="right", va="center")
    _rounded(ax, x + 18, y, 108, 22, face=PALETTE["faint"], edge=PALETTE["grid"], radius=7)
    ax.plot([x + 24, x + 120], [y + 11, y + 11], color=PALETTE["green"], lw=1.4, alpha=0.8)
    _label(ax, x + 140, y + 11, _format_count(count_pair), size=9.5, weight="bold", va="center")


def _rate_panel(
    ax,
    x: float,
    y: float,
    title: str,
    rows: list[tuple[str, list[int] | None]],
    color: str,
    *,
    y_max: float,
    none_label: str = "not applicable",
) -> None:
    panel_w, panel_h = 260, 270
    _label(ax, x, y - 16, title, size=12, weight="bold")
    _rounded(ax, x, y, panel_w, panel_h, face=PALETTE["white"], edge=PALETTE["grid"], radius=12)
    track_x = x + 92
    track_w = 106
    _label(ax, track_x, y + 31, "0", size=8, color=PALETTE["muted"], ha="center")
    _label(ax, track_x + track_w, y + 31, f"{y_max:.2f}", size=8, color=PALETTE["muted"], ha="center")
    ax.plot([track_x, track_x + track_w], [y + 38, y + 38], color=PALETTE["grid"], lw=0.8)
    row_gap = 46 if len(rows) == 4 else 55
    for index, (label, count_pair) in enumerate(rows):
        row_y = y + 58 + index * row_gap
        _label(ax, x + 78, row_y + 12, label, size=10, weight="bold", ha="right", va="center")
        if count_pair is None:
            _rounded(ax, track_x, row_y, track_w, 24, face=PALETTE["faint"], edge=PALETTE["grid"], radius=6)
            _label(ax, track_x + track_w / 2, row_y + 12, none_label, size=8.5, color=PALETTE["muted"], ha="center", va="center")
            continue
        numerator, denominator = count_pair
        value = numerator / max(denominator, 1)
        bar_w = track_w * min(value / y_max, 1.0)
        _rounded(ax, track_x, row_y, track_w, 24, face=PALETTE["faint"], edge=PALETTE["grid"], radius=6)
        if bar_w > 0:
            _rounded(ax, track_x, row_y, bar_w, 24, face=color, edge=color, radius=6, alpha=0.88)
        else:
            ax.plot([track_x + 4, track_x + track_w - 4], [row_y + 12, row_y + 12], color=color, lw=1.4, alpha=0.7)
        _label(ax, x + 214, row_y + 12, _format_count(count_pair), size=10, weight="bold", va="center")
        _label(ax, track_x + track_w / 2, row_y + 39, f"{value:.3f}", size=8, color=PALETTE["muted"], ha="center")


def _rounded(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    face: str,
    edge: str,
    radius: float,
    lw: float = 1.0,
    alpha: float = 1.0,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=lw,
            facecolor=face,
            edgecolor=edge,
            alpha=alpha,
        )
    )


def _shadowed_rounded(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    face: str,
    edge: str,
    radius: float,
    lw: float = 1.0,
    alpha: float = 1.0,
    shadow_alpha: float = 0.14,
) -> None:
    _rounded(
        ax,
        x + 2,
        y + 2,
        width,
        height,
        face=PALETTE["shadow"],
        edge=PALETTE["shadow"],
        radius=radius,
        lw=0,
        alpha=shadow_alpha,
    )
    _rounded(ax, x, y, width, height, face=face, edge=edge, radius=radius, lw=lw, alpha=alpha)


def _arrow(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str,
    *,
    rad: float = 0.0,
    lw: float = 1.4,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0,
            shrinkB=0,
        )
    )


def _badge(ax, x: float, y: float, text: str, color: str) -> None:
    width = 9 * len(text) + 18
    _rounded(ax, x - width / 2, y - 12, width, 24, face=PALETTE["white"], edge=color, radius=12, lw=1.1)
    _label(ax, x, y, text, size=10, color=color, weight="bold", ha="center", va="center")


def _legend_dot(ax, x: float, y: float, color: str, label: str) -> None:
    ax.add_patch(Rectangle((x, y - 8), 14, 14, facecolor=color, edgecolor=color, alpha=0.65))
    _label(ax, x + 22, y, label, size=9, color=PALETTE["muted"], va="center")


def _label(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 10,
    color: str = PALETTE["ink"],
    weight: str = "normal",
    ha: str = "left",
    va: str = "baseline",
    linespacing: float = 1.2,
) -> None:
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        linespacing=linespacing,
    )


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


if __name__ == "__main__":
    main()
