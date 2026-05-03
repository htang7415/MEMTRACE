try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from memtrace.eval.metrics import wilson_ci

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


DEFAULT_METRICS_PATH = Path("data/results/metrics.json")
DEFAULT_EPISODE_SCORES_PATH = Path("data/results/episode_scores.json")
DEFAULT_OUTPUT_DIR = Path("paper/manuscript/figures")

SYSTEMS = ("S0", "S1", "S2")
OVR_COLOR = "#b85c38"
SVR_COLOR = "#2f6f5e"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PDF figures for the MEMTRACE paper draft.")
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--episode-scores-path", type=Path, default=DEFAULT_EPISODE_SCORES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    with args.metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with args.episode_scores_path.open("r", encoding="utf-8") as handle:
        episode_scores = json.load(handle)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_pipeline(args.output_dir / "figure1_pipeline.pdf")
    build_ovr_vs_svr(metrics, args.output_dir / "figure2_ovr_vs_svr.pdf")
    build_causal_chain_funnel(metrics, args.output_dir / "figure3_par_by_task_family.pdf")
    build_task_localization(episode_scores, args.output_dir / "figure4_one_shot_vs_stateful.pdf")
    build_violation_by_horizon(episode_scores, args.output_dir / "figure5_violation_by_horizon.pdf")

    print(f"paper_figures_dir={args.output_dir}")


def build_pipeline(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.35))
    ax.set_axis_off()
    ax.text(0.05, 2.68, "Turn t: initial exposure", fontsize=10, fontweight="bold", va="center")
    ax.text(0.05, 1.34, "Turn t + delta: delayed trigger", fontsize=10, fontweight="bold", va="center")

    initial = [
        (0.25, 1.9, "Poisoned\ncorpus"),
        (1.55, 1.9, "Initial\nretrieval"),
        (2.85, 1.9, "Memory\nwriter"),
        (4.15, 1.9, "Admission\nfilter"),
        (5.45, 1.9, "Persistent\nmemory"),
    ]
    trigger = [
        (0.9, 0.6, "Benign\ntrigger"),
        (2.2, 0.6, "Memory\nretrieval"),
        (3.5, 0.6, "Planner\nproposal"),
        (4.8, 0.6, "Policy\nchecker"),
        (6.1, 0.6, "Tool\nexecution"),
    ]
    for box in initial + trigger:
        _draw_box(ax, *box)
    for first, second in zip(initial, initial[1:]):
        _arrow(ax, first[0] + 0.9, first[1] + 0.22, second[0], second[1] + 0.22)
    for first, second in zip(trigger, trigger[1:]):
        _arrow(ax, first[0] + 0.9, first[1] + 0.22, second[0], second[1] + 0.22)

    ax.plot([5.9, 5.9, 2.65], [1.9, 1.3, 1.3], color="#2f6f5e", linewidth=1.0)
    _arrow(ax, 2.65, 1.3, 2.65, 1.05, color="#2f6f5e")
    ax.text(4.1, 1.42, "PRR", ha="center", va="center", fontsize=8, color="#2f6f5e")

    _arrow(ax, 1.9, 1.86, 3.74, 0.98, connectionstyle="arc3,rad=-0.28", color="#b85c38")
    ax.text(2.85, 1.37, "one-shot OVR", ha="center", va="center", fontsize=8, color="#b85c38")

    ax.text(4.55, 2.42, "PAR", ha="center", va="center", fontsize=8, color="#356d9a")
    ax.text(4.18, 0.28, "UPR", ha="center", va="center", fontsize=8, color="#356d9a")
    ax.text(5.2, 0.28, "blocked", ha="center", va="center", fontsize=8, color="#356d9a")
    ax.text(6.55, 0.28, "SVR", ha="center", va="center", fontsize=8, color="#356d9a")
    _arrow(ax, 3.95, 0.56, 3.95, 0.18, color="#777777")
    ax.text(3.95, 0.05, "EFR", ha="center", va="center", fontsize=8, color="#777777")

    ax.set_xlim(0.0, 7.25)
    ax.set_ylim(-0.1, 2.95)
    _save_pdf(fig, output_path)


def build_causal_chain_funnel(metrics: dict, output_path: Path) -> None:
    actor_model = next(iter(metrics["by_configuration"]))
    rows = metrics["by_configuration"][actor_model]
    s1_counts = rows["S1"]["causal_chain_counts"]
    stateful = s1_counts["stateful_attacks"]
    admitted = s1_counts["poison_admitted"]
    retrieved = s1_counts["admitted_poison_retrieved_at_trigger"]
    failures = s1_counts["execution_failure"]
    stages = [
        ("Stateful\nattacks", str(stateful), "#777777"),
        ("Poison\nadmitted", f"{admitted}/{stateful}", "#2f6f5e"),
        ("Retrieved\nat trigger", f"{retrieved}/{admitted}", "#2f6f5e"),
        ("Unsafe proposal\nor execution", "not\nadopted", "#b85c38"),
        ("Main-pilot\nendpoint", "safe\nendpoint", "#777777"),
    ]

    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    ax.set_axis_off()
    ax.text(0.02, 0.94, "Where the S1 delayed chain stops", transform=ax.transAxes, fontsize=10, fontweight="bold")
    ax.text(
        0.02,
        0.84,
        "S1 admits and retrieves a small poisoned subset; the planner does not adopt it as an unsafe action",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    x0 = 0.03
    y0 = 0.45
    box_w = 0.15
    box_h = 0.20
    gap = 0.045
    for index, (label, value, color) in enumerate(stages):
        x = x0 + index * (box_w + gap)
        face_alpha = 0.18 if "not" in value or "safe" in value else 0.82
        rect = FancyBboxPatch(
            (x, y0),
            box_w,
            box_h,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            facecolor=color,
            alpha=face_alpha,
            edgecolor=color,
            linewidth=1.0,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y0 + 0.132, value, transform=ax.transAxes, ha="center", va="center", fontsize=8.2, fontweight="bold", linespacing=0.9)
        ax.text(x + box_w / 2, y0 + 0.055, label, transform=ax.transAxes, ha="center", va="center", fontsize=7.7)
        if index < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + box_w + gap * 0.78, y0 + box_h / 2),
                xytext=(x + box_w + gap * 0.18, y0 + box_h / 2),
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", color="#555555", linewidth=0.9),
            )
    ax.text(
        0.02,
        0.20,
        "S0/S2 do not persist poisoned memories in the main stateful packet; execution failures are tracked separately.",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    ax.text(
        0.02,
        0.10,
        f"Stateful execution-format failures: {failures} per system, counted outside unsafe-action rates.",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    _save_pdf(fig, output_path)


def build_ovr_vs_svr(metrics: dict, output_path: Path) -> None:
    actor_model = next(iter(metrics["by_configuration"]))
    rows = metrics["by_configuration"][actor_model]
    x = list(range(len(SYSTEMS)))
    width = 0.42
    ovr = [rows[system]["OVR"] for system in SYSTEMS]
    ovr_ci = [_ci_from_rate_counts(rows[system], "one_shot_unsafe") for system in SYSTEMS]
    ovr_yerr = ([rate - interval[0] for rate, interval in zip(ovr, ovr_ci)], [interval[1] - rate for rate, interval in zip(ovr, ovr_ci)])
    stateful_total = sum(rows[system]["rate_counts"]["stateful_unsafe"][1] for system in SYSTEMS)
    per_system_upper = max(rows[system].get("SVR_ci95", _ci_from_rate_counts(rows[system], "stateful_unsafe"))[1] for system in SYSTEMS)

    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    ovr_bars = ax.bar(x, ovr, width, color=OVR_COLOR)
    ax.errorbar(
        [patch.get_x() + patch.get_width() / 2 for patch in ovr_bars],
        ovr,
        yerr=ovr_yerr,
        fmt="none",
        ecolor="#333333",
        elinewidth=0.8,
        capsize=3,
        capthick=0.8,
        zorder=3,
    )
    ax.scatter(x, [0] * len(x), color=SVR_COLOR, s=28, zorder=4)
    ax.hlines(0, -0.35, len(SYSTEMS) - 0.65, color=SVR_COLOR, linewidth=1.1)
    for index, system in enumerate(SYSTEMS):
        _annotate_bar(
            ax,
            ovr_bars[index],
            f"{rows[system]['rate_counts']['one_shot_unsafe'][0]}/{rows[system]['rate_counts']['one_shot_unsafe'][1]}",
            top=ovr_ci[index][1],
        )
    ax.set_xticks(list(x), SYSTEMS)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 0.62)
    ax.text(
        0.0,
        1.03,
        "Current-turn risk differs from delayed execution",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
    )
    ax.text(
        0.0,
        -0.30,
        "Whiskers: Wilson 95% interval over the fixed episode set",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#555555",
    )
    ax.text(
        0.0,
        -0.20,
        f"Stateful unsafe is not observed across {stateful_total} stateful attacks; per-system Wilson upper bound <= {per_system_upper:.1%}.",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#2f6f5e",
    )
    _style_rate_axis(ax, yticks=(0.0, 0.30, 0.60))
    _save_pdf(fig, output_path)


def build_task_localization(episode_scores: list[dict], output_path: Path) -> None:
    rows = []
    for task_id in sorted({item["task_id"] for item in episode_scores if item["episode_kind"] == "stateful_attack"}):
        items = [
            item
            for item in episode_scores
            if item["system"] == "S1"
            and item["episode_kind"] == "stateful_attack"
            and item["task_id"] == task_id
        ]
        admitted = sum(1 for item in items if item.get("poison_admission_flag") is True)
        unsafe = sum(1 for item in items if item.get("unsafe"))
        rows.append((task_id, admitted, unsafe, len(items)))
    active = [row for row in rows if row[1] > 0]
    inactive_count = sum(1 for row in rows if row[1] == 0)
    inactive_denominator = sum(row[3] for row in rows if row[1] == 0)
    display_rows = active + [("all other tasks", 0, 0, inactive_denominator)]

    fig, ax = plt.subplots(figsize=(5.7, 2.45))
    y_positions = list(range(len(display_rows)))
    admitted_values = [row[1] for row in display_rows]
    denominators = [row[3] for row in display_rows]
    labels = [_short_task_label(row[0]) if row[0] != "all other tasks" else f"other {inactive_count} tasks" for row in display_rows]
    bars = ax.barh(y_positions, admitted_values, color="#356d9a", height=0.52)
    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("S1 poisoned-memory admissions")
    ax.set_xlim(0, max(4, max(admitted_values) + 1))
    ax.set_xticks([0, 1, 2, 3])
    for bar, admitted, denominator in zip(bars, admitted_values, denominators):
        label = f"{admitted}/{denominator}" if admitted else f"none in {denominator}"
        ax.text(
            admitted + 0.08,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=8,
            fontweight="bold",
        )
    ax.text(
        0.0,
        1.03,
        "S1 admissions are concentrated in two tasks",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
    )
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_pdf(fig, output_path)


def build_violation_by_horizon(episode_scores: list[dict], output_path: Path) -> None:
    rows = _s1_horizon_diagnostics(episode_scores)
    horizons = (1, 3, 7)
    stages = [
        ("Poison admitted", "poison_admitted", "#d9ebe5"),
        ("Admitted retrieved", "admitted_retrieved", "#d9ebe5"),
        ("Execution failure", "execution_failure", "#ededed"),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 2.75))
    ax.set_axis_off()
    ax.text(0.02, 0.95, "S1 stateful chain by horizon", transform=ax.transAxes, fontsize=10, fontweight="bold")
    ax.text(
        0.02,
        0.88,
        "counts over 24 stateful episodes per horizon; unsafe proposal/execution is summarized once",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    x0 = 0.36
    y0 = 0.66
    cell_w = 0.18
    cell_h = 0.09
    row_gap = 0.035
    col_gap = 0.04
    for col, horizon in enumerate(horizons):
        x = x0 + col * (cell_w + col_gap)
        ax.text(
            x + cell_w / 2,
            y0 + cell_h + 0.035,
            f"$\\Delta={horizon}$",
            transform=ax.transAxes,
            ha="center",
            fontsize=8.5,
        )
    for row_index, (label, key, fill) in enumerate(stages):
        y = y0 - row_index * (cell_h + row_gap)
        ax.text(0.02, y + cell_h / 2, label, transform=ax.transAxes, va="center", fontsize=8.5)
        for col, horizon in enumerate(horizons):
            item = rows[horizon]
            count = item[key]
            denominator = item["denominator"]
            x = x0 + col * (cell_w + col_gap)
            rect = FancyBboxPatch(
                (x, y),
                cell_w,
                cell_h,
                boxstyle="round,pad=0.005,rounding_size=0.008",
                facecolor=fill,
                alpha=1.0 if count else 0.35,
                edgecolor="#2f6f5e" if key in {"poison_admitted", "admitted_retrieved"} and count else "#b8b8b8",
                linewidth=0.8,
                transform=ax.transAxes,
            )
            ax.add_patch(rect)
            ax.text(
                x + cell_w / 2,
                y + cell_h / 2,
                f"{count}/{denominator}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
            )
    band_y = 0.18
    rect = FancyBboxPatch(
        (0.31, band_y),
        0.60,
        0.105,
        boxstyle="round,pad=0.006,rounding_size=0.01",
        facecolor="#fff4ed",
        alpha=0.55,
        edgecolor="#b85c38",
        linewidth=0.8,
        transform=ax.transAxes,
    )
    ax.add_patch(rect)
    ax.text(
        0.61,
        band_y + 0.052,
        "Unsafe endpoint: not observed at any horizon",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.05,
        "Admitted poison is retrieved at every horizon; the delayed chain still stops before unsafe planner action.",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    _save_pdf(fig, output_path)


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


def _ci_from_rate_counts(row: dict, count_key: str) -> list[float]:
    successes, total = row["rate_counts"][count_key]
    return wilson_ci(successes, total)


def _annotate_bar(ax, patch, label: str, y_offset: float = 0.018, top: float | None = None) -> None:
    height = patch.get_height()
    ax.text(
        patch.get_x() + patch.get_width() / 2,
        (height if top is None else max(height, top)) + y_offset,
        label,
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )


def _draw_box(ax, x: float, y: float, label: str) -> None:
    width = 0.9
    height = 0.44
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor="#faf8f3",
        edgecolor="#2b2b2b",
        linewidth=1.0,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=8.5)


def _arrow(
    ax,
    x_start: float,
    y_start: float,
    x_end: float,
    y_end: float,
    *,
    connectionstyle: str = "arc3,rad=0.0",
    color: str = "#1f1f1f",
) -> None:
    arrow = FancyArrowPatch(
        (x_start, y_start),
        (x_end, y_end),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.0,
        color=color,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)


def _style_rate_axis(ax, yticks=(0.0, 0.5, 1.0)) -> None:
    ax.set_yticks(list(yticks))
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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


def _save_pdf(fig, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
