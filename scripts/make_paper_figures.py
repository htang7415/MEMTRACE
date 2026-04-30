try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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
    build_par_by_task_family(episode_scores, args.output_dir / "figure3_par_by_task_family.pdf")
    build_one_shot_vs_stateful(metrics, args.output_dir / "figure4_one_shot_vs_stateful.pdf")
    build_violation_by_horizon(metrics, args.output_dir / "figure5_violation_by_horizon.pdf")

    print(f"paper_figures_dir={args.output_dir}")


def build_pipeline(output_path: Path) -> None:
    labels = [
        "Retriever",
        "Memory\nWriter",
        "S1/S2\nFilter",
        "Planner",
        "Tool\nRouter",
        "Policy\nChecker",
        "Responder",
    ]
    fig, ax = plt.subplots(figsize=(10.5, 1.8))
    ax.set_axis_off()
    box_w = 1.25
    gap = 0.28
    y = 0.42
    for index, label in enumerate(labels):
        x = index * (box_w + gap)
        box = FancyBboxPatch(
            (x, y),
            box_w,
            0.42,
            boxstyle="round,pad=0.03,rounding_size=0.05",
            facecolor="#faf8f3",
            edgecolor="#2b2b2b",
            linewidth=1.0,
        )
        ax.add_patch(box)
        ax.text(x + box_w / 2, y + 0.21, label, ha="center", va="center", fontsize=9)
        if index < len(labels) - 1:
            arrow = FancyArrowPatch(
                (x + box_w + 0.03, y + 0.21),
                (x + box_w + gap - 0.03, y + 0.21),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.0,
                color="#1f1f1f",
            )
            ax.add_patch(arrow)
    ax.set_xlim(-0.12, len(labels) * (box_w + gap) - gap + 0.12)
    ax.set_ylim(0.15, 1.05)
    _save_pdf(fig, output_path)


def build_ovr_vs_svr(metrics: dict, output_path: Path) -> None:
    averages = metrics["system_average"]
    x = range(len(SYSTEMS))
    width = 0.34
    ovr = [averages[system]["OVR"] for system in SYSTEMS]
    svr = [averages[system]["SVR"] for system in SYSTEMS]

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.bar([item - width / 2 for item in x], ovr, width, label="OVR", color=OVR_COLOR)
    ax.bar([item + width / 2 for item in x], svr, width, label="SVR", color=SVR_COLOR)
    _label_bars(ax)
    ax.set_xticks(list(x), SYSTEMS)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, ncols=2, loc="upper right")
    _style_rate_axis(ax)
    _save_pdf(fig, output_path)


def build_par_by_task_family(episode_scores: list[dict], output_path: Path) -> None:
    grouped = defaultdict(list)
    for item in episode_scores:
        if item["episode_kind"] == "stateful_attack":
            grouped[(item["system"], item["family"])].append(item)

    families = ("policy_memory", "tool_argument_memory")
    labels = {"policy_memory": "Policy memory", "tool_argument_memory": "Tool-argument memory"}
    colors = {"policy_memory": "#356d9a", "tool_argument_memory": "#8a4156"}
    x = range(len(SYSTEMS))
    width = 0.34

    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    for family_index, family in enumerate(families):
        offset = -width / 2 if family_index == 0 else width / 2
        values = []
        for system in SYSTEMS:
            rows = grouped[(system, family)]
            values.append(sum(item["poison_admission_flag"] is True for item in rows) / len(rows))
        ax.bar([item + offset for item in x], values, width, label=labels[family], color=colors[family])
    _label_bars(ax)
    ax.set_xticks(list(x), SYSTEMS)
    ax.set_ylabel("PAR")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, ncols=1, loc="upper right")
    _style_rate_axis(ax)
    _save_pdf(fig, output_path)


def build_one_shot_vs_stateful(metrics: dict, output_path: Path) -> None:
    actor_model = next(iter(metrics["by_configuration"]))
    rows = metrics["by_configuration"][actor_model]
    systems = ("S1", "S2")

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for index, system in enumerate(systems):
        ovr = rows[system]["OVR"]
        svr = rows[system]["SVR"]
        ax.plot([index - 0.12, index + 0.12], [ovr, svr], color="#555555", linewidth=1.2)
        ax.scatter(index - 0.12, ovr, color=OVR_COLOR, s=36, label="OVR" if index == 0 else None)
        ax.scatter(index + 0.12, svr, color=SVR_COLOR, s=36, label="SVR" if index == 0 else None)
        ax.text(index - 0.12, ovr + 0.035, f"{ovr:.2f}", ha="center", fontsize=8)
        ax.text(index + 0.12, svr + 0.035, f"{svr:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(systems)), systems)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, ncols=2, loc="upper right")
    _style_rate_axis(ax)
    _save_pdf(fig, output_path)


def build_violation_by_horizon(metrics: dict, output_path: Path) -> None:
    averages = metrics["system_average"]
    horizons = ("1", "3", "7")
    colors = {"S0": "#777777", "S1": OVR_COLOR, "S2": SVR_COLOR}

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for system in SYSTEMS:
        values = [averages[system]["stateful_by_horizon"].get(horizon, 0.0) for horizon in horizons]
        linestyle = "--" if system == "S0" else "-"
        ax.plot(horizons, values, marker="o", label=system, color=colors[system], linestyle=linestyle)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("SVR")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, ncols=3, loc="upper right")
    _style_rate_axis(ax)
    _save_pdf(fig, output_path)


def _label_bars(ax) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + 0.025,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _style_rate_axis(ax) -> None:
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_pdf(fig, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
