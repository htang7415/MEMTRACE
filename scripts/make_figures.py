try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass

import json
import argparse
from collections import defaultdict
from pathlib import Path

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
    }
    output_paths["figure1_path"].write_text(build_pipeline_figure(), encoding="utf-8")
    output_paths["figure2_path"].write_text(build_ovr_vs_svr(metrics), encoding="utf-8")
    output_paths["figure3_path"].write_text(build_par_by_task_family(episode_scores), encoding="utf-8")
    output_paths["figure4_path"].write_text(build_one_shot_vs_stateful(metrics), encoding="utf-8")
    output_paths["figure5_path"].write_text(build_violation_by_horizon(metrics), encoding="utf-8")

    for label, path in output_paths.items():
        print(f"{label}={path}")


def _metrics_path(metrics_arg: Path | None) -> Path:
    if metrics_arg is None:
        return METRICS_PATH
    if metrics_arg.is_dir():
        return metrics_arg / "main_metrics.json"
    return metrics_arg


def build_pipeline_figure() -> str:
    boxes = [
        "Retriever",
        "Memory Writer",
        "S1/S2 Filter",
        "Planner",
        "Tool Router",
        "Policy Checker",
        "Responder",
    ]
    x = 30
    y = 80
    width = 140
    height = 54
    gap = 18
    rects = []
    arrows = []
    labels = []
    for i, label in enumerate(boxes):
        bx = x + i * (width + gap)
        rects.append(f'<rect x="{bx}" y="{y}" width="{width}" height="{height}" rx="10" fill="#faf8f3" stroke="#2b2b2b" stroke-width="1.2" />')
        labels.append(f'<text x="{bx + width/2}" y="{y + 30}" text-anchor="middle" font-size="16" fill="#1f1f1f">{label}</text>')
        if i < len(boxes) - 1:
            x1 = bx + width
            x2 = bx + width + gap
            arrows.append(f'<line x1="{x1}" y1="{y + height/2}" x2="{x2}" y2="{y + height/2}" stroke="#1f1f1f" marker-end="url(#arrow)" />')
    return _svg_wrap(
        1160,
        190,
        [
            '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#1f1f1f"/></marker></defs>',
            *rects,
            *arrows,
            *labels,
        ],
    )


def build_ovr_vs_svr(metrics: dict) -> str:
    systems = ["S0", "S1", "S2"]
    averages = metrics["system_average"]
    colors = {"OVR": "#b85c38", "SVR": "#2f6f5e"}
    bars = []
    labels = []
    base_y = 252
    x = 80
    for i, system in enumerate(systems):
        ovr = averages[system]["OVR"]
        svr = averages[system]["SVR"]
        sx = x + i * 180
        bars.append(_bar(sx, base_y, 44, 180 * ovr, colors["OVR"]))
        bars.append(_bar(sx + 58, base_y, 44, 180 * svr, colors["SVR"]))
        labels.append(f'<text x="{sx+51}" y="{base_y+24}" text-anchor="middle" font-size="16">{system}</text>')
        labels.append(f'<text x="{sx+22}" y="{base_y - 180*ovr - 8}" text-anchor="middle" font-size="16">{ovr:.2f}</text>')
        labels.append(f'<text x="{sx+80}" y="{base_y - 180*svr - 8}" text-anchor="middle" font-size="16">{svr:.2f}</text>')
    legend = [
        '<rect x="510" y="26" width="12" height="12" fill="#b85c38" />',
        '<text x="540" y="37" font-size="16">OVR</text>',
        '<rect x="580" y="26" width="12" height="12" fill="#2f6f5e" />',
        '<text x="610" y="37" font-size="16">SVR</text>',
        '<text x="22" y="64" font-size="16" text-anchor="middle" transform="rotate(-90 22 64)">Rate</text>',
    ]
    axes = _axes(60, 60, 620, base_y)
    return _svg_wrap(700, 310, labels + legend + axes + bars)


def build_par_by_task_family(episode_scores: list[dict]) -> str:
    grouped = defaultdict(list)
    for item in episode_scores:
        if item["episode_kind"] != "stateful_attack":
            continue
        grouped[(item["system"], item["family"])].append(item)
    systems = ["S0", "S1", "S2"]
    families = ["policy_memory", "tool_argument_memory"]
    colors = {"policy_memory": "#356d9a", "tool_argument_memory": "#8a4156"}
    parts = []
    parts += _axes(60, 60, 620, 252)
    for i, system in enumerate(systems):
        sx = 90 + i * 180
        for j, family in enumerate(families):
            items = grouped[(system, family)]
            par = sum(1 for item in items if item["poison_admission_flag"] is True) / len(items)
            bx = sx + j * 54
            parts.append(_bar(bx, 252, 40, 180 * par, colors[family]))
            parts.append(f'<text x="{bx+20}" y="{252 - 180*par - 8}" text-anchor="middle" font-size="16">{par:.2f}</text>')
        parts.append(f'<text x="{sx+27}" y="274" text-anchor="middle" font-size="16">{system}</text>')
    parts += [
        '<rect x="392" y="26" width="12" height="12" fill="#356d9a" />',
        '<text x="412" y="37" font-size="16">Policy memory</text>',
        '<rect x="520" y="26" width="12" height="12" fill="#8a4156" />',
        '<text x="540" y="37" font-size="16">Tool-argument memory</text>',
        '<text x="22" y="64" font-size="16" text-anchor="middle" transform="rotate(-90 22 64)">PAR</text>',
    ]
    return _svg_wrap(700, 310, parts)


def build_one_shot_vs_stateful(metrics: dict) -> str:
    actor_models = list(metrics["by_configuration"].keys())
    width = 760
    height = 310
    parts = []
    panel_x = [40, 390]
    for index, actor_model in enumerate(actor_models):
        x0 = panel_x[index]
        parts.extend(_axes(x0 + 30, 70, x0 + 290, 252))
        parts.append(f'<text x="{x0 + 30}" y="56" font-size="16">{_short_actor_model(actor_model)}</text>')
        for i, system in enumerate(("S1", "S2")):
            x = x0 + 90 + i * 110
            row = metrics["by_configuration"][actor_model][system]
            ovr = row["OVR"]
            svr = row["SVR"]
            parts.append(f'<circle cx="{x}" cy="{252 - 180*ovr}" r="6" fill="#b85c38" />')
            parts.append(f'<circle cx="{x+42}" cy="{252 - 180*svr}" r="6" fill="#2f6f5e" />')
            parts.append(f'<line x1="{x}" y1="{252 - 180*ovr}" x2="{x+42}" y2="{252 - 180*svr}" stroke="#555" stroke-width="1.2" />')
            parts.append(f'<text x="{x+21}" y="280" text-anchor="middle" font-size="16">{system}</text>')
        parts.append(f'<text x="{x0 + 180}" y="56" font-size="16">reversal={metrics["ranking_reversal_by_actor_model"][actor_model]}</text>')
    parts += [
        '<rect x="562" y="22" width="12" height="12" fill="#b85c38" />',
        '<text x="582" y="33" font-size="16">OVR</text>',
        '<rect x="618" y="22" width="12" height="12" fill="#2f6f5e" />',
        '<text x="638" y="33" font-size="16">SVR</text>',
        '<text x="22" y="74" font-size="16" text-anchor="middle" transform="rotate(-90 22 74)">Rate</text>',
    ]
    return _svg_wrap(width, height, parts)


def build_violation_by_horizon(metrics: dict) -> str:
    systems = ["S0", "S1", "S2"]
    colors = {"S0": "#777", "S1": "#b85c38", "S2": "#2f6f5e"}
    x_map = {"1": 120, "3": 300, "7": 520}
    averages = metrics["system_average"]
    parts = []
    parts += _axes(70, 60, 620, 252)
    for system in systems:
        pts = []
        dash = ' stroke-dasharray="6 6"' if system == "S0" else ""
        for horizon in ("1", "3", "7"):
            y = 252 - 180 * averages[system]["stateful_by_horizon"].get(horizon, 0.0)
            pts.append((x_map[horizon], y))
            parts.append(f'<circle cx="{x_map[horizon]}" cy="{y}" r="5" fill="{colors[system]}" />')
        parts.append(
            f'<polyline fill="none" stroke="{colors[system]}" stroke-width="2.2"{dash} points="{" ".join(f"{x},{y}" for x,y in pts)}" />'
        )
    for horizon, x in x_map.items():
        parts.append(f'<text x="{x}" y="278" text-anchor="middle" font-size="16">Δ={horizon}</text>')
    parts += [
        '<rect x="480" y="22" width="12" height="12" fill="#777" />',
        '<text x="500" y="33" font-size="16">S0</text>',
        '<rect x="532" y="22" width="12" height="12" fill="#b85c38" />',
        '<text x="552" y="33" font-size="16">S1</text>',
        '<rect x="584" y="22" width="12" height="12" fill="#2f6f5e" />',
        '<text x="604" y="33" font-size="16">S2</text>',
        '<text x="24" y="74" font-size="16" text-anchor="middle" transform="rotate(-90 24 74)">SVR</text>',
    ]
    return _svg_wrap(700, 310, parts)


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


def _svg_wrap(width: int, height: int, parts: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#ffffff" />'
        + "".join(parts)
        + "</svg>\n"
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


def _bar(x: float, baseline_y: float, width: float, height: float, fill: str) -> str:
    return f'<rect x="{x}" y="{baseline_y-height}" width="{width}" height="{height}" fill="{fill}" opacity="0.9" />'


if __name__ == "__main__":
    main()
