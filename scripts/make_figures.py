try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass

import json
from collections import defaultdict

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
    with METRICS_PATH.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with EPISODE_SCORES_PATH.open("r", encoding="utf-8") as handle:
        episode_scores = json.load(handle)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE1_PATH.write_text(build_pipeline_figure(), encoding="utf-8")
    FIGURE2_PATH.write_text(build_ovr_vs_svr(metrics), encoding="utf-8")
    FIGURE3_PATH.write_text(build_par_by_task_family(episode_scores), encoding="utf-8")
    FIGURE4_PATH.write_text(build_one_shot_vs_stateful(metrics), encoding="utf-8")
    FIGURE5_PATH.write_text(build_violation_by_horizon(metrics), encoding="utf-8")

    print(f"figure1_path={FIGURE1_PATH}")
    print(f"figure2_path={FIGURE2_PATH}")
    print(f"figure3_path={FIGURE3_PATH}")
    print(f"figure4_path={FIGURE4_PATH}")
    print(f"figure5_path={FIGURE5_PATH}")


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
        rects.append(f'<rect x="{bx}" y="{y}" width="{width}" height="{height}" rx="10" fill="#f4f1ea" stroke="#2b2b2b" />')
        labels.append(f'<text x="{bx + width/2}" y="{y + 30}" text-anchor="middle" font-size="16" fill="#1f1f1f">{label}</text>')
        if i < len(boxes) - 1:
            x1 = bx + width
            x2 = bx + width + gap
            arrows.append(f'<line x1="{x1}" y1="{y + height/2}" x2="{x2}" y2="{y + height/2}" stroke="#1f1f1f" marker-end="url(#arrow)" />')
    return _svg_wrap(
        1160,
        220,
        [
            '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#1f1f1f"/></marker></defs>',
            '<text x="30" y="32" font-size="22" fill="#1f1f1f">Figure 1. MEMTRACE pipeline</text>',
            '<text x="30" y="56" font-size="14" fill="#555">Shared scaffold order used for every turn.</text>',
            *rects,
            *arrows,
            *labels,
        ],
    )


def build_ovr_vs_svr(metrics: dict) -> str:
    systems = ["S0", "S1", "S2"]
    averages = metrics["system_average"]
    colors = {"OVR": "#c76d3a", "SVR": "#2a7f62"}
    bars = []
    labels = ['<text x="30" y="32" font-size="22" fill="#1f1f1f">Figure 2. OVR vs SVR by system, averaged over actor models</text>']
    base_y = 250
    x = 80
    for i, system in enumerate(systems):
        ovr = averages[system]["OVR"]
        svr = averages[system]["SVR"]
        sx = x + i * 180
        bars.append(_bar(sx, base_y, 44, 180 * ovr, colors["OVR"]))
        bars.append(_bar(sx + 58, base_y, 44, 180 * svr, colors["SVR"]))
        labels.append(f'<text x="{sx+51}" y="{base_y+24}" text-anchor="middle" font-size="14">{system}</text>')
        labels.append(f'<text x="{sx+22}" y="{base_y - 180*ovr - 8}" text-anchor="middle" font-size="12">{ovr:.2f}</text>')
        labels.append(f'<text x="{sx+80}" y="{base_y - 180*svr - 8}" text-anchor="middle" font-size="12">{svr:.2f}</text>')
    legend = [
        '<rect x="520" y="22" width="14" height="14" fill="#c76d3a" />',
        '<text x="542" y="34" font-size="13">OVR</text>',
        '<rect x="590" y="22" width="14" height="14" fill="#2a7f62" />',
        '<text x="612" y="34" font-size="13">SVR</text>',
    ]
    axes = _axes(60, 60, 620, 250)
    return _svg_wrap(700, 320, labels + legend + axes + bars)


def build_par_by_task_family(episode_scores: list[dict]) -> str:
    grouped = defaultdict(list)
    for item in episode_scores:
        if item["episode_kind"] != "stateful_attack":
            continue
        grouped[(item["system"], item["family"])].append(item)
    systems = ["S0", "S1", "S2"]
    families = ["policy_memory", "tool_argument_memory"]
    colors = {"policy_memory": "#1f6aa5", "tool_argument_memory": "#9c3d54"}
    parts = ['<text x="30" y="32" font-size="22" fill="#1f1f1f">Figure 3. PAR by task family and system</text>']
    parts += _axes(60, 60, 620, 250)
    for i, system in enumerate(systems):
        sx = 90 + i * 180
        for j, family in enumerate(families):
            items = grouped[(system, family)]
            par = sum(1 for item in items if item["poison_admission_flag"] is True) / len(items)
            bx = sx + j * 54
            parts.append(_bar(bx, 250, 40, 180 * par, colors[family]))
            parts.append(f'<text x="{bx+20}" y="{250 - 180*par - 8}" text-anchor="middle" font-size="12">{par:.2f}</text>')
        parts.append(f'<text x="{sx+27}" y="274" text-anchor="middle" font-size="14">{system}</text>')
    parts += [
        '<rect x="410" y="22" width="14" height="14" fill="#1f6aa5" />',
        '<text x="432" y="34" font-size="13">policy_memory</text>',
        '<rect x="540" y="22" width="14" height="14" fill="#9c3d54" />',
        '<text x="562" y="34" font-size="13">tool_argument_memory</text>',
    ]
    return _svg_wrap(700, 320, parts)


def build_one_shot_vs_stateful(metrics: dict) -> str:
    actor_models = list(metrics["by_configuration"].keys())
    width = 760
    height = 340
    parts = ['<text x="30" y="32" font-size="22" fill="#1f1f1f">Figure 4. One-shot vs stateful ranking by actor model</text>']
    panel_x = [40, 390]
    for index, actor_model in enumerate(actor_models):
        x0 = panel_x[index]
        parts.extend(_axes(x0 + 30, 70, x0 + 290, 270))
        parts.append(f'<text x="{x0 + 30}" y="58" font-size="16">{_short_actor_model(actor_model)}</text>')
        for i, system in enumerate(("S1", "S2")):
            x = x0 + 90 + i * 110
            row = metrics["by_configuration"][actor_model][system]
            ovr = row["OVR"]
            svr = row["SVR"]
            parts.append(f'<circle cx="{x}" cy="{270 - 180*ovr}" r="7" fill="#c76d3a" />')
            parts.append(f'<circle cx="{x+42}" cy="{270 - 180*svr}" r="7" fill="#2a7f62" />')
            parts.append(f'<line x1="{x}" y1="{270 - 180*ovr}" x2="{x+42}" y2="{270 - 180*svr}" stroke="#444" />')
            parts.append(f'<text x="{x+21}" y="298" text-anchor="middle" font-size="14">{system}</text>')
        parts.append(f'<text x="{x0 + 210}" y="58" font-size="12">ranking reversal={metrics["ranking_reversal_by_actor_model"][actor_model]}</text>')
    parts += [
        '<rect x="560" y="18" width="14" height="14" fill="#c76d3a" />',
        '<text x="582" y="30" font-size="13">OVR</text>',
        '<rect x="620" y="18" width="14" height="14" fill="#2a7f62" />',
        '<text x="642" y="30" font-size="13">SVR</text>',
    ]
    return _svg_wrap(width, height, parts)


def build_violation_by_horizon(metrics: dict) -> str:
    systems = ["S0", "S1", "S2"]
    colors = {"S0": "#777", "S1": "#c76d3a", "S2": "#2a7f62"}
    x_map = {"1": 120, "3": 300, "7": 520}
    averages = metrics["system_average"]
    parts = ['<text x="30" y="32" font-size="22" fill="#1f1f1f">Figure 5. Violation rate by horizon</text>']
    parts += _axes(70, 60, 620, 250)
    for system in systems:
        pts = []
        dash = ' stroke-dasharray="6 6"' if system == "S0" else ""
        for horizon in ("1", "3", "7"):
            y = 250 - 180 * averages[system]["stateful_by_horizon"].get(horizon, 0.0)
            pts.append((x_map[horizon], y))
            parts.append(f'<circle cx="{x_map[horizon]}" cy="{y}" r="5" fill="{colors[system]}" />')
        parts.append(
            f'<polyline fill="none" stroke="{colors[system]}" stroke-width="2"{dash} points="{" ".join(f"{x},{y}" for x,y in pts)}" />'
        )
    for horizon, x in x_map.items():
        parts.append(f'<text x="{x}" y="278" text-anchor="middle" font-size="14">Δ={horizon}</text>')
    parts += [
        '<rect x="470" y="18" width="14" height="14" fill="#777" />',
        '<text x="492" y="30" font-size="13">S0</text>',
        '<rect x="530" y="18" width="14" height="14" fill="#c76d3a" />',
        '<text x="552" y="30" font-size="13">S1</text>',
        '<rect x="590" y="18" width="14" height="14" fill="#2a7f62" />',
        '<text x="612" y="30" font-size="13">S2</text>',
    ]
    return _svg_wrap(700, 320, parts)


def _short_actor_model(actor_model: str) -> str:
    if "Qwen2.5-3B" in actor_model:
        return "Qwen2.5-3B"
    if "Llama-3.2-3B" in actor_model:
        return "Llama-3.2-3B"
    return actor_model


def _svg_wrap(width: int, height: int, parts: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#fffdf8" />'
        + "".join(parts)
        + "</svg>\n"
    )


def _axes(x: int, y: int, width: int, baseline_y: int) -> list[str]:
    parts = [
        f'<line x1="{x}" y1="{y}" x2="{x}" y2="{baseline_y}" stroke="#1f1f1f" />',
        f'<line x1="{x}" y1="{baseline_y}" x2="{width}" y2="{baseline_y}" stroke="#1f1f1f" />',
    ]
    for value in (0.0, 0.5, 1.0):
        ty = baseline_y - 180 * value
        parts.append(f'<line x1="{x-5}" y1="{ty}" x2="{x}" y2="{ty}" stroke="#1f1f1f" />')
        parts.append(f'<text x="{x-10}" y="{ty+4}" text-anchor="end" font-size="12">{value:.1f}</text>')
    return parts


def _bar(x: float, baseline_y: float, width: float, height: float, fill: str) -> str:
    return f'<rect x="{x}" y="{baseline_y-height}" width="{width}" height="{height}" fill="{fill}" opacity="0.9" />'


if __name__ == "__main__":
    main()
