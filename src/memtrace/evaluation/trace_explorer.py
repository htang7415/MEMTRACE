"""Trace exploration for failure analysis beyond static tables/figures.

Filters episode-score rows by causal/label criteria, then renders a readable per-turn
narrative from the underlying retained trace -- retrieval, memory writer/admission,
planner proposal, and tool execution -- so a specific failure can be inspected without
hand-parsing JSONL trace files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from memtrace.config import EPISODE_SCORES_PATH


def filter_episodes(episode_scores: list[dict], **criteria: Any) -> list[dict]:
    """Select episode-score rows matching every given criterion.

    A criterion whose value is `None` is ignored (not filtered on), so callers can pass
    every possible filter unconditionally (as the CLI does) without excluding rows that
    simply don't set that field.
    """
    active = {key: value for key, value in criteria.items() if value is not None}
    return [item for item in episode_scores if all(item.get(key) == value for key, value in active.items())]


def load_trace(trace_path: str | Path) -> list[dict]:
    with Path(trace_path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def render_episode_narrative(score: dict, trace: list[dict]) -> str:
    ambiguity_suffix = f" (ambiguity: {score['ambiguity_reason']})" if score.get("ambiguity_reason") else ""
    lines = [
        f"## {score['episode_id']}",
        "",
        f"- system: {score.get('system')} · task: {score.get('task_id')} · kind: {score.get('episode_kind')} "
        f"· payload: {score.get('payload_type')} · horizon: {score.get('horizon')}",
        f"- label: {score.get('label')}{ambiguity_suffix}",
        f"- poison_admission_flag: {score.get('poison_admission_flag')}",
        "",
    ]
    for turn in trace:
        lines.extend(_render_turn(turn))
    return "\n".join(lines) + "\n"


def _render_turn(turn: dict) -> list[str]:
    lines = [f"### Turn {turn.get('turn')}: {turn.get('query')}", ""]

    poison_hits = [p for p in turn.get("retrieved_passages", []) if p.get("passage_kind") == "poison"]
    if poison_hits:
        lines.append(
            "- retrieved poison: " + ", ".join(f"{p['source_id']} ({p.get('payload_type')})" for p in poison_hits)
        )

    admitted = turn.get("admitted_memory_records") or []
    rejected = turn.get("rejected_memory_records") or []
    if admitted:
        lines.append(
            "- admitted memory: "
            + "; ".join(f"[{record['memory_type']}] {record['content'][:80]}" for record in admitted)
        )
    if rejected:
        lines.append(
            "- rejected memory: "
            + "; ".join(
                f"[{record['memory_type']}] {record['content'][:80]} ({record['validation_status']})"
                for record in rejected
            )
        )

    tool_call = turn.get("tool_router_log")
    if tool_call is not None:
        lines.append(f"- tool call: {tool_call['tool_name']}({tool_call['arguments']})")
    elif turn.get("raw_planner_output") is not None:
        lines.append(f"- planner output (no tool call): {turn['raw_planner_output'][:200]!r}")

    if turn.get("label") is not None:
        lines.append(f"- turn label: {turn['label']}")

    lines.append("")
    return lines


def render_report(matches: list[dict]) -> str:
    sections = [f"# Trace Exploration ({len(matches)} matching episode(s))", ""]
    for score in matches:
        trace = load_trace(score["trace_path"])
        sections.append(render_episode_narrative(score, trace))
    return "\n".join(sections)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Filter episodes and render a readable trace narrative for failure analysis."
    )
    parser.add_argument("--episode-scores", type=Path, default=EPISODE_SCORES_PATH)
    parser.add_argument("--actor-model")
    parser.add_argument("--system")
    parser.add_argument("--task-id")
    parser.add_argument("--episode-kind")
    parser.add_argument("--payload-type")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--label")
    parser.add_argument("--ambiguity-reason")
    parser.add_argument("--poison-admission-flag", type=_optional_bool, default=None, help="true|false")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, help="Write the rendered narrative here instead of stdout.")
    args = parser.parse_args(argv)

    with args.episode_scores.open("r", encoding="utf-8") as handle:
        episode_scores = json.load(handle)

    matches = filter_episodes(
        episode_scores,
        actor_model=args.actor_model,
        system=args.system,
        task_id=args.task_id,
        episode_kind=args.episode_kind,
        payload_type=args.payload_type,
        horizon=args.horizon,
        label=args.label,
        ambiguity_reason=args.ambiguity_reason,
        poison_admission_flag=args.poison_admission_flag,
    )
    if args.limit is not None:
        matches = matches[: args.limit]

    rendered = render_report(matches)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
        print(f"matched_episodes={len(matches)}")
        print(f"trace_explorer_report_path={args.out}")
    else:
        print(rendered)


def _optional_bool(value: str) -> bool:
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


if __name__ == "__main__":
    main()
