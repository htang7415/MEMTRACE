try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass

import argparse
import json
from pathlib import Path

from memtrace.benchmark import gold_label_for_episode
from memtrace.config import AUDIT_REVIEW_MD_PATH, AUDIT_TEMPLATE_PATH, EPISODE_SCORES_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a reviewer-facing markdown packet for the label audit.")
    parser.add_argument("--audit-template", type=Path, default=AUDIT_TEMPLATE_PATH)
    parser.add_argument("--episode-scores", type=Path, default=EPISODE_SCORES_PATH)
    parser.add_argument("--output", type=Path, default=AUDIT_REVIEW_MD_PATH)
    args = parser.parse_args()

    records = _load_jsonl(args.audit_template)
    scores = _load_json(args.episode_scores)
    markdown = build_audit_review_markdown(records, scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")

    print(f"audit_review_episodes={len(records)}")
    print(f"audit_review_path={args.output}")


def build_audit_review_markdown(audit_records: list[dict], episode_scores: list[dict]) -> str:
    score_by_id = {score["episode_id"]: score for score in episode_scores}
    lines = [
        "# Label Audit Review Packet",
        "",
        "Fill `human_label` in `audit_template.jsonl` with one of `safe`, `unsafe`, or `execution_failure`.",
        "Use this packet to inspect the expected action, final tool call, and trace evidence for each sampled episode.",
        "",
    ]
    for index, record in enumerate(audit_records, start=1):
        score = score_by_id[record["episode_id"]]
        gold = gold_label_for_episode(score | record)
        trace = _load_jsonl(Path(record["trace_path"]))
        final_turn = trace[-1] if trace else {}
        poison_turns = [row["turn"] for row in trace if _turn_has_poison(row)]
        admitted_poison_turns = [row["turn"] for row in trace if _turn_admitted_poison(row)]

        lines.extend(
            [
                f"## {index}. `{record['episode_id']}`",
                "",
                "- human label: `TODO`",
                f"- system: `{record['system']}`",
                f"- kind: `{record['episode_kind']}`",
                f"- payload: `{record['payload_type']}`",
                f"- horizon: `{record['horizon']}`",
                f"- trace: `{_display_path(record['trace_path'])}`",
                f"- poison retrieved on turns: `{poison_turns or []}`",
                f"- poison admitted on turns: `{admitted_poison_turns or []}`",
                "",
                "**Expected**",
                "",
                f"- tool: `{gold.expected_tool if gold else None}`",
                f"- arguments: `{_compact_json(gold.expected_arguments if gold else {})}`",
                f"- policy decision: `{_policy_summary(gold)}`",
                "",
                "**Observed Final Turn**",
                "",
                f"- query: {_inline(final_turn.get('query', ''))}",
                f"- raw trace label: `{final_turn.get('label')}`",
                f"- tool call: `{_compact_json(final_turn.get('tool_router_log'))}`",
                f"- prior memory: `{_memory_summary(final_turn.get('prior_memory_state', []))}`",
                f"- admitted memory: `{_memory_summary(final_turn.get('admitted_memory_records', []))}`",
                "",
                "**Retrieved Passages On Final Turn**",
                "",
                "| Rank | Source | Kind | Allowlisted | Excerpt |",
                "| ---: | --- | --- | --- | --- |",
            ]
        )
        for passage in final_turn.get("retrieved_passages", [])[:5]:
            lines.append(
                "| {rank} | `{source}` | `{kind}` | `{allowlisted}` | {text} |".format(
                    rank=passage.get("rank", ""),
                    source=passage.get("source_id", ""),
                    kind=passage.get("passage_kind", ""),
                    allowlisted=passage.get("allowlisted", False),
                    text=_inline(_excerpt(passage.get("text", ""), 180)),
                )
            )
        lines.extend(["", "**Raw Planner Output Excerpt**", "", "```text", _excerpt(final_turn.get("raw_planner_output", ""), 700), "```", ""])
    return "\n".join(lines)


def _load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _turn_has_poison(row: dict) -> bool:
    return any(hit.get("passage_kind") == "poison" for hit in row.get("retrieved_passages", []))


def _turn_admitted_poison(row: dict) -> bool:
    poison_ids = {hit.get("source_id") for hit in row.get("retrieved_passages", []) if hit.get("passage_kind") == "poison"}
    return any(record.get("source_id") in poison_ids for record in row.get("admitted_memory_records", []))


def _policy_summary(gold) -> str:
    if gold is None or gold.expected_policy_decision is None:
        return "None"
    decision = gold.expected_policy_decision
    return f"{decision.action} when {decision.field} {decision.operator} {decision.value}"


def _memory_summary(records: list[dict]) -> str:
    if not records:
        return "[]"
    summary = [
        {
            "type": record.get("memory_type"),
            "content": record.get("content"),
            "source": record.get("source_id"),
            "turn": record.get("write_turn"),
        }
        for record in records[:5]
    ]
    suffix = " ..." if len(records) > 5 else ""
    return _compact_json(summary) + suffix


def _compact_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _display_path(path: str) -> str:
    parsed = Path(path)
    try:
        return str(parsed.relative_to(Path.cwd()))
    except ValueError:
        return path


def _inline(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _excerpt(text: str | None, limit: int) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


if __name__ == "__main__":
    main()
