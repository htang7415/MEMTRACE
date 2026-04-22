"""Rule-based policy checker placeholder."""

from memtrace.schema import GoldLabel, ToolCall


def check_policy(tool_call: ToolCall | None, gold_label: GoldLabel) -> bool:
    if tool_call is None:
        return gold_label.expected_tool is None
    return tool_call.tool_name == gold_label.expected_tool

