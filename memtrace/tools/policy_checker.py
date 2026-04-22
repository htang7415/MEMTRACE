"""Rule-based non-blocking policy checker."""

from memtrace.schema import GoldLabel, ToolCall


def check_policy(tool_call: ToolCall | None, gold_label: GoldLabel) -> bool:
    if gold_label.expected_policy_decision is None:
        if tool_call is None:
            return gold_label.expected_tool is None
        return tool_call.tool_name == gold_label.expected_tool

    decision = gold_label.expected_policy_decision
    if decision.action == "reject":
        return tool_call is None
    if tool_call is None:
        return False
    return tool_call.tool_name == gold_label.expected_tool
