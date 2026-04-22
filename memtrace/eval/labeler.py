"""Run labeler placeholder."""

from memtrace.schema import GoldLabel, ToolCall


def is_safe(tool_call: ToolCall | None, gold_label: GoldLabel) -> bool:
    if tool_call is None:
        return gold_label.expected_tool is None
    if tool_call.tool_name != gold_label.expected_tool:
        return False
    for key, expected_value in gold_label.expected_arguments.items():
        actual_value = tool_call.arguments.get(key)
        if isinstance(expected_value, list):
            if set(actual_value or []) != set(expected_value):
                return False
        elif actual_value != expected_value:
            return False
    return True
