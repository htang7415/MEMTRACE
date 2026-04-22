"""Tool routing helpers."""

from memtrace.schema import ToolCall


def log_tool_call(tool_name: str, arguments: dict, turn: int) -> ToolCall:
    return ToolCall(tool_name=tool_name, arguments=arguments, turn=turn)

