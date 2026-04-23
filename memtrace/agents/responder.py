"""Responder stage for formatting natural-language replies."""

from __future__ import annotations

from memtrace.schema import ToolCall


def format_response(tool_call: ToolCall | None) -> str:
    if tool_call is None:
        return "I cannot comply with that request."
    return f"Prepared {tool_call.tool_name} with the planned arguments."
