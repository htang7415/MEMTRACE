"""Rule-based non-blocking policy checker."""

from memtrace.schema import GoldLabel, ToolCall
from memtrace.eval.labeler import is_safe


def check_policy(tool_call: ToolCall | None, gold_label: GoldLabel) -> bool:
    return is_safe(tool_call, gold_label)
