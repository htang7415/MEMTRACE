"""Rule-based non-blocking policy checker."""

from memtrace.memrisk.core.schema import GoldLabel, ToolCall
from memtrace.memrisk.evaluation.labeler import is_safe


def check_policy(tool_call: ToolCall | None, gold_label: GoldLabel) -> bool:
    return is_safe(tool_call, gold_label)
