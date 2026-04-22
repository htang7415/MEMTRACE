from memtrace.eval.labeler import is_safe
from memtrace.schema import GoldLabel, ToolCall


def test_labeler_matches_expected_tool() -> None:
    gold = GoldLabel(task_id="t1", expected_tool="send_email")
    tool = ToolCall(tool_name="send_email", arguments={}, turn=1)
    assert is_safe(tool, gold) is True


def test_labeler_checks_list_arguments_with_set_equality() -> None:
    gold = GoldLabel(
        task_id="t1",
        expected_tool="create_calendar_event",
        expected_arguments={"attendees": ["a@example.com", "b@example.com"]},
    )
    tool = ToolCall(
        tool_name="create_calendar_event",
        arguments={"attendees": ["b@example.com", "a@example.com"]},
        turn=1,
    )
    assert is_safe(tool, gold) is True
