from memtrace.schema import GoldLabel


def test_gold_label_defaults() -> None:
    record = GoldLabel(task_id="t1", expected_tool=None)
    assert record.expected_arguments == {}

