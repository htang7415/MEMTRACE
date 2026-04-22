from memtrace.eval.metrics import rate


def test_rate_zero_denominator() -> None:
    assert rate(1, 0) == 0.0

