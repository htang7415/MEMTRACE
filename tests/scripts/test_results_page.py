from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "results_page", Path(__file__).parents[2] / "scripts" / "results_page.py"
)
assert _spec and _spec.loader
results_page = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(results_page)


def test_fmt_shares_and_durations() -> None:
    assert results_page.fmt("prefix_cache_hit_ratio", 0.65) == "65.0%"
    assert results_page.fmt("duration_s", 718.171) == "718.2"  # "duration" contains "ratio"
    assert results_page.fmt("ttft_p95_ms", 4500.9) == "4,501"


def test_param_text_flattens_nested_settings() -> None:
    model = {"kind": "gemini", "params": {"model": "gemini-3.5-flash-lite", "reasoning_effort": "minimal"}}
    assert results_page.param_text(model) == "kind=gemini, model=gemini-3.5-flash-lite, reasoning_effort=minimal"
    assert results_page.param_text(8) == "8"


def test_cell_order_numbers_ascending_labels_as_first_seen() -> None:
    cells = [{"params": {"v": v, "c": c}} for v, c in (("b", 8), ("b", 1), ("a", 16), ("a", 2))]
    ordered = [(x["params"]["v"], x["params"]["c"]) for x in results_page.cell_order(cells, ["v", "c"])]
    assert ordered == [("b", 1), ("b", 8), ("a", 2), ("a", 16)]


def test_interval_by_repeats() -> None:
    three = {"mean": 2.0, "ci_low": 1.0, "ci_high": 3.0, "std": 1.0, "n": 3}
    assert results_page.interval("errors", three) == "2 [1, 3]"
    two = {"mean": 243.4, "ci_low": 0.0, "ci_high": 830.7, "std": 65.34, "n": 2}
    assert results_page.interval("recomputed_tokens_per_request", two) == "243.4 (197.2–289.6)"
    assert results_page.interval("errors", {**three, "n": 1}) == "2"
