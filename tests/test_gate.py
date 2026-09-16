import json

import pytest

from memtrace.evaluation.gate import compare_metrics, main


def test_compare_metrics_reports_no_diff_for_identical_trees() -> None:
    baseline = {"S0": {"CSR": 0.5, "nested": {"a": 1}}}
    assert compare_metrics(baseline, baseline, tolerance=1e-9, path="by_configuration") == []


def test_compare_metrics_tolerates_small_float_noise() -> None:
    baseline = {"S0": {"CSR": 0.5}}
    candidate = {"S0": {"CSR": 0.5 + 1e-12}}
    assert compare_metrics(baseline, candidate, tolerance=1e-9, path="by_configuration") == []


def test_compare_metrics_flags_value_beyond_tolerance() -> None:
    baseline = {"S0": {"CSR": 0.5}}
    candidate = {"S0": {"CSR": 0.6}}
    diffs = compare_metrics(baseline, candidate, tolerance=1e-9, path="by_configuration")
    assert diffs == ["by_configuration.S0.CSR: baseline=0.5 candidate=0.6"]


def test_compare_metrics_flags_missing_and_added_keys() -> None:
    baseline = {"S0": {"CSR": 0.5}, "S1": {"CSR": 0.4}}
    candidate = {"S0": {"CSR": 0.5}, "S2": {"CSR": 0.3}}
    diffs = compare_metrics(baseline, candidate, tolerance=1e-9, path="by_configuration")
    assert "by_configuration.S1: missing from candidate (present in baseline)" in diffs
    assert "by_configuration.S2: added in candidate (not in baseline)" in diffs


def test_compare_metrics_flags_list_length_and_element_changes() -> None:
    baseline = {"S0": {"CSR_ci95": [0.1, 0.2]}}
    candidate = {"S0": {"CSR_ci95": [0.1, 0.3]}}
    diffs = compare_metrics(baseline, candidate, tolerance=1e-9, path="by_configuration")
    assert diffs == ["by_configuration.S0.CSR_ci95[1]: baseline=0.2 candidate=0.3"]

    candidate_short = {"S0": {"CSR_ci95": [0.1]}}
    diffs = compare_metrics(baseline, candidate_short, tolerance=1e-9, path="by_configuration")
    assert diffs == ["by_configuration.S0.CSR_ci95: length baseline=2 candidate=1"]


def test_main_exits_zero_when_metrics_match(tmp_path) -> None:
    metrics = {"by_configuration": {"S0": {"CSR": 0.5}}}
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(metrics), encoding="utf-8")
    candidate_path.write_text(json.dumps(metrics), encoding="utf-8")

    main(["--baseline", str(baseline_path), "--candidate", str(candidate_path)])


def test_main_exits_nonzero_when_metrics_regress(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps({"by_configuration": {"S0": {"CSR": 0.5}}}), encoding="utf-8")
    candidate_path.write_text(json.dumps({"by_configuration": {"S0": {"CSR": 0.9}}}), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        main(["--baseline", str(baseline_path), "--candidate", str(candidate_path)])
    assert excinfo.value.code == 1
