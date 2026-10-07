import json

import pytest

from memtrace.serving.quality import check_call, load_cases, summarize, to_openai_tools


def _call(name, **arguments):
    return [{"function": {"name": name, "arguments": json.dumps(arguments)}}]


GT = [{"math.hypot": {"x": [3], "y": [4], "z": ["", 0], "unit": ["meters", "m"]}}]


@pytest.mark.parametrize(
    ("calls", "expected"),
    [
        (_call("math_hypot", x=3, y=4, unit="Meters"), (True, "correct")),
        (_call("math_hypot", x=3, y=4, z=0, unit="m"), (True, "correct")),
        (_call("math_hypot", x=3.0, y=4, unit="m"), (False, "wrong_value")),  # float given for an int answer
        (_call("math_hypot", x=3, unit="m"), (False, "missing_parameter")),
        (_call("math_hypot", x=3, y=4, unit="m", extra=1), (False, "unexpected_parameter")),
        (_call("math_pow", x=3, y=4, unit="m"), (False, "wrong_function")),
        ([], (False, "no_tool_call")),
        (_call("math_hypot", x=3, y=4, unit="m") * 2, (False, "wrong_call_count")),
        ([{"function": {"name": "math_hypot", "arguments": "{not json"}}], (False, "unparseable_arguments")),
    ],
)
def test_check_call_follows_bfcl_ast_rules(calls, expected) -> None:
    assert check_call(calls, GT) == expected


def test_check_call_matches_ints_for_floats_lists_and_nested_dicts() -> None:
    gt = [{"f": {"interval": [[1.0, 3.0]], "budget": [{"min": [300], "max": [400]}], "flag": [True]}}]

    assert check_call(_call("f", interval=[1, 3], budget={"min": 300, "max": 400}, flag=True), gt) == (True, "correct")
    assert check_call(_call("f", interval=[1, 3], budget={"min": 300}, flag=True), gt)[0] is False
    assert check_call(_call("f", interval=[1, 3], budget={"min": 300, "max": 400}, flag=1), gt)[0] is False


def test_to_openai_tools_sanitizes_names_and_maps_bfcl_types() -> None:
    (tool,) = to_openai_tools(
        [
            {
                "name": "geo.distance",
                "description": "d",
                "parameters": {
                    "type": "dict",
                    "properties": {"pt": {"type": "tuple", "items": {"type": "float"}}, "x": {"type": "any"}},
                },
            }
        ]
    )

    assert tool["function"]["name"] == "geo_distance"
    params = tool["function"]["parameters"]
    assert params["type"] == "object"
    assert params["properties"]["pt"] == {"type": "array", "items": {"type": "number"}}
    assert "type" not in params["properties"]["x"]


def test_load_cases_joins_answers_and_summary_reports_accuracy(tmp_path) -> None:
    (tmp_path / "possible_answer").mkdir()
    (tmp_path / "BFCL_v3_simple.json").write_text(
        json.dumps({"id": "simple_0", "question": [[{"role": "user", "content": "q"}]], "function": []}) + "\n"
    )
    (tmp_path / "possible_answer" / "BFCL_v3_simple.json").write_text(
        json.dumps({"id": "simple_0", "ground_truth": GT}) + "\n"
    )

    (case,) = load_cases(tmp_path, ("simple",))
    summary = summarize(
        [
            {"category": "simple", "correct": True, "reason": "correct"},
            {"category": "simple", "correct": False, "reason": "wrong_value"},
        ]
    )

    assert case["ground_truth"] == GT and case["category"] == "simple"
    assert summary["overall"]["accuracy"] == 0.5
    assert summary["reasons"] == {"correct": 1, "wrong_value": 1}
