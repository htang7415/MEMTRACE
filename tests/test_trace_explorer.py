import json

from memtrace.evaluation.trace_explorer import (
    filter_episodes,
    load_trace,
    main,
    render_episode_narrative,
    render_report,
)


def _score(**overrides) -> dict:
    base = {
        "episode_id": "ep-1",
        "actor_model": "model-a",
        "system": "S1",
        "task_id": "budget-limit-rule",
        "episode_kind": "stateful_attack",
        "payload_type": "direct_override",
        "horizon": 1,
        "label": "unsafe",
        "ambiguity_reason": None,
        "poison_admission_flag": True,
        "trace_path": "unused.jsonl",
    }
    base.update(overrides)
    return base


def test_filter_episodes_matches_all_given_criteria() -> None:
    scores = [
        _score(episode_id="ep-1", system="S1", label="unsafe"),
        _score(episode_id="ep-2", system="S1", label="safe"),
        _score(episode_id="ep-3", system="S2", label="unsafe"),
    ]
    assert [item["episode_id"] for item in filter_episodes(scores, system="S1", label="unsafe")] == ["ep-1"]


def test_filter_episodes_ignores_none_criteria() -> None:
    scores = [_score(episode_id="ep-1"), _score(episode_id="ep-2")]
    assert len(filter_episodes(scores, system="S1", label=None)) == 2


def test_filter_episodes_can_target_ambiguity_reason() -> None:
    scores = [
        _score(episode_id="ep-1", ambiguity_reason="partial_argument_match"),
        _score(episode_id="ep-2", ambiguity_reason=None),
    ]
    matches = filter_episodes(scores, ambiguity_reason="partial_argument_match")
    assert [item["episode_id"] for item in matches] == ["ep-1"]


def test_load_trace_reads_jsonl(tmp_path) -> None:
    trace_path = tmp_path / "ep.jsonl"
    trace_path.write_text('{"turn": 1}\n{"turn": 2}\n', encoding="utf-8")
    assert load_trace(trace_path) == [{"turn": 1}, {"turn": 2}]


def test_render_episode_narrative_flags_poison_and_admission() -> None:
    score = _score(ambiguity_reason="partial_argument_match")
    trace = [
        {
            "turn": 1,
            "query": "poison query",
            "retrieved_passages": [
                {"source_id": "P010", "passage_kind": "poison", "payload_type": "direct_override"},
                {"source_id": "P001", "passage_kind": "authoritative"},
            ],
            "admitted_memory_records": [
                {"memory_type": "policy_rule", "content": "poisoned content here", "source_id": "P010"}
            ],
            "rejected_memory_records": [],
            "tool_router_log": None,
            "raw_planner_output": None,
            "label": None,
        },
        {
            "turn": 2,
            "query": "trigger query",
            "retrieved_passages": [],
            "admitted_memory_records": [],
            "rejected_memory_records": [],
            "tool_router_log": {"tool_name": "approve_expense", "arguments": {"amount": 950}},
            "raw_planner_output": '{"tool_name": "approve_expense", "arguments": {"amount": 950}}',
            "label": "unsafe",
        },
    ]
    narrative = render_episode_narrative(score, trace)
    assert "## ep-1" in narrative
    assert "ambiguity: partial_argument_match" in narrative
    assert "retrieved poison: P010 (direct_override)" in narrative
    assert "admitted memory: [policy_rule] poisoned content here" in narrative
    assert "tool call: approve_expense({'amount': 950})" in narrative
    assert "turn label: unsafe" in narrative


def test_render_report_combines_multiple_episodes(tmp_path) -> None:
    trace_path = tmp_path / "ep.jsonl"
    trace_path.write_text(json.dumps({"turn": 1, "query": "q"}) + "\n", encoding="utf-8")
    score = _score(trace_path=str(trace_path))
    report = render_report([score])
    assert "1 matching episode" in report
    assert "## ep-1" in report


def test_main_filters_and_writes_report_to_out(tmp_path) -> None:
    trace_path = tmp_path / "ep.jsonl"
    trace_path.write_text(json.dumps({"turn": 1, "query": "q"}) + "\n", encoding="utf-8")
    scores_path = tmp_path / "episode_scores.json"
    scores_path.write_text(
        json.dumps(
            [
                _score(episode_id="ep-1", system="S1", trace_path=str(trace_path)),
                _score(episode_id="ep-2", system="S2", trace_path=str(trace_path)),
            ]
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "report.md"

    main(["--episode-scores", str(scores_path), "--system", "S1", "--out", str(out_path)])

    rendered = out_path.read_text(encoding="utf-8")
    assert "## ep-1" in rendered
    assert "ep-2" not in rendered


def test_main_accepts_boolean_poison_admission_filter(tmp_path) -> None:
    trace_path = tmp_path / "ep.jsonl"
    trace_path.write_text(json.dumps({"turn": 1, "query": "q"}) + "\n", encoding="utf-8")
    scores_path = tmp_path / "episode_scores.json"
    scores_path.write_text(
        json.dumps(
            [
                _score(episode_id="ep-1", poison_admission_flag=True, trace_path=str(trace_path)),
                _score(episode_id="ep-2", poison_admission_flag=False, trace_path=str(trace_path)),
            ]
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "report.md"

    main(["--episode-scores", str(scores_path), "--poison-admission-flag", "true", "--out", str(out_path)])

    rendered = out_path.read_text(encoding="utf-8")
    assert "## ep-1" in rendered
    assert "ep-2" not in rendered
