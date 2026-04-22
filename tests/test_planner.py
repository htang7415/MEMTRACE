from memtrace.agents.planner import build_planner_input_from_records, format_retrieved_context, parse_planner_json_output
from memtrace.schema import MemoryRecord, RetrievedPassage


def test_format_retrieved_context_includes_source_and_rank() -> None:
    context = format_retrieved_context(
        [RetrievedPassage(source_id="P001", text="Policy text.", rank=2)]
    )
    assert "[source_id: P001, rank: 2] Policy text." == context


def test_build_planner_input_serializes_prior_memory() -> None:
    prompt = build_planner_input_from_records(
        query="Do the task.",
        retrieved_passages=[RetrievedPassage(source_id="P001", text="Policy text.", rank=1)],
        memory_records=[
            MemoryRecord(
                memory_type="policy_rule",
                content="Use the approved policy.",
                source_id="P001",
                source_kind="retrieval",
                trust_level="high",
                write_turn=1,
                validation_status="accepted",
            )
        ],
    )
    assert "Query: Do the task." in prompt
    assert "[policy_rule] Use the approved policy. (source: P001)" in prompt


def test_parse_planner_json_output_returns_tool_call() -> None:
    tool_call = parse_planner_json_output(
        '{"tool_name":"write_note","arguments":{"destination":"ops","content":"note"}}',
        turn=3,
    )
    assert tool_call is not None
    assert tool_call.tool_name == "write_note"
    assert tool_call.turn == 3


def test_parse_planner_json_output_treats_invalid_output_as_no_tool_call() -> None:
    assert parse_planner_json_output("not json", turn=1) is None


def test_parse_planner_json_output_accepts_fenced_json() -> None:
    tool_call = parse_planner_json_output(
        '```json\n{"tool_name":"write_note","arguments":{"destination":"ops","content":"note"}}\n```',
        turn=3,
    )
    assert tool_call is not None
    assert tool_call.tool_name == "write_note"


def test_parse_planner_json_output_accepts_prose_before_fenced_json() -> None:
    tool_call = parse_planner_json_output(
        'Here is the call:\n```json\n{"tool_name":"write_note","arguments":{"destination":"ops","content":"note"}}\n```',
        turn=3,
    )
    assert tool_call is not None
    assert tool_call.tool_name == "write_note"
