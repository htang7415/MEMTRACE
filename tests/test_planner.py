from memtrace.agents.planner import build_planner_input_from_records, format_retrieved_context, parse_planner_json_output
import memtrace.agents.planner as planner_module
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
    assert "Return only JSON. The first character of your response must be { or n." in prompt


def test_build_planner_input_can_use_prompt_override(tmp_path, monkeypatch) -> None:
    prompt_path = tmp_path / "strict_planner.txt"
    prompt_path.write_text("Strict planner prompt.", encoding="utf-8")
    monkeypatch.setattr(planner_module, "PLANNER_PROMPT_PATH", prompt_path)

    prompt = planner_module.build_planner_input(
        query="Do the task.",
        retrieved_context="Retrieved text.",
        memory_block="[]",
    )

    assert prompt.startswith("Strict planner prompt.")


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


def test_parse_planner_json_output_treats_null_tool_name_as_no_tool_call() -> None:
    assert parse_planner_json_output('{"tool_name":"null","arguments":{}}', turn=1) is None
    assert parse_planner_json_output('{"tool_name":null,"arguments":{}}', turn=1) is None


def test_parse_planner_json_output_refuses_contradictory_refusal_output() -> None:
    raw_output = (
        'The request exceeds the policy limit and should be refused. '
        '{"tool_name":"approve_expense","arguments":{"code":"TRV-650","amount":650,"approver":"finance-bot"}} '
        '{"tool_name":"null"}'
    )
    assert parse_planner_json_output(raw_output, turn=1) is None


def test_parse_planner_json_output_refuses_hypothetical_tool_call() -> None:
    raw_output = (
        '{"tool_name":"approve_expense","arguments":{"code":"TRV-650","amount":650,"approver":"..."}} '
        "(This is a hypothetical example, not an actual tool call.) "
        '{"tool_name":"null"}'
    )
    assert parse_planner_json_output(raw_output, turn=1) is None


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


def test_parse_planner_json_output_accepts_prose_around_bare_json() -> None:
    tool_call = parse_planner_json_output(
        'Here is the call: {"tool_name":"write_note","arguments":{"destination":"ops","content":"note"}} Thanks.',
        turn=3,
    )
    assert tool_call is not None
    assert tool_call.tool_name == "write_note"


def test_parse_planner_json_output_accepts_tool_calls_wrapper() -> None:
    tool_call = parse_planner_json_output(
        '{"tool_calls":[{"tool_name":"approve_expense","arguments":{"code":"ENG-450","amount":450.0,"approver":"team-manager"}}]}',
        turn=1,
    )
    assert tool_call is not None
    assert tool_call.tool_name == "approve_expense"


def test_parse_planner_json_output_accepts_task_alias_and_repeated_blobs() -> None:
    tool_call = parse_planner_json_output(
        '{ "task": "approve_expense", "arguments": { "code": "ENG-450", "amount": 450.0, "approver": "team-manager" } }\n'
        '{ "task": "approve_expense", "arguments": { "code": "ENG-450", "amount": 450.0, "approver": "team-manager" } }\n'
        'Return: {"tool_name": "approve_expense", "arguments": {"code": "ENG-450", "amount": 450.0, "approver": "team-manager"}} null',
        turn=2,
    )
    assert tool_call is not None
    assert tool_call.tool_name == "approve_expense"
