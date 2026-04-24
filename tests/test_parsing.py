from memtrace.parsing import extract_json_payload


def test_extract_json_payload_finds_bare_object_inside_prose() -> None:
    raw_output = 'Here is the answer: {"tool_name":"write_note","arguments":{"destination":"ops","content":"note"}} Thanks.'
    assert extract_json_payload(raw_output) == '{"tool_name":"write_note","arguments":{"destination":"ops","content":"note"}}'


def test_extract_json_payload_finds_bare_list_inside_prose() -> None:
    raw_output = 'Result follows. [{"memory_type":"policy_rule","content":"Policy text.","source_id":"P001","source_kind":"retrieval"}] End.'
    assert extract_json_payload(raw_output) == '[{"memory_type":"policy_rule","content":"Policy text.","source_id":"P001","source_kind":"retrieval"}]'


def test_extract_json_payload_finds_bare_null_inside_prose() -> None:
    raw_output = "No tool should be called. null Final answer."
    assert extract_json_payload(raw_output) == "null"


def test_extract_json_payload_falls_back_before_non_json_code_fence() -> None:
    raw_output = '\n'.join(
        [
            '{"tool_name":"approve_expense","arguments":{"code":"ENG-450","amount":450.0,"approver":"team-manager"}}',
            "``` html",
            "<details>",
            "</details>",
            "```",
        ]
    )
    assert extract_json_payload(raw_output) == '{"tool_name":"approve_expense","arguments":{"code":"ENG-450","amount":450.0,"approver":"team-manager"}}'
