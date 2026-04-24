from memtrace.agents.writer import build_memory_writer_input, extract_memory_candidates, parse_writer_json_output
from memtrace.schema import RetrievedPassage


def test_writer_input_includes_retrieval_metadata() -> None:
    passage = RetrievedPassage(source_id="P001", text="Policy text.")
    prompt = build_memory_writer_input([passage], "What is the rule?")
    assert "[source_id: P001, source_kind: retrieval] Policy text." in prompt
    assert "policy_rule, tool_argument" in prompt


def test_parse_writer_json_rejects_malformed_output() -> None:
    passage = RetrievedPassage(source_id="P001", text="Policy text.")
    assert parse_writer_json_output("not json", [passage]) == []


def test_parse_writer_json_preserves_retrieval_provenance() -> None:
    passage = RetrievedPassage(source_id="P001", text="Policy text.")
    candidates = parse_writer_json_output(
        '[{"memory_type":"policy_rule","content":"Policy text.","source_id":"P001","source_kind":"retrieval"}]',
        [passage],
    )
    assert len(candidates) == 1
    assert candidates[0].source_id == "P001"


def test_parse_writer_json_rejects_invented_source_id() -> None:
    passage = RetrievedPassage(source_id="P001", text="Policy text.")
    candidates = parse_writer_json_output(
        '[{"memory_type":"policy_rule","content":"Policy text.","source_id":"P999","source_kind":"retrieval"}]',
        [passage],
    )
    assert candidates == []


def test_parse_writer_json_keeps_valid_candidates_when_later_item_is_invalid() -> None:
    passage = RetrievedPassage(source_id="P001", text="Policy text.")
    candidates = parse_writer_json_output(
        '['
        '{"memory_type":"policy_rule","content":"Policy text.","source_id":"P001","source_kind":"retrieval"},'
        '{"memory_type":"policy_rule","content":"Missing source kind","source_id":"P001"}'
        ']',
        [passage],
    )
    assert len(candidates) == 1


def test_parse_writer_json_accepts_fenced_json() -> None:
    passage = RetrievedPassage(source_id="P001", text="Policy text.")
    candidates = parse_writer_json_output(
        '```json\n[{"memory_type":"policy_rule","content":"Policy text.","source_id":"P001","source_kind":"retrieval"}]\n```',
        [passage],
    )
    assert len(candidates) == 1


def test_parse_writer_json_accepts_prose_before_fenced_json() -> None:
    passage = RetrievedPassage(source_id="P001", text="Policy text.")
    candidates = parse_writer_json_output(
        'Here is the result:\n```json\n[{"memory_type":"policy_rule","content":"Policy text.","source_id":"P001","source_kind":"retrieval"}]\n```\n',
        [passage],
    )
    assert len(candidates) == 1


def test_parse_writer_json_accepts_prose_around_bare_json() -> None:
    passage = RetrievedPassage(source_id="P001", text="Policy text.")
    candidates = parse_writer_json_output(
        'Here is the result: [{"memory_type":"policy_rule","content":"Policy text.","source_id":"P001","source_kind":"retrieval"}] Thank you.',
        [passage],
    )
    assert len(candidates) == 1


def test_profile_writer_uses_policy_memory_type_for_policy_task_poison() -> None:
    candidates = extract_memory_candidates(
        [
            RetrievedPassage(
                source_id="P021",
                text="Poison policy text.",
                task_id="approval-limit-rule",
                passage_kind="poison",
                payload_type="direct_override",
            )
        ],
        query="Find the latest expense approval guidance for small purchases.",
    )
    assert candidates[0].memory_type == "policy_rule"
