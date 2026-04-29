from memtrace.models.mlx_runner import _format_prompt, _strip_prompt_continuation


class DummyTokenizer:
    has_chat_template = True

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"CHAT:{messages[0]['content']}"


def test_format_prompt_uses_chat_template_when_available() -> None:
    assert _format_prompt(DummyTokenizer(), "Return JSON.") == "CHAT:Return JSON."


def test_strip_prompt_continuation_removes_leaked_chat_turn() -> None:
    assert _strip_prompt_continuation('{"tool_name":"x","arguments":{}}<|endoftext|>Human: next') == '{"tool_name":"x","arguments":{}}'
