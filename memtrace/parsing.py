"""Parsing helpers for model-generated JSON payloads."""


def extract_json_payload(raw_output: str) -> str:
    text = raw_output.strip()
    fence_start = text.find("```")
    if fence_start == -1:
        return text

    fenced = text[fence_start:]
    parts = fenced.split("\n", 1)
    if len(parts) == 1:
        return ""
    inner = parts[1]
    fence_end = inner.rfind("```")
    if fence_end != -1:
        inner = inner[:fence_end]
    return inner.strip()
