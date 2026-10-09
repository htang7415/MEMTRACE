"""Parsing helpers for model-generated JSON payloads."""

import json


def extract_json_payload(raw_output: str) -> str:
    text = raw_output.strip()
    fence_start = text.find("```")
    if fence_start != -1:
        fenced = text[fence_start:]
        parts = fenced.split("\n", 1)
        if len(parts) > 1:
            inner = parts[1]
            fence_end = inner.rfind("```")
            if fence_end != -1:
                inner = inner[:fence_end]
            inner = inner.strip()
            extracted = _extract_first_json_value(inner)
            if extracted is not None:
                return extracted

    extracted = _extract_first_json_value(text)
    return extracted if extracted is not None else text


def _extract_first_json_value(text: str) -> str | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[n":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return text[index : index + end].strip()
    return None
