"""BFCL single-call tool-calling accuracy as a task-level quality gate for engine changes.

Runs BFCL v3 `simple` and `multiple` cases through any OpenAI-compatible `/v1/chat/completions`
endpoint with the `tools` API and checks the returned call against BFCL's possible answers.

The checker follows BFCL's AST rules in simplified form: exactly one call; the right function;
every required parameter present; no unknown parameters; each value equal to one allowed value,
where strings compare case-insensitively ignoring spaces and `,./-_*^` punctuation, ints are
accepted for floats, and "" in an allowed list marks an optional parameter. Scores are therefore
close to, but not identical with, the official leaderboard.

    memtrace evaluate bfcl --base-url http://localhost:8200/v1 --model Qwen/Qwen3-0.6B --out-dir data/bfcl/vllm-metal
    memtrace evaluate bfcl ... --baseline data/bfcl/vllm-metal/summary.json --tolerance 0.03   # exit 1 on regression
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from memtrace.memrisk.evaluation.metrics import wilson_ci

CATEGORIES = ("simple", "multiple")
_TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array"}
_STRIP = re.compile(r"[ ,./\-_*^]")


def load_cases(data_dir: Path, categories: tuple[str, ...]) -> list[dict[str, Any]]:
    cases = []
    for category in categories:
        questions = (data_dir / f"BFCL_v3_{category}.json").read_text().splitlines()
        answers = (data_dir / "possible_answer" / f"BFCL_v3_{category}.json").read_text().splitlines()
        answer_by_id = {entry["id"]: entry["ground_truth"] for entry in map(json.loads, answers)}
        for entry in map(json.loads, questions):
            cases.append({**entry, "category": category, "ground_truth": answer_by_id[entry["id"]]})
    return cases


def to_openai_tools(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": _tool_name(function["name"]),
                "description": function.get("description", ""),
                "parameters": _schema(function["parameters"]),
            },
        }
        for function in functions
    ]


def check_call(tool_calls: list[dict[str, Any]], ground_truth: list[dict[str, Any]]) -> tuple[bool, str]:
    """Return (correct, reason) for a model's tool calls against BFCL possible answers."""
    if not tool_calls:
        return False, "no_tool_call"
    if len(tool_calls) != len(ground_truth):
        return False, "wrong_call_count"
    expected_name, allowed = next(iter(ground_truth[0].items()))
    call = tool_calls[0]["function"]
    if call["name"] != _tool_name(expected_name):
        return False, "wrong_function"
    try:
        arguments = json.loads(call.get("arguments") or "{}")
    except json.JSONDecodeError:
        return False, "unparseable_arguments"
    if set(arguments) - set(allowed):
        return False, "unexpected_parameter"
    for name, values in allowed.items():
        if name not in arguments:
            if "" not in values:
                return False, "missing_parameter"
            continue
        if not any(_matches(arguments[name], value) for value in values if value != ""):
            return False, "wrong_value"
    return True, "correct"


def run_case(base_url: str, model: str, case: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": case["question"][0],
        "tools": to_openai_tools(case["function"]),
        "tool_choice": "auto",
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    record: dict[str, Any] = {"id": case["id"], "category": case["category"]}
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            message = json.loads(response.read())["choices"][0]["message"]
    except OSError as exc:
        return {**record, "correct": False, "reason": "request_error", "error": str(exc)}
    tool_calls = message.get("tool_calls") or []
    correct, reason = check_call(tool_calls, case["ground_truth"])
    return {**record, "correct": correct, "reason": reason, "tool_calls": tool_calls, "content": message.get("content")}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        k, n = sum(r["correct"] for r in rows), len(rows)
        return {"correct": k, "total": n, "accuracy": k / n if n else None, "wilson_95": wilson_ci(k, n)}

    reasons: dict[str, int] = {}
    for r in records:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    return {
        "overall": block(records),
        "by_category": {
            c: block([r for r in records if r["category"] == c]) for c in sorted({r["category"] for r in records})
        },
        "reasons": dict(sorted(reasons.items())),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--categories", default=",".join(CATEGORIES))
    parser.add_argument("--limit", type=int, help="First N cases per category")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--data-dir", type=Path, default=Path("data/public/bfcl"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="summary.json to gate against")
    parser.add_argument("--tolerance", type=float, default=0.03, help="Allowed accuracy drop vs baseline")
    args = parser.parse_args(argv)

    categories = tuple(args.categories.split(","))
    cases = load_cases(args.data_dir, categories)
    if args.limit:
        cases = [c for cat in categories for c in [x for x in cases if x["category"] == cat][: args.limit]]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        records = list(pool.map(lambda case: run_case(args.base_url, args.model, case, args.max_tokens), cases))

    summary = {"model": args.model, "base_url": args.base_url, "concurrency": args.concurrency, **summarize(records)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    overall = summary["overall"]
    print(
        f"BFCL accuracy {overall['accuracy']:.1%} ({overall['correct']}/{overall['total']}), reasons {summary['reasons']}"
    )

    if args.baseline:
        reference = json.loads(args.baseline.read_text())["overall"]["accuracy"]
        drop = reference - overall["accuracy"]
        verdict = "FAIL" if drop > args.tolerance else "PASS"
        print(
            f"gate {verdict}: accuracy {overall['accuracy']:.1%} vs baseline {reference:.1%} (tolerance {args.tolerance:.0%})"
        )
        if verdict == "FAIL":
            sys.exit(1)


def _tool_name(name: str) -> str:
    return name.replace(".", "_")


def _schema(node: Any) -> Any:
    if isinstance(node, dict):
        out = {key: _schema(value) for key, value in node.items()}
        if out.get("type") == "any":
            del out["type"]
        elif isinstance(out.get("type"), str):
            out["type"] = _TYPE_MAP.get(out["type"], out["type"])
        return out
    if isinstance(node, list):
        return [_schema(item) for item in node]
    return node


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual is expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return float(actual) == float(expected) and (isinstance(expected, float) or isinstance(actual, int))
    if isinstance(expected, str) and isinstance(actual, str):
        return _STRIP.sub("", actual.lower()) == _STRIP.sub("", expected.lower())
    if isinstance(expected, list) and isinstance(actual, list):
        return len(actual) == len(expected) and all(_matches(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, dict) and isinstance(actual, dict):
        return all(
            (key not in actual and "" in allowed) or (key in actual and any(_matches(actual[key], v) for v in allowed))
            for key, allowed in expected.items()
        ) and not (set(actual) - set(expected))
    return False


if __name__ == "__main__":
    main()
