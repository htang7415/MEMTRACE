"""BFCL single-call tool-calling accuracy as a task-level quality gate for engine changes.

Runs BFCL v3 `simple` and `multiple` cases through any OpenAI-compatible `/v1/chat/completions`
endpoint with the `tools` API and checks the returned call against BFCL's possible answers.

Cases load through `memtrace.datasets.loaders.public` and are graded by `memtrace.evals.graders.bfcl` (BFCL's
AST rules), the same grader as the E5 accuracy study. Scores are close to, but not identical with, the
official leaderboard.

    memtrace evaluate bfcl --base-url http://localhost:8200/v1 --model Qwen/Qwen3-0.6B --out-dir data/bfcl/vllm-metal
    memtrace evaluate bfcl ... --baseline data/bfcl/vllm-metal/summary.json --tolerance 0.03   # exit 1 on regression
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from memtrace.datasets.loaders.public import BfclCase, load_bfcl
from memtrace.evals.graders import bfcl

CATEGORIES = ("simple", "multiple")
# Grader messages by prefix -> the reason codes reported in summary.json
_REASONS = (
    ("wrong function", "wrong_function"),
    ("expected ", "wrong_call_count"),
    ("each call", "wrong_call_count"),
    ("missing", "missing_parameter"),
    ("unexpected parameter", "unexpected_parameter"),
)


def wilson_ci(k: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson score interval for k successes in n trials ([0, 1] when n is 0)."""
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    denominator = n + z**2
    center = (k + z**2 / 2) / denominator
    margin = z * math.sqrt(n * p * (1 - p) + z**2 / 4) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def load_cases(data_dir: Path | None, categories: tuple[str, ...]) -> list[BfclCase]:
    """Cases of each category; `data_dir` None reads the pinned files under data/public/bfcl."""
    return [case for category in categories for case in load_bfcl(category, data_dir)]


def check_call(tool_calls: list[dict[str, Any]], case: BfclCase) -> tuple[bool, str]:
    """Return (correct, reason) for a model's OpenAI `tool_calls` against the case's possible answers."""
    if not tool_calls:
        return False, "no_tool_call"
    try:
        calls = bfcl.parse_openai_tool_calls(tool_calls)
    except ValueError:
        return False, "unparseable_arguments"
    grade = bfcl.grade(case, calls)
    if grade.correct:
        return True, "correct"
    error = grade.error or ""
    return False, next((code for prefix, code in _REASONS if error.startswith(prefix)), "wrong_value")


def run_case(base_url: str, model: str, case: BfclCase, max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": list(case.messages),
        "tools": bfcl.to_openai_tools(case.functions),
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
    record: dict[str, Any] = {"id": case.id, "category": case.category}
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            message = json.loads(response.read())["choices"][0]["message"]
    except OSError as exc:
        return {**record, "correct": False, "reason": "request_error", "error": str(exc)}
    tool_calls = message.get("tool_calls") or []
    correct, reason = check_call(tool_calls, case)
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
    parser.add_argument("--data-dir", type=Path, help="BFCL files to use instead of the pinned data/public/bfcl")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="summary.json to gate against")
    parser.add_argument("--tolerance", type=float, default=0.03, help="Allowed accuracy drop vs baseline")
    args = parser.parse_args(argv)

    categories = tuple(args.categories.split(","))
    cases = load_cases(args.data_dir, categories)
    if args.limit:
        cases = [c for cat in categories for c in [x for x in cases if x.category == cat][: args.limit]]
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


if __name__ == "__main__":
    main()
