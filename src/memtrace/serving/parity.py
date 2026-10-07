"""Greedy-output parity between engines: compare concurrency-1 outputs request by request."""

from __future__ import annotations

import argparse
import difflib
import re
import json
from pathlib import Path
from typing import Any


def parity(reference: Path, candidate: Path) -> dict[str, Any]:
    """Exact-match rate and mean character similarity between two `requests.jsonl` files."""
    ref = {r["index"]: _generated_text(r) for r in _records(reference) if r.get("ok")}
    cand = {r["index"]: _generated_text(r) for r in _records(candidate) if r.get("ok")}
    shared = sorted(ref.keys() & cand.keys())
    if not shared:
        return {"compared": 0, "exact_match_rate": None, "mean_similarity": None, "first_divergence_chars": None}
    exact = sum(ref[i] == cand[i] for i in shared)
    similarity = [difflib.SequenceMatcher(None, ref[i], cand[i]).ratio() for i in shared]
    divergence = [_common_prefix(ref[i], cand[i]) for i in shared if ref[i] != cand[i]]
    return {
        "compared": len(shared),
        "exact_match_rate": exact / len(shared),
        "mean_similarity": sum(similarity) / len(similarity),
        "first_divergence_chars": sorted(divergence)[len(divergence) // 2] if divergence else None,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/engine_bench"))
    parser.add_argument("--reference", required=True, help="Engine whose outputs are the reference")
    parser.add_argument("--workload", default="sharegpt")
    parser.add_argument("--level", default="c1")
    args = parser.parse_args(argv)
    reference = args.root / args.reference / args.workload / args.level / "requests.jsonl"
    for candidate in sorted(args.root.glob(f"*/{args.workload}/{args.level}/requests.jsonl")):
        if candidate != reference:
            print(
                json.dumps(
                    {"reference": args.reference, "candidate": candidate.parts[-4], **parity(reference, candidate)}
                )
            )


def _generated_text(record: dict[str, Any]) -> str:
    """All generated text, normalized across servers that inline `<think>` vs stream it separately."""
    text = (record.get("reasoning_text") or "") + (record.get("output_text") or "")
    return re.sub(r"\s+", " ", re.sub(r"</?think>", " ", text)).strip()


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


if __name__ == "__main__":
    main()
