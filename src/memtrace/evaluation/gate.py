"""Regression gate: fail when metrics differ from a frozen baseline beyond tolerance."""

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fail if candidate metrics regress beyond tolerance from a frozen baseline."
    )
    parser.add_argument("--baseline", type=Path, required=True, help="Path to a frozen baseline metrics.json.")
    parser.add_argument("--candidate", type=Path, required=True, help="Path to freshly computed metrics.json.")
    parser.add_argument("--tolerance", type=float, default=1e-9, help="Absolute tolerance for numeric fields.")
    args = parser.parse_args(argv)

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))

    diffs = compare_metrics(
        baseline.get("by_configuration", {}),
        candidate.get("by_configuration", {}),
        tolerance=args.tolerance,
        path="by_configuration",
    )
    if diffs:
        print(f"metrics regression gate FAILED: {len(diffs)} difference(s)")
        for diff in diffs:
            print(f"  {diff}")
        raise SystemExit(1)
    print("metrics regression gate passed")


def compare_metrics(baseline: object, candidate: object, *, tolerance: float, path: str) -> list[str]:
    """Recursively diff two metrics trees, returning one message per differing leaf."""

    if isinstance(baseline, dict) and isinstance(candidate, dict):
        diffs: list[str] = []
        for key in sorted(set(baseline) | set(candidate)):
            key_path = f"{path}.{key}"
            if key not in baseline:
                diffs.append(f"{key_path}: added in candidate (not in baseline)")
                continue
            if key not in candidate:
                diffs.append(f"{key_path}: missing from candidate (present in baseline)")
                continue
            diffs.extend(compare_metrics(baseline[key], candidate[key], tolerance=tolerance, path=key_path))
        return diffs

    if isinstance(baseline, list) and isinstance(candidate, list):
        if len(baseline) != len(candidate):
            return [f"{path}: length baseline={len(baseline)} candidate={len(candidate)}"]
        diffs = []
        for index, (base_item, cand_item) in enumerate(zip(baseline, candidate)):
            diffs.extend(compare_metrics(base_item, cand_item, tolerance=tolerance, path=f"{path}[{index}]"))
        return diffs

    if isinstance(baseline, (int, float)) and isinstance(candidate, (int, float)):
        if abs(baseline - candidate) > tolerance:
            return [f"{path}: baseline={baseline} candidate={candidate}"]
        return []

    if baseline != candidate:
        return [f"{path}: baseline={baseline!r} candidate={candidate!r}"]
    return []


if __name__ == "__main__":
    main()
