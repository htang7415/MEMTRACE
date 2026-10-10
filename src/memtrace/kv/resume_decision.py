"""R1's decision rule: does the SSD tier beat equal memory on resumed calls, repeat by repeat?

Pairs two arms' trials by seed across one or more resume runs (each repeat replays one seeded session order, the
same for every arm). A later run's trial replaces an earlier one with the same arm and seed (a rerun), unless the
later one is invalid and the earlier one is not. A trial counts only if it ran, the host slept at most 5 s and
paged nothing out; paging is read from `host_swapouts`, since runs older than the paging flag did not mark it.
Per metric (lower is better) the treatment gets

    A  ≥ 15% better than the baseline in every repeat
    C  ≥ 15% worse in every repeat
    B  otherwise

and A/B/C on recompute and TTFT p50 decide the next step. p95 rests on a few calls per repeat, so it is reported
but decides nothing; all-calls TTFT p50 is reported as the tier's cost.

    memtrace report resume-decision data/runs/<r1> data/runs/<r1-rerun> [--repeats 3] [--out decision.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from memtrace.harness.bundle import MAX_SLEEP_S

METRICS = ("resumed_recompute_share", "resumed_ttft_p50_ms", "resumed_ttft_all_p95_ms")
COST_METRIC = "all_ttft_p50_ms"
THRESHOLD = 0.15


def invalid_reason(trial: dict[str, Any]) -> str | None:
    if trial["status"] != "ok":
        return "failed"
    m = trial["metrics"]
    if m.get("host_slept_s", 0.0) > MAX_SLEEP_S:
        return f"host slept {m['host_slept_s']:.0f} s"
    if m.get("host_swapouts", 0.0) > 0:
        return f"host paged out {m['host_swapouts']:.0f} pages"
    return None


def latest_trials(runs: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    """(arm, seed) -> the trial that counts, taking runs in order."""
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for run in runs:
        for trial in run["trials"]:
            key = (trial["cell_id"].removeprefix("arm="), int(trial["seed"]))
            if key not in out or invalid_reason(trial) is None or invalid_reason(out[key]) is not None:
                out[key] = {**trial, "run_id": run["run_id"]}
    return out


def improvement(base: float, treat: float) -> float:
    """The treatment's relative improvement over the baseline; lower values are better."""
    if base == 0:
        return 0.0 if treat == 0 else -math.inf
    return (base - treat) / base


def grade(changes: list[float]) -> str:
    if all(c >= THRESHOLD for c in changes):
        return "A"
    if all(c <= -THRESHOLD for c in changes):
        return "C"
    return "B"


def next_step(grades: dict[str, str]) -> str:
    recompute, p50 = grades["resumed_recompute_share"], grades["resumed_ttft_p50_ms"]
    if recompute != "A":
        return "stop: the tier does not cut resumed recompute (S1's rule)"
    if p50 == "A":
        return "S2: tier-aware routing"
    if p50 == "B":
        return "R2 before S2: grow the prefix size (larger scale or the 4B model), not only the session count"
    return "not covered by the rule: recompute A but TTFT p50 C"


def decide(runs: list[dict[str, Any]], baseline: str, treatment: str, repeats: int) -> dict[str, Any]:
    trials = latest_trials(runs)
    seeds = sorted({seed for arm, seed in trials if arm in (baseline, treatment)})
    to_rerun = []
    for seed in seeds:
        for arm in (baseline, treatment):
            trial = trials.get((arm, seed))
            reason = "missing" if trial is None else invalid_reason(trial)
            if reason:
                to_rerun.append({"arm": arm, "seed": seed, "reason": reason})
    rerun_seeds = {r["seed"] for r in to_rerun}
    paired = [seed for seed in seeds if seed not in rerun_seeds]
    rows = [
        {
            "seed": seed,
            "runs": {arm: trials[(arm, seed)]["run_id"] for arm in (baseline, treatment)},
            **{
                metric: {
                    "baseline": trials[(baseline, seed)]["metrics"][metric],
                    "treatment": trials[(treatment, seed)]["metrics"][metric],
                    "improvement": improvement(
                        trials[(baseline, seed)]["metrics"][metric], trials[(treatment, seed)]["metrics"][metric]
                    ),
                }
                for metric in (*METRICS, COST_METRIC)
            },
        }
        for seed in paired
    ]
    result: dict[str, Any] = {"baseline": baseline, "treatment": treatment, "repeats": rows, "to_rerun": to_rerun}
    if len(paired) < repeats:
        result["next_step"] = f"incomplete: {len(paired)} of {repeats} repeats have valid trials in both arms"
        return result
    result["grades"] = {metric: grade([row[metric]["improvement"] for row in rows]) for metric in METRICS}
    result["next_step"] = next_step(result["grades"])
    return result


def render(result: dict[str, Any]) -> str:
    lines = [f"{result['treatment']} vs {result['baseline']}, paired by seed"]
    for row in result["repeats"]:
        lines.append(f"  seed {row['seed']}:")
        for metric in (*METRICS, COST_METRIC):
            m = row[metric]
            lines.append(
                f"    {metric:28} {m['baseline']:>10.4g} -> {m['treatment']:>10.4g}  ({m['improvement']:+.0%})"
            )
    for r in result["to_rerun"]:
        lines.append(f"  rerun {r['arm']} seed {r['seed']}: {r['reason']}")
    for metric, g in result.get("grades", {}).items():
        lines.append(f"  {metric}: {g}" + ("  (decides nothing)" if metric == "resumed_ttft_all_p95_ms" else ""))
    lines.append(f"next step: {result['next_step']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+", help="resume run directories, earliest first")
    parser.add_argument("--baseline", default="stock-equal-memory")
    parser.add_argument("--treatment", default="host-pool+ssd")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    runs = [json.loads((d / "results.json").read_text(encoding="utf-8")) for d in args.runs]
    result = decide(runs, args.baseline, args.treatment, args.repeats)
    print(render(result))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
