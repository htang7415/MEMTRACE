"""Rebuild results/benchmark/memtrace_results.html (the memory-risk benchmark report) from the retained result files
under data/ (kept locally, not in the repository).

Run from the repository root:  python results/benchmark/build.py
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ACTOR = "mlx-community/Qwen2.5-7B-Instruct-4bit"


def load(path: str):
    return json.loads((ROOT / path).read_text())


def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def rate(counts):
    k, n = counts
    return {"k": k, "n": n, "p": k / n if n else None, "ci": wilson(k, n)}


def page_data() -> dict:
    metrics = load("data/results/metrics.json")["by_configuration"][ACTOR]
    systems = []
    for system in ("S0", "S1", "S2"):
        rc = metrics[system]["rate_counts"]
        cc = metrics[system]["causal_chain_counts"]
        k, n = rc["unsafe_proposal_before_checker"]
        systems.append(
            {
                "id": system,
                "clean": rate(rc["clean_success"]),
                "oneshot": rate(rc["one_shot_unsafe"]),
                "admitted": rate(rc["poison_admitted"]),
                "delayed": rate(rc["stateful_unsafe"]),
                "efr": rate(rc["execution_failure"]),
                "proposal": {"k": k, "n": n},
                "chain": [
                    cc["stateful_attacks"],
                    cc["poison_admitted"],
                    cc["admitted_poison_retrieved_at_trigger"],
                    cc["unsafe_proposal_before_checker"],
                    cc["unsafe_executed"],
                ],
            }
        )

    cal = load("data/calibration/oracle_memory/metrics.json")["calibration_by_condition"]["S1-ORACLE-RETRIEVED-MEMORY"]
    util = {}
    for name, folder in (("tool", "trusted_utility_toolargs_qwen25_7b"), ("policy", "trusted_utility_qwen25_7b")):
        by_system = load(f"data/extensions/{folder}/metrics.json")["by_system"]
        util[name] = {
            system: {
                "admitted": v["trusted_memory_admitted"][:2],
                "retrieved": v["trusted_memory_retrieved_at_trigger"][:2],
                "correct": v["correct_delayed_use"][:2],
                "efr": v["execution_failure"][:2],
            }
            for system, v in by_system.items()
        }

    canonical = {e["episode_id"]: e for e in load("data/results/episode_scores.json")}
    serving = []
    for c in (1, 4):
        folder = f"data/serving_sweep/concurrency_{c}"
        sv = load(f"{folder}/serving.json")
        scores = load(f"{folder}/episode_scores.json")
        same = sum(
            {k: v for k, v in e.items() if k != "trace_path"}
            == {k: v for k, v in canonical[e["episode_id"]].items() if k != "trace_path"}
            for e in scores
        )
        latency = sv["latency"]
        serving.append(
            {
                "c": c,
                "tok_s": sv["output_tokens_per_second"],
                "wall": sv["wall_seconds"],
                "calls": sv["calls"],
                "episodes": sv["episodes"],
                "cache": sv["prefix_cache_hit_rate"],
                "same": same,
                "ttft": latency["ttft_seconds"],
                "tpot": latency["tpot_seconds"],
                "e2e": latency["e2e_seconds"],
            }
        )

    return {
        "systems": systems,
        "cal": {
            "episodes": cal["episodes"],
            "unsafe": cal["unsafe_executed"],
            "proposal": cal["unsafe_proposal_before_checker"],
            "efr": cal["execution_failure"],
        },
        "util": util,
        "serving": serving,
    }


def main() -> None:
    body = (HERE / "template.html").read_text().replace("__DATA__", json.dumps(page_data()))
    page = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        "</head>\n<body>\n" + body + "\n</body>\n</html>\n"
    )
    out = HERE / "memtrace_results.html"
    out.write_text(page)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
