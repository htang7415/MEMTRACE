"""Rebuild results/report/index.html (serving platform and agent-session KV memory) from the retained outputs under
data/ (kept locally, not in the repository).

Run from the repository root:  python results/report/build.py
"""

import glob
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "data"
WEEKDAYS = ("2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05")
WEEKEND = ("2026-06-06", "2026-06-07")


def load(path: Path):
    return json.loads(path.read_text())


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pick(rows: list[dict], **k) -> dict:
    return next(r for r in rows if all(r.get(a) == b for a, b in k.items()))


def mean(values) -> float:
    return statistics.fmean(values)


def engines() -> list[dict]:
    out = []
    for engine, name in (
        ("vllm-metal", "vllm-metal (GPU)"),
        ("mlx-lm", "mlx_lm.server (GPU)"),
        ("vllm-cpu", "vLLM CPU"),
    ):
        tok = [
            load(DATA / f"engine_bench_r2/{engine}/sharegpt/c{c}/summary.json")["output_tokens_per_second"]
            for c in (1, 2, 4, 8)
        ]
        out.append({"name": name, "tok": tok})
    return out


def routing() -> dict:
    levels = (8, 16, 32)
    policies = ("random", "queue", "prefix", "combined")
    tok = {
        p: [
            load(DATA / f"routing_study_v2/sim4-{p}/agent-sessions/c{c}/summary.json")["output_tokens_per_second"]
            for c in levels
        ]
        for p in policies
    }
    return {"levels": levels, "tok": tok}


def replay() -> list[dict]:
    out = []
    for policy in ("combined", "capacity", "cache-cost"):
        runs = [
            load(Path(f))
            for f in glob.glob(str(DATA / f"copilot/rep*/hetero-{policy}/copilot-agent/open/summary.json"))
        ]
        out.append(
            {
                "policy": policy,
                "reps": len(runs),
                "wall": mean(r["wall_seconds"] for r in runs),
                "ttft_p95": mean(r["latency"]["ttft_seconds"]["p95"] for r in runs),
            }
        )
    return out


def kv() -> dict:
    m1 = load(DATA / "kvmem/m1-week.json")
    days = m1["days"]
    bins = list(next(iter(days.values()))["hit_by_gap"])

    def avg(group, f):
        return mean(f(days[d]) for d in group)

    causes = ["expiry", "short_gap_grew", "short_gap_shrank", "mid_gap", "compaction", "model_switch"]
    lifetimes = ("5min", "1h", "24h")
    sweep = jsonl(DATA / "kvmem/m2-2026-06-03.jsonl")
    routers = ("least-loaded", "session-key", "sticky", "kv-aware")
    tiers = ("none", "ram256", "ram1024", "ram1024+ssd4000")
    pilot = load(DATA / "m3/pilot/validate.json")
    return {
        "sessions": sum(d["sessions"] for d in days.values()),
        "waste": [
            min(d["waste_share_of_later_prompt"] for d in days.values()),
            max(d["waste_share_of_later_prompt"] for d in days.values()),
        ],
        "first_call": [
            min(d["first_call_cached_share"] for d in days.values()),
            max(d["first_call_cached_share"] for d in days.values()),
        ],
        "bins": bins,
        "hit_weekday": [avg(WEEKDAYS, lambda d, b=b: d["hit_by_gap"][b]["hit_ratio_tokens"]) for b in bins],
        "hit_weekend": [avg(WEEKEND, lambda d, b=b: d["hit_by_gap"][b]["hit_ratio_tokens"]) for b in bins],
        "causes": causes,
        "cause_share": [avg(WEEKDAYS, lambda d, c=c: d["waste_by_cause"][c]["share"]) for c in causes],
        "observed": avg(WEEKDAYS, lambda d: d["lifetimes"]["5min"]["observed_cached_share_of_reusable"]),
        "retained": [avg(WEEKDAYS, lambda d, k=k: d["lifetimes"][k]["retained_share_of_reusable"]) for k in lifetimes],
        "working_tb": [
            avg(WEEKDAYS, lambda d, k=k: d["lifetimes"][k]["working_set_gb"]["Qwen3-4B"]["mean"] / 1000)
            for k in lifetimes
        ],
        "routers": routers,
        "router_hit": {
            str(g): [
                pick(sweep, gpu_gb=g, lower="none", retention="lru", router=r)["hit_share_of_reusable"] for r in routers
            ]
            for g in (16, 32, 64)
        },
        "tiers": tiers,
        "tier_hit": {
            str(g): [
                pick(sweep, gpu_gb=g, lower=t, retention="lru", router="sticky")["hit_share_of_reusable"] for t in tiers
            ]
            for g in (16, 64)
        },
        "pilot": {
            "kv_tokens": pilot["kv_tokens"],
            "bins": [
                {"bin": b, **pilot["by_gap"][b]}
                for b in ("<10s", "10-60s", "1-5min", "5-10min", "10-60min", ">1h")
                if b in pilot["by_gap"]
            ],
        },
        "commits": {
            "retention": m1["provenance"]["git_commit"][:7],
            "simulator": sweep[0]["provenance"]["git_commit"][:7],
            "pilot": pilot["provenance"]["git_commit"][:7],
        },
    }


def main() -> None:
    data = {"engines": engines(), "routing": routing(), "replay": replay(), "kv": kv()}
    body = (HERE / "template.html").read_text().replace("__DATA__", json.dumps(data))
    page = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        "</head>\n<body>\n" + body + "\n</body>\n</html>\n"
    )
    out = HERE / "index.html"
    out.write_text(page)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
