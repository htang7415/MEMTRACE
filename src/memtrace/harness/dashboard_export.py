"""Copy the latest results.json of each published experiment into the dashboard's static data dir, plus
analyses.json: aggregates of the analyses that are not harness runs (provider-cache retention over the Copilot
week, the simulator-vs-engine check, the BFCL engine gate).

The dashboard reads only these files (plus index.json); nothing is fetched at runtime. Each result is
validated against the result model first, so a stale or hand-edited bundle fails the export. The exported
files are committed and published with the site, so paths under the home directory are written as `~` and
machine details (chip, memory) are left out.

    memtrace report dashboard   (python -m memtrace.harness.dashboard_export [--out dashboard/public/data])
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any

import yaml

from memtrace.harness.results import ExperimentResult, from_dict

# experiment name -> dashboard page
EXPERIMENTS = {
    "e1-engines-gpu": "engines",
    "e1-engines-cpu": "engines",
    "e7-engines-sharegpt": "engines",
    "e7b-engines-mooncake": "engines",
    "e8-routing-kind-sims": "routing",
    "e9-hetero-pool": "routing",
    "e11-copilot-replay": "replay",
    "e11b-copilot-replay-4b": "replay",
    "e11c-copilot-precise-index": "replay",
    "e10-hosted-overflow": "hybrid",
    "k10a-copilot-reuse-routing": "reuse",
    "k10b-copilot-reuse-tiers": "reuse",
    "k10c-copilot-reuse-retention": "reuse",
    "k11-engine-kv-check": "simulator",
    "e2-prefix-caching": "caching",
    "e6-gemini-caching": "caching",
    "e3-llmd-sim": "routing",
    "e3-llmd-metal": "routing",
    "e4-hybrid-gateway": "hybrid",
    "e4b-slo-overflow": "hybrid",
    "e5-gemini": "quality",
    "e5-qwen3-4b": "quality",
    "c1-context-policies": "context",
    "k6-copilot-context-policies": "context",
    "k7-llmd-copilot-context-policies": "context",
    "k8-vllm-metal-copilot-context-policies": "context",
    "k9-gateway-context-qwen3-8b": "gateway",
    "k9b-gateway-mask-min-growth-qwen3-8b": "gateway",
    "c2a-gateway-context": "gateway",
    "c2b-gateway-context": "gateway",
}
SEARCH_DIRS = (Path("data/runs"),)  # every runner writes its bundles here
EXPERIMENTS_DIR = Path("experiments")
RETENTION = Path("data/kvmem/m1-week.json")  # memtrace report retention --by-day
BFCL = Path("data/bfcl")  # memtrace evaluate bfcl, one directory per engine configuration
WEEKDAYS = ("2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05")
WEEKEND = ("2026-06-06", "2026-06-07")
GAP_BINS = ("<10s", "10-60s", "1-5min", "5-10min", "10-60min", ">1h")
CAUSES = ("expiry", "short_gap_grew", "short_gap_shrank", "mid_gap", "compaction", "model_switch")
LIFETIMES = ("5min", "1h", "24h")


def latest_results(search_dirs: tuple[Path, ...] = SEARCH_DIRS) -> dict[str, Path]:
    """Newest complete results.json per experiment name (run ids start with a UTC timestamp)."""
    found: dict[str, tuple[str, Path]] = {}
    for root in search_dirs:
        for path in sorted(root.glob("*/results.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            tools = data["provenance"]["tools"]
            if "trials_planned" in tools and tools.get("trials_completed") != tools["trials_planned"]:
                continue  # interrupted or still running (runners without a plan write results only when done)
            name, run_id = data["name"], data["run_id"]
            if name in EXPERIMENTS and (name not in found or run_id > found[name][0]):
                found[name] = (run_id, path)
    return {name: path for name, (_, path) in found.items()}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def retention(path: Path = RETENTION) -> dict[str, Any] | None:
    """Provider-cache behaviour over the Copilot week: aggregates of the per-day retention analysis."""
    if not path.exists():
        return None
    m1 = json.loads(path.read_text(encoding="utf-8"))
    days = m1["days"]

    def avg(group: tuple[str, ...], f: Any) -> float:
        return _mean([f(days[d]) for d in group])

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
        "bins": list(GAP_BINS),
        "hit_weekday": [avg(WEEKDAYS, lambda d, b=b: d["hit_by_gap"][b]["hit_ratio_tokens"]) for b in GAP_BINS],
        "hit_weekend": [avg(WEEKEND, lambda d, b=b: d["hit_by_gap"][b]["hit_ratio_tokens"]) for b in GAP_BINS],
        "causes": list(CAUSES),
        "cause_share": [avg(WEEKDAYS, lambda d, c=c: d["waste_by_cause"][c]["share"]) for c in CAUSES],
        "observed": avg(WEEKDAYS, lambda d: d["lifetimes"]["5min"]["observed_cached_share_of_reusable"]),
        "retained": [avg(WEEKDAYS, lambda d, k=k: d["lifetimes"][k]["retained_share_of_reusable"]) for k in LIFETIMES],
        "working_tb": [
            avg(WEEKDAYS, lambda d, k=k: d["lifetimes"][k]["working_set_gb"]["Qwen3-4B"]["mean"] / 1000)
            for k in LIFETIMES
        ],
        "git_commit": m1["provenance"]["git_commit"],
    }


def simulator_check(k11: Path | None) -> dict[str, Any] | None:
    """`memtrace report kv-validate` output saved as validate.json next to the K11 run."""
    if k11 is None or not (k11.parent / "validate.json").exists():
        return None
    v = json.loads((k11.parent / "validate.json").read_text(encoding="utf-8"))
    return {
        "kv_tokens": v["kv_tokens"],
        "bins": [{"bin": b, **v["by_gap"][b]} for b in GAP_BINS if b in v["by_gap"]],
        "overall": v["overall"],
        "git_commit": v["provenance"]["git_commit"],
    }


def bfcl_gate(root: Path = BFCL) -> list[dict[str, Any]]:
    """Single-call BFCL accuracy per engine configuration (memtrace evaluate bfcl)."""
    rows = []
    for summary in sorted(root.glob("*/summary.json")):
        s = json.loads(summary.read_text(encoding="utf-8"))["overall"]
        rows.append(
            {
                "config": summary.parent.name,
                "accuracy": s["accuracy"],
                "correct": s["correct"],
                "total": s["total"],
                "wilson_95": s["wilson_95"],
            }
        )
    return rows


def spec_descriptions(experiments: Path = EXPERIMENTS_DIR) -> dict[str, str]:
    """Experiment name -> description of its spec in experiments/."""
    out = {}
    for path in sorted(experiments.glob("[!.]*.yaml")):  # not macOS ._ sidecars
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(spec, dict) and "name" in spec:
            out[str(spec["name"])] = str(spec.get("description", "")).strip()
    return out


def publishable(data: dict[str, Any], descriptions: dict[str, str]) -> str:
    """Result JSON as published: no local user paths and no machine details. The description is the current
    spec's (reviewed text), not the copy recorded with the run."""
    host = data["provenance"]["host"]
    data["provenance"]["host"] = {k: v for k, v in host.items() if k in ("os", "platform", "python_version")}
    if data["name"] in descriptions:
        data["description"] = data["spec"]["description"] = descriptions[data["name"]]
    return json.dumps(data, separators=(",", ":")).replace(str(Path.home()), "~")


def export(out_dir: Path, search_dirs: tuple[Path, ...] = SEARCH_DIRS) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    latest = latest_results(search_dirs)
    descriptions = spec_descriptions()
    for name, path in sorted(latest.items()):
        data = json.loads(path.read_text(encoding="utf-8"))
        result: ExperimentResult = from_dict(ExperimentResult, data)  # strict schema check
        text = publishable(data, descriptions)
        (out_dir / f"{name}.json").write_text(text + "\n", encoding="utf-8")
        entries.append(
            {
                "name": name,
                "page": EXPERIMENTS[name],
                "run_id": result.run_id,
                "file": f"{name}.json",
                "git_commit": result.provenance.git_commit,
                "git_dirty": result.provenance.git_dirty,
                "finished_at": result.provenance.finished_at,
            }
        )
    index = {"experiments": entries}
    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    analyses = {
        "retention": retention(),
        "simulator_check": simulator_check(latest.get("k11-engine-kv-check")),
        "bfcl_gate": bfcl_gate(),
    }
    (out_dir / "analyses.json").write_text(json.dumps(analyses, indent=1) + "\n", encoding="utf-8")
    return index


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("dashboard/public/data"))
    args = parser.parse_args(argv)
    index = export(args.out)
    for e in index["experiments"]:
        print(f"{e['page']:<11} {e['name']:<20} {e['run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
