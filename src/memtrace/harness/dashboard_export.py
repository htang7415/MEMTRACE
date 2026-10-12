"""Copy the latest results.json of each published experiment into the dashboard's static data dir, plus
analyses.json: aggregates of the analyses that are not harness runs (provider-cache retention over the Copilot
week, the simulator-vs-engine check, the BFCL engine gate), and the resume track: each study's trials as its fixed
decision rule counted them, repeat by repeat, with its verdict.

The dashboard reads only these files (plus index.json); nothing is fetched at runtime. Each result is
validated against the result model first, so a stale or hand-edited bundle fails the export. The exported
files are committed and published with the site, so paths under the home directory are written as `~` and
machine details (chip, memory) are left out.

    memtrace report dashboard   (python -m memtrace.harness.dashboard_export [--out dashboard/public/data])
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import yaml

from memtrace.harness.results import ExperimentResult, aggregate_cells, from_dict
from memtrace.kv.resume_decision import invalid_reason, latest_trials

# experiment name -> dashboard page
EXPERIMENTS = {
    "e1-engines-gpu": "engines",
    "e1-engines-cpu": "engines",
    "e7-engines-sharegpt": "engines",
    "e7b-engines-mooncake": "engines",
    "e8-routing-kind-sims": "routing",
    "e9-hetero-pool": "routing",
    "e9b-hetero-policies-agent": "routing",
    "e9c-hetero-policies-mooncake": "routing",
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

# The resume track (`memtrace run resume`): per study, its runs earliest first (a later valid trial replaces an
# earlier one of the same arm and repeat, as the decision rule reads them) and the verdict of its fixed rule.
RESUME_STUDIES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "R1: host pool + SSD vs stock at equal memory, one replica",
        ("20261010T214304Z-r1-resume-ssd-rerun", "20261011T052531Z-r1-resume-ssd-rerun"),  # R1's own run paged
        "Resumed recompute 87 / 57 / 95% lower and resumed TTFT p95 41–53% lower with the tier, but resumed TTFT p50"
        " only 0 / 7 / 18% lower: the rule's 15% in every repeat holds for recompute, not for the median. All-calls"
        " TTFT p50 is 1–7% higher (the tier writes every computed block).",
    ),
    (
        "R2: R1 with every prompt doubled",
        ("20261011T174012Z-r2-resume-prefix",),
        "Only repeat 1 ran without paging: recompute 63% lower, resumed TTFT p50 7% and p95 38% lower. The median"
        " misses the 15% bar in that repeat, so a larger prefix does not move it either; the paged repeats were not"
        " rerun.",
    ),
    (
        "Item 1: two replicas, round robin + one shared SSD tier vs session affinity + private tiers",
        ("20261011T224922Z-s2-shared-tier", "20261012T032542Z-s2-shared-tier-rerun"),
        "Affinity recomputes 45–89% less resumed prefix than round robin with private tiers. A shared tier wins back"
        " 91–100% of the tokens round robin recomputes, but it is not within 15% of affinity in every repeat (repeat"
        " 2: 68% more resumed recompute, 38% higher p95), and its all-calls TTFT p50 is 10–26% higher, as calls"
        " affinity serves from the GPU are restored from the SSD. Every trial paged while it measured; the verdict"
        " is the rule's fallback (a paged repeat counts only if all its copies agree).",
    ),
    (
        "Item 2 probe: load levels for the capacity sweep (stock arm)",
        ("20261012T045950Z-item2-load-probe",),
        "Requests queue from 24 sessions at once (waiting in 27% of samples), and from 36 the GPU cache thrashes:"
        " every reusable prefix is recomputed and the median TTFT reaches 23–33 s. This set the sweep's levels (12, 24, 36)"
        " and its SLO: resumed TTFT p95 (failed calls counted) ≤ 5,300 ms, twice the value at 12.",
    ),
    (
        "Item 2: sessions served at once within the SLO, one replica",
        ("20261012T060631Z-item2-capacity", "20261012T152348Z-item2-capacity-rerun"),
        "Capacity (the highest level within the SLO in every repeat) is 12 for both arms: no win. The tier met 24 in"
        " two of three repeats (1.8–1.9 s against stock's 5.9–9.6 s) and missed repeat 2 (5.9 s; rerun 6.3 s; both"
        " copies paged and agree). One session there outgrows the context window and fails 9 calls in both arms."
        " At 36 both arms miss; the tier serves 20–105% more calls.",
    ),
)
RESUME_METRICS = (
    "resumed_recompute_share",
    "resumed_ttft_p50_ms",
    "resumed_ttft_all_p95_ms",
    "all_ttft_p50_ms",
    "calls",
)


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


def resume_track(runs_dir: Path = SEARCH_DIRS[0]) -> list[dict[str, Any]]:
    """Each resume study's trials as its decision rule reads them: one per arm and repeat, flagged when it does not
    count (paged or slept while it measured). Studies whose runs are not on disk are left out."""
    studies = []
    for title, run_ids, verdict in RESUME_STUDIES:
        paths = [runs_dir / r / "results.json" for r in run_ids]
        if not all(p.exists() for p in paths):
            continue
        runs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
        trials = latest_trials(runs)
        order = list(dict.fromkeys(t["cell_id"].removeprefix("arm=") for run in runs for t in run["trials"]))
        rows = [
            {
                "cell": cell,
                "repeat": seed,
                "run_id": trials[(cell, seed)]["run_id"],
                "excluded": invalid_reason(trials[(cell, seed)]),
                **{m: trials[(cell, seed)]["metrics"].get(m) for m in RESUME_METRICS},
            }
            for cell, seed in sorted(trials, key=lambda k: (order.index(k[0]), k[1]))
        ]
        studies.append({"title": title, "runs": list(run_ids), "verdict": verdict, "rows": rows})
    return studies


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
        # Re-aggregate so runs recorded before a fix to the aggregation are published under the current rules
        # (means are unchanged; intervals are clipped to each metric's range).
        cells = aggregate_cells(result.trials, {c.cell_id: c.params for c in result.cells})
        data["cells"] = [asdict(c) for c in cells]
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
        "resume": resume_track(search_dirs[0]),
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
