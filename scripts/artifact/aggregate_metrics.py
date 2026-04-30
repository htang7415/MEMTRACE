import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "github_harness"))

from memtrace.eval.metrics import aggregate_metrics  # noqa: E402
from memtrace.eval.scoring import score_run_summary_items  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate MEMTRACE metrics from packaged traces.")
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    traces_dir = args.traces if args.traces.is_absolute() else ROOT / args.traces
    run_summary_path = _run_summary_path_for_traces(traces_dir)
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    updated_summary = []
    for item in run_summary:
        updated = dict(item)
        updated["trace_path"] = str(traces_dir / Path(item["trace_path"]).name)
        updated_summary.append(updated)
    metrics = aggregate_metrics(score_run_summary_items(updated_summary))

    output_path = args.out if args.out.is_absolute() else ROOT / args.out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"metrics_path={output_path}")


def _run_summary_path_for_traces(traces_dir: Path) -> Path:
    if traces_dir.name == "calibration_oracle_memory_72":
        return ROOT / "results" / "calibration_oracle_memory_run_summary.json"
    return ROOT / "results" / "run_summary.json"


if __name__ == "__main__":
    main()
