try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "github_harness").exists():
    sys.path.insert(0, str(ROOT / "github_harness"))

from memtrace.eval.metrics import aggregate_metrics  # noqa: E402
from memtrace.eval.scoring import score_run_summary_items  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute MEMTRACE metrics from retained traces without model inference.")
    parser.add_argument("--main", type=Path, default=Path("data/results"))
    parser.add_argument("--calibration", type=Path, default=Path("data/calibration"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    output_dir = _resolve(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_metrics = _metrics_for("main", _resolve(args.main))
    calibration_metrics = _metrics_for("calibration", _resolve(args.calibration))
    (output_dir / "main_metrics.json").write_text(json.dumps(main_metrics, indent=2), encoding="utf-8")
    (output_dir / "calibration_metrics.json").write_text(json.dumps(calibration_metrics, indent=2), encoding="utf-8")
    print(f"main_metrics={output_dir / 'main_metrics.json'}")
    print(f"calibration_metrics={output_dir / 'calibration_metrics.json'}")


def _metrics_for(kind: str, directory: Path) -> dict:
    if kind == "main":
        summary_path = directory / "run_summary.json"
        trace_dir = _artifact_root() / "traces" / "v1_main_324"
    else:
        summary_path = _calibration_summary_path(directory)
        trace_dir = _artifact_root() / "traces" / "calibration_oracle_memory_72"
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    updated_rows = []
    for row in rows:
        updated = dict(row)
        if trace_dir.exists():
            updated["trace_path"] = str(trace_dir / Path(row["trace_path"]).name)
        updated_rows.append(updated)
    return aggregate_metrics(score_run_summary_items(updated_rows))


def _calibration_summary_path(directory: Path) -> Path:
    candidates = [
        directory / "oracle_memory" / "run_summary.json",
        directory / "s1_oracle_retrieved_memory" / "run_summary.json",
        directory / "run_summary.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("missing calibration run_summary.json under " + str(directory))


def _artifact_root() -> Path:
    if (ROOT / "traces" / "v1_main_324").exists():
        return ROOT
    candidate = ROOT / "paper" / "anonymous_memtrace"
    if (candidate / "traces" / "v1_main_324").exists():
        return candidate
    return ROOT


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
