import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract confidence intervals from MEMTRACE metrics.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    metrics_path = args.metrics if args.metrics.is_absolute() else ROOT / args.metrics
    output_path = args.out if args.out.is_absolute() else ROOT / args.out
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    intervals = {}
    for actor_model, systems in metrics.get("by_configuration", {}).items():
        intervals[actor_model] = {}
        for system, row in systems.items():
            intervals[actor_model][system] = {
                key: value for key, value in row.items() if key.endswith("_ci95")
            }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(intervals, indent=2), encoding="utf-8")
    print(f"confidence_intervals_path={output_path}")


if __name__ == "__main__":
    main()
