import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a no-model MEMTRACE artifact smoke test.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    if not config_path.exists():
        raise SystemExit(f"missing config: {config_path}")

    subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_artifact.py")], check=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "aggregate_metrics.py"),
            "--traces",
            "traces/v1_main_324",
            "--out",
            "tables/main_metrics.json",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "aggregate_metrics.py"),
            "--traces",
            "traces/calibration_oracle_memory_72",
            "--out",
            "tables/calibration_metrics.json",
        ],
        check=True,
    )
    print("smoke_ok")


if __name__ == "__main__":
    main()
