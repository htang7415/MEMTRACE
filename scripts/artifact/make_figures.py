import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize packaged MEMTRACE figures without model inference.")
    parser.add_argument("--metrics", type=Path, required=False)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.out if args.out.is_absolute() else ROOT / args.out
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / "figures").glob("*.svg")):
        shutil.copyfile(source, output_dir / source.name)
        print(f"figure_path={output_dir / source.name}")


if __name__ == "__main__":
    main()
