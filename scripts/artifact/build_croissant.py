import json
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or validate Croissant metadata for the MEMTRACE artifact.")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    path = ROOT / "croissant.json"
    if not path.exists():
        path = ROOT / "croissant_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    required = {"@context", "@type", "name", "description", "license", "distribution"}
    missing = sorted(required - set(metadata))
    if missing:
        raise SystemExit("missing Croissant fields: " + ", ".join(missing))
    if args.validate:
        print(f"croissant_metadata_valid={path}")
        return
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"croissant_metadata_path={path}")


if __name__ == "__main__":
    main()
