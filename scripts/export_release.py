try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass

import json
import shutil
from pathlib import Path

from memtrace.config import (
    DOCS_DIR,
    EPISODE_SCORES_PATH,
    FIGURES_DIR,
    GOLD_DIR,
    METRICS_PATH,
    RELEASE_DIR,
    RELEASE_MANIFEST_PATH,
    RESULTS_DIR,
    SUPPLEMENTARY_TABLES_MD_PATH,
    TABLE1_MD_PATH,
    TRACES_DIR,
)


def main() -> None:
    export_release_bundle()
    print(f"release_dir={RELEASE_DIR}")
    print(f"manifest_path={RELEASE_MANIFEST_PATH}")


def export_release_bundle() -> None:
    _reset_release_dir(RELEASE_DIR)

    targets = [
        (GOLD_DIR / "tasks.json", RELEASE_DIR / "gold" / "tasks.json"),
        (GOLD_DIR / "labels.json", RELEASE_DIR / "gold" / "labels.json"),
        (RESULTS_DIR / "run_summary.json", RELEASE_DIR / "results" / "run_summary.json"),
        (EPISODE_SCORES_PATH, RELEASE_DIR / "results" / "episode_scores.json"),
        (METRICS_PATH, RELEASE_DIR / "results" / "metrics.json"),
        (TABLE1_MD_PATH, RELEASE_DIR / "results" / "table1.md"),
        (SUPPLEMENTARY_TABLES_MD_PATH, RELEASE_DIR / "results" / "supplementary_tables.md"),
        (DOCS_DIR / "dataset_card.md", RELEASE_DIR / "docs" / "dataset_card.md"),
    ]
    for source, destination in targets:
        _copy_file(source, destination)

    _copy_tree(FIGURES_DIR, RELEASE_DIR / "figures", suffixes={".svg"})
    _copy_tree(TRACES_DIR, RELEASE_DIR / "traces", suffixes={".jsonl"})

    manifest = build_manifest(RELEASE_DIR)
    RELEASE_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_manifest(release_dir: Path) -> dict:
    files = []
    for path in sorted(release_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("._"):
            continue
        files.append(
            {
                "path": str(path.relative_to(release_dir)),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "name": "MEMTRACE",
        "version": "0.1.0",
        "files": files,
    }


def _reset_release_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree(source_dir: Path, destination_dir: Path, suffixes: set[str]) -> None:
    for source in source_dir.glob("*"):
        if source.is_file() and source.suffix in suffixes and not source.name.startswith("._"):
            _copy_file(source, destination_dir / source.name)


if __name__ == "__main__":
    main()
