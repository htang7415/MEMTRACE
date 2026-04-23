try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass

import json
import shutil
from pathlib import Path

from memtrace.config import (
    ALLOWLIST_PATH,
    EPISODES_PATH,
    PASSAGES_PATH,
    PROMPTS_DIR,
    RETRIEVAL_VERIFICATION_PATH,
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
    leaderboard_path = RELEASE_DIR / "docs" / "static_leaderboard.md"

    targets = [
        (Path("LICENSE"), RELEASE_DIR / "LICENSE"),
        (PASSAGES_PATH, RELEASE_DIR / "corpus" / "passages.jsonl"),
        (ALLOWLIST_PATH, RELEASE_DIR / "corpus" / "allowlist.json"),
        (EPISODES_PATH, RELEASE_DIR / "episodes" / "episodes.json"),
        (GOLD_DIR / "tasks.json", RELEASE_DIR / "gold" / "tasks.json"),
        (GOLD_DIR / "labels.json", RELEASE_DIR / "gold" / "labels.json"),
        (RETRIEVAL_VERIFICATION_PATH, RELEASE_DIR / "indices" / "verification.json"),
        (RESULTS_DIR / "run_summary.json", RELEASE_DIR / "results" / "run_summary.json"),
        (EPISODE_SCORES_PATH, RELEASE_DIR / "results" / "episode_scores.json"),
        (METRICS_PATH, RELEASE_DIR / "results" / "metrics.json"),
        (TABLE1_MD_PATH, RELEASE_DIR / "results" / "table1.md"),
        (SUPPLEMENTARY_TABLES_MD_PATH, RELEASE_DIR / "results" / "supplementary_tables.md"),
        (DOCS_DIR / "dataset_card.md", RELEASE_DIR / "docs" / "dataset_card.md"),
        (Path("project.md"), RELEASE_DIR / "docs" / "project.md"),
    ]
    for source, destination in targets:
        _copy_file(source, destination)

    _copy_tree(FIGURES_DIR, RELEASE_DIR / "figures", suffixes={".svg"})
    _copy_tree(TRACES_DIR, RELEASE_DIR / "traces", suffixes={".jsonl"})
    _copy_tree(PROMPTS_DIR, RELEASE_DIR / "prompts", suffixes={".txt"})
    _copy_tree_recursive(Path("memtrace"), RELEASE_DIR / "github_harness" / "memtrace")
    _copy_tree_recursive(Path("scripts"), RELEASE_DIR / "github_harness" / "scripts")
    _copy_tree_recursive(Path("tests"), RELEASE_DIR / "github_harness" / "tests")
    _copy_file(Path("pyproject.toml"), RELEASE_DIR / "github_harness" / "pyproject.toml")
    _copy_file(Path("README.md"), RELEASE_DIR / "github_harness" / "README.md")
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_path.write_text(build_static_leaderboard(METRICS_PATH), encoding="utf-8")

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


def build_static_leaderboard(metrics_path: Path) -> str:
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    lines = [
        "# Static Leaderboard",
        "",
        "| Actor Model | System | CSR | OVR | SVR | SRG | PAR |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for actor_model, rows in metrics["by_configuration"].items():
        for system in ("S0", "S1", "S2"):
            row = rows[system]
            lines.append(
                "| {actor_model} | {system} | {CSR:.3f} | {OVR:.3f} | {SVR:.3f} | {SRG:.3f} | {PAR:.3f} |".format(
                    actor_model=_short_actor_model(actor_model),
                    system=system,
                    CSR=row["CSR"],
                    OVR=row["OVR"],
                    SVR=row["SVR"],
                    SRG=row["SRG"],
                    PAR=row["PAR"],
                )
            )
    return "\n".join(lines) + "\n"


def _short_actor_model(actor_model: str) -> str:
    if "Qwen2.5-3B" in actor_model:
        return "Qwen2.5-3B"
    if "Llama-3.2-3B" in actor_model:
        return "Llama-3.2-3B"
    return actor_model


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


def _copy_tree_recursive(source_dir: Path, destination_dir: Path) -> None:
    for source in source_dir.rglob("*"):
        if not source.is_file():
            continue
        if source.name.startswith("._") or "__pycache__" in source.parts:
            continue
        _copy_file(source, destination_dir / source.relative_to(source_dir))


if __name__ == "__main__":
    main()
