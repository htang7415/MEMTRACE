try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass

import json
import shutil
from pathlib import Path

from memtrace.config import (
    ALLOWLIST_PATH,
    ATTRIBUTION_LABELS_PATH,
    ATTRIBUTION_REPORT_PATH,
    AUDIT_REPORT_JSON_PATH,
    AUDIT_REPORT_MD_PATH,
    AUDIT_REVIEW_MD_PATH,
    AUDIT_SAMPLE_PATH,
    AUDIT_TEMPLATE_PATH,
    EPISODES_PATH,
    PASSAGES_PATH,
    PROMPTS_DIR,
    RETRIEVAL_VERIFICATION_PATH,
    DOCS_DIR,
    EPISODE_SCORES_PATH,
    FIGURES_DIR,
    GOLD_DIR,
    METRICS_PATH,
    NEURIPS_READINESS_PATH,
    PAPER_BRIEF_PATH,
    RELEASE_DIR,
    RELEASE_MANIFEST_PATH,
    RESULTS_DIR,
    SUPPLEMENTARY_TABLES_MD_PATH,
    TABLE1_MD_PATH,
    TRACES_DIR,
)


APPLEDOUBLE_PREFIX = "._"


def main() -> None:
    export_release_bundle()
    print(f"release_dir={RELEASE_DIR}")
    print(f"manifest_path={RELEASE_MANIFEST_PATH}")


def export_release_bundle() -> None:
    _reset_release_dir(RELEASE_DIR)
    leaderboard_path = RELEASE_DIR / "docs" / "static_leaderboard.md"
    run_summary_path = RESULTS_DIR / "run_summary.json"

    targets = [
        (Path("LICENSE"), RELEASE_DIR / "LICENSE"),
        (PASSAGES_PATH, RELEASE_DIR / "corpus" / "passages.jsonl"),
        (ALLOWLIST_PATH, RELEASE_DIR / "corpus" / "allowlist.json"),
        (EPISODES_PATH, RELEASE_DIR / "episodes" / "episodes.json"),
        (GOLD_DIR / "tasks.json", RELEASE_DIR / "gold" / "tasks.json"),
        (GOLD_DIR / "labels.json", RELEASE_DIR / "gold" / "labels.json"),
        (RETRIEVAL_VERIFICATION_PATH, RELEASE_DIR / "indices" / "verification.json"),
        (METRICS_PATH, RELEASE_DIR / "results" / "metrics.json"),
        (TABLE1_MD_PATH, RELEASE_DIR / "results" / "table1.md"),
        (SUPPLEMENTARY_TABLES_MD_PATH, RELEASE_DIR / "results" / "supplementary_tables.md"),
        (DOCS_DIR / "dataset_card.md", RELEASE_DIR / "docs" / "dataset_card.md"),
        (_project_doc_path(), RELEASE_DIR / "docs" / "project.md"),
    ]
    for source, destination in targets:
        _copy_file(source, destination)
    _copy_json_with_release_trace_paths(run_summary_path, RELEASE_DIR / "results" / "run_summary.json")
    _copy_json_with_release_trace_paths(EPISODE_SCORES_PATH, RELEASE_DIR / "results" / "episode_scores.json")

    _copy_tree(FIGURES_DIR, RELEASE_DIR / "figures", suffixes={".svg"})
    _copy_referenced_traces(run_summary_path, RELEASE_DIR / "traces")
    _copy_tree(PROMPTS_DIR, RELEASE_DIR / "prompts", suffixes={".txt"})
    _copy_optional_file(ATTRIBUTION_LABELS_PATH, RELEASE_DIR / "results" / "attribution_labels.json")
    _copy_optional_file(ATTRIBUTION_REPORT_PATH, RELEASE_DIR / "results" / "attribution_report.md")
    _copy_optional_json_with_release_trace_paths(AUDIT_SAMPLE_PATH, RELEASE_DIR / "audit" / "audit_sample.json")
    _copy_optional_jsonl_with_release_trace_paths(AUDIT_TEMPLATE_PATH, RELEASE_DIR / "audit" / "audit_template.jsonl")
    _copy_optional_file(AUDIT_REPORT_JSON_PATH, RELEASE_DIR / "audit" / "audit_report.json")
    _copy_optional_file(AUDIT_REPORT_MD_PATH, RELEASE_DIR / "audit" / "audit_report.md")
    _copy_optional_text_with_release_trace_paths(AUDIT_REVIEW_MD_PATH, RELEASE_DIR / "audit" / "audit_review.md")
    _copy_optional_file(PAPER_BRIEF_PATH, RELEASE_DIR / "docs" / "neurips_paper_brief.md")
    _copy_optional_file(NEURIPS_READINESS_PATH, RELEASE_DIR / "docs" / "neurips_readiness.md")
    if Path("paper").exists():
        _copy_paper_dir(Path("paper"), RELEASE_DIR / "paper")
    _copy_tree_recursive(Path("memtrace"), RELEASE_DIR / "github_harness" / "memtrace")
    _copy_tree_recursive(Path("scripts"), RELEASE_DIR / "github_harness" / "scripts")
    _copy_tree_recursive(Path("tests"), RELEASE_DIR / "github_harness" / "tests")
    _copy_file(Path("pyproject.toml"), RELEASE_DIR / "github_harness" / "pyproject.toml")
    _copy_file(Path("README.md"), RELEASE_DIR / "github_harness" / "README.md")
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_path.write_text(build_static_leaderboard(METRICS_PATH), encoding="utf-8")

    remove_appledouble_files(RELEASE_DIR)
    manifest = build_manifest(RELEASE_DIR)
    RELEASE_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    remove_appledouble_files(RELEASE_DIR)


def build_manifest(release_dir: Path) -> dict:
    files = []
    for path in sorted(release_dir.rglob("*")):
        if not path.is_file() or path.name.startswith(APPLEDOUBLE_PREFIX):
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


def remove_appledouble_files(root_dir: Path) -> None:
    for path in root_dir.rglob("*"):
        if path.is_file() and path.name.startswith(APPLEDOUBLE_PREFIX):
            path.unlink()


def _short_actor_model(actor_model: str) -> str:
    if "Qwen2.5-3B" in actor_model:
        return "Qwen2.5-3B"
    if "Qwen2.5-7B" in actor_model:
        return "Qwen2.5-7B"
    if "Llama-3.2-3B" in actor_model:
        return "Llama-3.2-3B"
    if "Llama-3.1-8B" in actor_model:
        return "Llama-3.1-8B"
    return actor_model


def _reset_release_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _copy_optional_file(source: Path, destination: Path) -> None:
    if source.exists():
        _copy_file(source, destination)


def _copy_optional_text_with_release_trace_paths(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    text = source.read_text(encoding="utf-8")
    text = text.replace(str(TRACES_DIR) + "/", "traces/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def _copy_json_with_release_trace_paths(source: Path, destination: Path) -> None:
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        payload = [_with_release_trace_path(item) for item in payload]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_optional_json_with_release_trace_paths(source: Path, destination: Path) -> None:
    if source.exists():
        _copy_json_with_release_trace_paths(source, destination)


def _copy_optional_jsonl_with_release_trace_paths(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as in_handle, destination.open("w", encoding="utf-8") as out_handle:
        for line in in_handle:
            if not line.strip():
                continue
            out_handle.write(json.dumps(_with_release_trace_path(json.loads(line))) + "\n")


def _with_release_trace_path(item):
    if not isinstance(item, dict) or "trace_path" not in item:
        return item
    updated = dict(item)
    updated["trace_path"] = f"traces/{Path(str(item['trace_path'])).name}"
    return updated


def _copy_referenced_traces(run_summary_path: Path, destination_dir: Path) -> None:
    with run_summary_path.open("r", encoding="utf-8") as handle:
        run_summary = json.load(handle)
    seen = set()
    for item in run_summary:
        source = Path(item["trace_path"])
        if source in seen:
            continue
        seen.add(source)
        _copy_file(source, destination_dir / source.name)


def _copy_tree(source_dir: Path, destination_dir: Path, suffixes: set[str]) -> None:
    for source in source_dir.glob("*"):
        if source.is_file() and source.suffix in suffixes and not source.name.startswith(APPLEDOUBLE_PREFIX):
            _copy_file(source, destination_dir / source.name)


def _copy_tree_recursive(source_dir: Path, destination_dir: Path) -> None:
    for source in source_dir.rglob("*"):
        if not source.is_file():
            continue
        if source.name.startswith(APPLEDOUBLE_PREFIX) or "__pycache__" in source.parts:
            continue
        _copy_file(source, destination_dir / source.relative_to(source_dir))


def _copy_paper_dir(source_dir: Path, destination_dir: Path) -> None:
    allowed_suffixes = {".tex", ".bib", ".pdf"}
    for source in source_dir.rglob("*"):
        if not source.is_file():
            continue
        if source.name.startswith(APPLEDOUBLE_PREFIX) or source.suffix not in allowed_suffixes:
            continue
        _copy_file(source, destination_dir / source.relative_to(source_dir))


def _project_doc_path() -> Path:
    for candidate in (Path("Project.md"), Path("project.md")):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("missing Project.md/project.md")


if __name__ == "__main__":
    main()
