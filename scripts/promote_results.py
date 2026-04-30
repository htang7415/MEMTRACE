try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass

import argparse
import json
import shutil
from pathlib import Path

from memtrace.config import AUDIT_DIR, AUDIT_REPORT_JSON_PATH, AUDIT_TEMPLATE_PATH, FIGURES_DIR, RESULTS_DIR
from memtrace.eval.audit import reviewed_audit_records


RESULT_FILES = (
    "run_summary.json",
    "episode_scores.json",
    "metrics.json",
    "table1.md",
    "supplementary_tables.md",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a validated result directory to paper-facing outputs.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--expected-episodes", type=int, default=324)
    parser.add_argument("--audit-template-path", type=Path, default=AUDIT_TEMPLATE_PATH)
    parser.add_argument("--audit-report-path", type=Path, default=AUDIT_REPORT_JSON_PATH)
    args = parser.parse_args()

    summary_count = promote_results(
        args.source_dir,
        args.results_dir,
        args.figures_dir,
        expected_episodes=args.expected_episodes,
        audit_template_path=args.audit_template_path,
        audit_report_path=args.audit_report_path,
    )
    print(f"promoted_source={args.source_dir}")
    print(f"promoted_episodes={summary_count}")
    print(f"results_dir={args.results_dir}")
    print(f"figures_dir={args.figures_dir}")


def promote_results(
    source_dir: Path,
    results_dir: Path,
    figures_dir: Path,
    *,
    expected_episodes: int | None = None,
    audit_template_path: Path | None = None,
    audit_report_path: Path | None = None,
    audit_dir: Path | None = None,
) -> int:
    _validate_source(
        source_dir,
        expected_episodes=expected_episodes,
        audit_template_path=audit_template_path,
        audit_report_path=audit_report_path,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for filename in RESULT_FILES:
        shutil.copy2(source_dir / filename, results_dir / filename)

    source_figures = source_dir / "figures"
    if source_figures.exists():
        for source in source_figures.glob("*.svg"):
            if not source.name.startswith("._"):
                shutil.copy2(source, figures_dir / source.name)

    if audit_template_path is not None and audit_report_path is not None:
        _copy_audit_outputs(
            source_dir,
            audit_dir or AUDIT_DIR,
            audit_template_path=audit_template_path,
            audit_report_path=audit_report_path,
        )

    with (source_dir / "run_summary.json").open("r", encoding="utf-8") as handle:
        return len(json.load(handle))


def _validate_source(
    source_dir: Path,
    *,
    expected_episodes: int | None,
    audit_template_path: Path | None,
    audit_report_path: Path | None,
) -> None:
    missing = [filename for filename in RESULT_FILES if not (source_dir / filename).exists()]
    if missing:
        raise SystemExit(f"missing required result files in {source_dir}: {', '.join(missing)}")

    with (source_dir / "run_summary.json").open("r", encoding="utf-8") as handle:
        run_summary = json.load(handle)
    if expected_episodes is not None and len(run_summary) != expected_episodes:
        raise SystemExit(f"cannot promote result directory with {len(run_summary)}/{expected_episodes} episodes")

    missing_traces = [
        item.get("episode_id", "<unknown>")
        for item in run_summary
        if not item.get("trace_path") or not Path(str(item["trace_path"])).exists()
    ]
    if missing_traces:
        preview = ", ".join(missing_traces[:5])
        raise SystemExit(f"cannot promote result directory with missing traces: {preview}")

    with (source_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    validation = metrics.get("pilot_validation_by_configuration", {})
    if not validation:
        raise SystemExit(f"{source_dir} has no pilot validation block")
    missing_systems = [
        f"{_short_actor_model(actor_model)}:{system}"
        for actor_model, rows in validation.items()
        for system in ("S0", "S1", "S2")
        if system not in rows
    ]
    if missing_systems:
        raise SystemExit(f"cannot promote incomplete pilot validation rows: {', '.join(missing_systems)}")
    invalid = [
        f"{_short_actor_model(actor_model)}:{system}"
        for actor_model, rows in validation.items()
        for system, row in rows.items()
        if not row.get("official_pilot_valid", False)
    ]
    if invalid:
        raise SystemExit(f"cannot promote pilot-invalid result rows: {', '.join(invalid)}")
    if audit_template_path is not None or audit_report_path is not None:
        if audit_template_path is None or audit_report_path is None:
            raise SystemExit("audit template and audit report paths must be provided together")
        _validate_audit_gate(audit_template_path, audit_report_path)


def _validate_audit_gate(audit_template_path: Path, audit_report_path: Path) -> None:
    if not audit_template_path.exists():
        raise SystemExit(f"missing audit template: {audit_template_path}")
    if not audit_report_path.exists():
        raise SystemExit(f"missing audit report: {audit_report_path}")
    audit_records = _load_jsonl(audit_template_path)
    reviewed = reviewed_audit_records(audit_records)
    if len(reviewed) != len(audit_records):
        raise SystemExit(f"cannot promote with incomplete audit labels: {len(reviewed)}/{len(audit_records)} reviewed")
    with audit_report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    agreement = report.get("agreement_rate", 0.0)
    if agreement < 0.90:
        raise SystemExit(f"cannot promote with audit agreement below 0.900: {agreement:.3f}")


def _copy_audit_outputs(
    source_dir: Path,
    audit_dir: Path,
    *,
    audit_template_path: Path,
    audit_report_path: Path,
) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(audit_template_path, audit_dir / "audit_template.jsonl")
    shutil.copy2(audit_report_path, audit_dir / "audit_report.json")
    for filename in ("audit_sample.json", "audit_report.md", "audit_review.md"):
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, audit_dir / filename)


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _short_actor_model(actor_model: str) -> str:
    if "Qwen2.5-7B" in actor_model:
        return "Qwen2.5-7B"
    if "Qwen2.5-3B" in actor_model:
        return "Qwen2.5-3B"
    if "Llama-3.1-8B" in actor_model:
        return "Llama-3.1-8B"
    if "Llama-3.2-3B" in actor_model:
        return "Llama-3.2-3B"
    return actor_model


if __name__ == "__main__":
    main()
