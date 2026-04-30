try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from memtrace.config import (
    ATTRIBUTION_LABELS_PATH,
    AUDIT_REPORT_JSON_PATH,
    AUDIT_TEMPLATE_PATH,
    METRICS_PATH,
    NEURIPS_READINESS_PATH,
    RELEASE_DIR,
    RUN_SUMMARY_PATH,
)
from memtrace.eval.audit import reviewed_audit_records
from scripts.check_claim_evidence import overall_claim_evidence_aligned
from scripts.validate_run import overall_pilot_valid


EXPECTED_INCLUDED_EPISODES = 324
ABSOLUTE_PATH_MARKERS = ("/" + "Volumes/Max", "/" + "Users/")
ANONYMITY_MARKERS = ("Hao " + "Tang", "hao" + "tang")
MIN_REFERENCE_COUNT = 20


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    status: str
    detail: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MEMTRACE readiness for a NeurIPS submission pass.")
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH)
    parser.add_argument("--run-summary-path", type=Path, default=RUN_SUMMARY_PATH)
    parser.add_argument("--audit-template-path", type=Path, default=AUDIT_TEMPLATE_PATH)
    parser.add_argument("--audit-report-path", type=Path, default=AUDIT_REPORT_JSON_PATH)
    parser.add_argument("--attribution-labels-path", type=Path, default=ATTRIBUTION_LABELS_PATH)
    parser.add_argument("--release-dir", type=Path, default=RELEASE_DIR)
    parser.add_argument("--paper-dir", type=Path, default=Path("paper/manuscript"))
    parser.add_argument("--output", type=Path, default=NEURIPS_READINESS_PATH)
    parser.add_argument("--expected-episodes", type=int, default=EXPECTED_INCLUDED_EPISODES)
    args = parser.parse_args()

    checks = collect_readiness_checks(
        metrics_path=args.metrics_path,
        run_summary_path=args.run_summary_path,
        audit_template_path=args.audit_template_path,
        audit_report_path=args.audit_report_path,
        attribution_labels_path=args.attribution_labels_path,
        release_dir=args.release_dir,
        paper_dir=args.paper_dir,
        expected_episodes=args.expected_episodes,
    )
    report = render_readiness_report(checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    raise SystemExit(0 if all(check.status == "PASS" for check in checks) else 1)


def collect_readiness_checks(
    *,
    metrics_path: Path,
    run_summary_path: Path,
    audit_template_path: Path,
    audit_report_path: Path,
    attribution_labels_path: Path | None = None,
    release_dir: Path,
    paper_dir: Path,
    expected_episodes: int,
) -> list[ReadinessCheck]:
    checks = [
        _check_pilot_valid(metrics_path),
        _check_claim_discipline(metrics_path, paper_dir),
        _check_claim_evidence(metrics_path, run_summary_path, audit_report_path, attribution_labels_path, paper_dir),
        _check_included_episode_count(run_summary_path, expected_episodes),
        _check_audit_review(audit_template_path, audit_report_path),
        _check_paper_artifacts(paper_dir),
        _check_official_style(paper_dir),
        _check_checklist_completed(paper_dir),
        _check_references(paper_dir),
        _check_release_bundle(release_dir, expected_episodes),
        _check_release_docs(release_dir),
        _check_release_no_appledouble(release_dir),
        _check_release_no_absolute_paths(release_dir),
        _check_release_no_author_markers(release_dir),
    ]
    return checks


def render_readiness_report(checks: list[ReadinessCheck]) -> str:
    blockers = [check for check in checks if check.status != "PASS"]
    lines = [
        "# NeurIPS Readiness",
        "",
        f"- overall: {'PASS' if not blockers else 'BLOCKED'}",
        f"- passing checks: {len(checks) - len(blockers)}/{len(checks)}",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in checks:
        lines.append(f"| {check.name} | {check.status} | {check.detail} |")

    lines.extend(["", "## Blockers", ""])
    if not blockers:
        lines.append("- none")
    else:
        for check in blockers:
            lines.append(f"- {check.name}: {check.detail}")
    return "\n".join(lines) + "\n"


def _check_pilot_valid(metrics_path: Path) -> ReadinessCheck:
    if not metrics_path.exists():
        return ReadinessCheck("pilot validity", "FAIL", f"missing {metrics_path}")
    metrics = _load_json(metrics_path)
    status = "PASS" if overall_pilot_valid(metrics) else "FAIL"
    return ReadinessCheck("pilot validity", status, "all included rows pass" if status == "PASS" else "one or more rows fail")


def _check_claim_discipline(metrics_path: Path, paper_dir: Path) -> ReadinessCheck:
    manuscript_path = paper_dir / "main.tex"
    if not metrics_path.exists():
        return ReadinessCheck("claim discipline", "FAIL", f"missing {metrics_path}")
    if not manuscript_path.exists():
        return ReadinessCheck("claim discipline", "FAIL", f"missing {manuscript_path}")
    metrics = _load_json(metrics_path)
    manuscript = manuscript_path.read_text(encoding="utf-8").lower()
    if overall_pilot_valid(metrics):
        return ReadinessCheck("claim discipline", "PASS", "pilot-valid metrics allow final framing")
    required_markers = ("diagnostic", "not submission-ready")
    missing = [marker for marker in required_markers if marker not in manuscript]
    status = "PASS" if not missing else "FAIL"
    detail = (
        "diagnostic framing present for pilot-invalid rows"
        if not missing
        else "pilot-invalid rows require manuscript markers: " + ", ".join(missing)
    )
    return ReadinessCheck("claim discipline", status, detail)


def _check_claim_evidence(
    metrics_path: Path,
    run_summary_path: Path,
    audit_report_path: Path,
    attribution_labels_path: Path | None,
    paper_dir: Path,
) -> ReadinessCheck:
    manuscript_path = paper_dir / "main.tex"
    aligned = overall_claim_evidence_aligned(
        metrics_path=metrics_path,
        run_summary_path=run_summary_path,
        audit_report_path=audit_report_path,
        attribution_labels_path=attribution_labels_path,
        manuscript_path=manuscript_path,
    )
    status = "PASS" if aligned else "FAIL"
    detail = "manuscript claims match metrics and audit" if aligned else "run scripts/check_claim_evidence.py for details"
    return ReadinessCheck("claim evidence", status, detail)


def _check_included_episode_count(run_summary_path: Path, expected_episodes: int) -> ReadinessCheck:
    if not run_summary_path.exists():
        return ReadinessCheck("included episode count", "FAIL", f"missing {run_summary_path}")
    run_summary = _load_json(run_summary_path)
    count = len(run_summary)
    status = "PASS" if count == expected_episodes else "FAIL"
    return ReadinessCheck("included episode count", status, f"{count}/{expected_episodes} run-summary rows")


def _check_audit_review(audit_template_path: Path, audit_report_path: Path) -> ReadinessCheck:
    if not audit_template_path.exists():
        return ReadinessCheck("human audit", "FAIL", f"missing {audit_template_path}")
    records = _load_jsonl(audit_template_path)
    reviewed = reviewed_audit_records(records)
    if len(reviewed) != len(records):
        return ReadinessCheck("human audit", "FAIL", f"{len(reviewed)}/{len(records)} labels reviewed")
    if not audit_report_path.exists():
        return ReadinessCheck("human audit", "FAIL", f"missing {audit_report_path}")
    report = _load_json(audit_report_path)
    agreement = report.get("agreement_rate", 0.0)
    status = "PASS" if agreement >= 0.90 else "FAIL"
    return ReadinessCheck("human audit", status, f"{len(reviewed)}/{len(records)} reviewed, agreement={agreement:.3f}")


def _check_paper_artifacts(paper_dir: Path) -> ReadinessCheck:
    required = [
        paper_dir / "main.tex",
        paper_dir / "main.pdf",
        paper_dir / "checklist.tex",
        paper_dir / "tables" / "main_results.tex",
        paper_dir / "tables" / "mechanism_counts.tex",
        paper_dir / "tables" / "causal_chain_counts.tex",
        paper_dir / "tables" / "confidence_intervals.tex",
        paper_dir / "tables" / "failure_attribution.tex",
        paper_dir / "tables" / "task_level_signal.tex",
        paper_dir / "tables" / "validity_gates.tex",
        paper_dir / "figures" / "figure1_pipeline.pdf",
        paper_dir / "figures" / "figure2_ovr_vs_svr.pdf",
        paper_dir / "figures" / "figure3_par_by_task_family.pdf",
    ]
    missing = [path for path in required if not path.exists()]
    status = "PASS" if not missing else "FAIL"
    detail = "paper source, PDF, tables, and figures exist" if not missing else "missing " + ", ".join(str(path) for path in missing)
    return ReadinessCheck("paper artifacts", status, detail)


def _check_official_style(paper_dir: Path) -> ReadinessCheck:
    style_path = paper_dir / "neurips_2026.sty"
    status = "PASS" if style_path.exists() else "FAIL"
    detail = "official style file present" if status == "PASS" else "add official neurips_2026.sty to paper/manuscript/"
    return ReadinessCheck("official NeurIPS style", status, detail)


def _check_checklist_completed(paper_dir: Path) -> ReadinessCheck:
    checklist_path = paper_dir / "checklist.tex"
    if not checklist_path.exists():
        return ReadinessCheck("official checklist", "FAIL", "missing paper/manuscript/checklist.tex")
    text = checklist_path.read_text(encoding="utf-8")
    markers = ("\\answerTODO", "\\justificationTODO", "%%% BEGIN INSTRUCTIONS %%%", "%%% END INSTRUCTIONS %%%")
    remaining = [marker for marker in markers if marker in text]
    status = "PASS" if not remaining else "FAIL"
    detail = "checklist answers completed" if status == "PASS" else "unfilled official checklist template remains"
    return ReadinessCheck("official checklist", status, detail)


def _check_references(paper_dir: Path) -> ReadinessCheck:
    references_path = paper_dir / "references.bib"
    if not references_path.exists():
        return ReadinessCheck("references", "FAIL", "missing paper/manuscript/references.bib")
    text = references_path.read_text(encoding="utf-8")
    count = text.count("@")
    status = "PASS" if count >= MIN_REFERENCE_COUNT else "FAIL"
    return ReadinessCheck("references", status, f"{count} bibliography entries")


def _check_release_bundle(release_dir: Path, expected_episodes: int) -> ReadinessCheck:
    manifest_path = release_dir / "manifest.json"
    run_summary_path = release_dir / "results" / "run_summary.json"
    if not manifest_path.exists() or not run_summary_path.exists():
        return ReadinessCheck("release bundle", "FAIL", "missing manifest or release run summary")
    run_summary = _load_json(run_summary_path)
    trace_count = len(list((release_dir / "traces" / "v1_main_324").glob("*.jsonl")))
    calibration_metrics_path = release_dir / "results" / "calibration_oracle_memory_metrics.json"
    calibration_expected = 0
    if calibration_metrics_path.exists():
        calibration_metrics = _load_json(calibration_metrics_path)
        calibration_expected = sum(
            row.get("episodes", 0) for row in calibration_metrics.get("calibration_by_condition", {}).values()
        )
    calibration_trace_count = len(list((release_dir / "traces" / "calibration_oracle_memory_72").glob("*.jsonl")))
    status = (
        "PASS"
        if len(run_summary) == expected_episodes
        and trace_count == expected_episodes
        and calibration_trace_count == calibration_expected
        else "FAIL"
    )
    return ReadinessCheck(
        "release bundle",
        status,
        (
            f"{len(run_summary)}/{expected_episodes} run-summary rows, "
            f"{trace_count}/{expected_episodes} main traces, "
            f"{calibration_trace_count}/{calibration_expected} calibration traces"
        ),
    )


def _check_release_docs(release_dir: Path) -> ReadinessCheck:
    required = [
        "VALIDATION.md",
        "REPRODUCE.md",
        "RELEASE_MANIFEST.md",
        "TRACE_SCHEMA.md",
        "DATASET_CARD.md",
        "EVAL_CARD.md",
        "EVALUATION_CARD.md",
        "THIRD_PARTY_ASSETS.md",
        "croissant_metadata.json",
        "croissant.json",
        "requirements.txt",
        "environment.yml",
    ]
    missing = [name for name in required if not (release_dir / name).exists()]
    status = "PASS" if not missing else "FAIL"
    detail = "artifact docs and environment files present" if not missing else "missing " + ", ".join(missing)
    return ReadinessCheck("release artifact docs", status, detail)


def _check_release_no_appledouble(release_dir: Path) -> ReadinessCheck:
    sidecars = sorted(path for path in release_dir.rglob("._*") if path.is_file())
    status = "PASS" if not sidecars else "FAIL"
    detail = "no AppleDouble sidecars" if not sidecars else f"{len(sidecars)} AppleDouble sidecars"
    return ReadinessCheck("release AppleDouble cleanup", status, detail)


def _check_release_no_absolute_paths(release_dir: Path) -> ReadinessCheck:
    offenders = []
    text_suffixes = {".json", ".jsonl", ".md", ".tex", ".bib", ".txt", ".toml", ".py"}
    for path in release_dir.rglob("*"):
        if not path.is_file() or path.suffix not in text_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(marker in text for marker in ABSOLUTE_PATH_MARKERS):
            offenders.append(path)
    status = "PASS" if not offenders else "FAIL"
    detail = (
        "no local absolute paths"
        if not offenders
        else "absolute paths in " + ", ".join(_release_relative(path, release_dir) for path in offenders[:5])
    )
    return ReadinessCheck("release path hygiene", status, detail)


def _check_release_no_author_markers(release_dir: Path) -> ReadinessCheck:
    offenders = _files_containing_markers(release_dir, ANONYMITY_MARKERS)
    status = "PASS" if not offenders else "FAIL"
    detail = (
        "no known author markers"
        if not offenders
        else "author markers in " + ", ".join(_release_relative(path, release_dir) for path in offenders[:5])
    )
    return ReadinessCheck("release anonymity", status, detail)


def _files_containing_markers(release_dir: Path, markers: tuple[str, ...]) -> list[Path]:
    offenders = []
    for path in release_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(marker in text for marker in markers):
            offenders.append(path)
    return offenders


def _release_relative(path: Path, release_dir: Path) -> str:
    try:
        return str(path.relative_to(release_dir))
    except ValueError:
        return path.name


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
