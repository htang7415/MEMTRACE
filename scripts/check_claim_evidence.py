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
    CLAIM_EVIDENCE_CHECK_PATH,
    METRICS_PATH,
    PAPER_DIR,
    RUN_SUMMARY_PATH,
)
from scripts.validate_run import overall_pilot_valid


@dataclass(frozen=True)
class ClaimEvidenceCheck:
    name: str
    status: str
    detail: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Check manuscript claims against current MEMTRACE evidence.")
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH)
    parser.add_argument("--run-summary-path", type=Path, default=RUN_SUMMARY_PATH)
    parser.add_argument("--audit-report-path", type=Path, default=AUDIT_REPORT_JSON_PATH)
    parser.add_argument("--attribution-labels-path", type=Path, default=ATTRIBUTION_LABELS_PATH)
    parser.add_argument("--manuscript", type=Path, default=PAPER_DIR / "manuscript" / "main.tex")
    parser.add_argument("--output", type=Path, default=CLAIM_EVIDENCE_CHECK_PATH)
    args = parser.parse_args()

    checks = collect_claim_evidence_checks(
        metrics_path=args.metrics_path,
        run_summary_path=args.run_summary_path,
        audit_report_path=args.audit_report_path,
        attribution_labels_path=args.attribution_labels_path,
        manuscript_path=args.manuscript,
    )
    report = render_claim_evidence_report(checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    raise SystemExit(0 if all(check.status == "PASS" for check in checks) else 1)


def collect_claim_evidence_checks(
    *,
    metrics_path: Path,
    run_summary_path: Path,
    audit_report_path: Path,
    attribution_labels_path: Path | None = None,
    manuscript_path: Path,
) -> list[ClaimEvidenceCheck]:
    missing = [path for path in (metrics_path, run_summary_path, audit_report_path, manuscript_path) if not path.exists()]
    if missing:
        return [ClaimEvidenceCheck("input files", "FAIL", "missing " + ", ".join(str(path) for path in missing))]

    metrics = _load_json(metrics_path)
    run_summary = _load_json(run_summary_path)
    audit_report = _load_json(audit_report_path)
    manuscript = manuscript_path.read_text(encoding="utf-8")
    manuscript_lower = manuscript.lower()
    summary = _evidence_summary(metrics, run_summary, audit_report)

    checks = [
        _check_contains("episode count", manuscript, f"{summary['episode_count']}-episode"),
        _check_contains("OVR range", manuscript, _range_text("OVR", summary["OVR_range"])),
        _check_contains("SVR range", manuscript, _range_text("SVR", summary["SVR_range"])),
        _check_contains("EFR range", manuscript, _range_text("EFR", summary["EFR_range"])),
        _check_audit_reconciliation_claim(manuscript_lower, summary),
        _check_pilot_framing(manuscript_lower, summary["pilot_valid"]),
        _check_mechanism_claims(manuscript_lower, summary),
    ]
    if attribution_labels_path is not None:
        checks.append(_check_attribution_claims(manuscript_lower, attribution_labels_path))
    return checks


def render_claim_evidence_report(checks: list[ClaimEvidenceCheck]) -> str:
    blockers = [check for check in checks if check.status != "PASS"]
    lines = [
        "# Claim-Evidence Check",
        "",
        f"- overall: {'PASS' if not blockers else 'BLOCKED'}",
        f"- passing checks: {len(checks) - len(blockers)}/{len(checks)}",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in checks:
        lines.append(f"| {check.name} | {check.status} | {check.detail} |")
    return "\n".join(lines) + "\n"


def overall_claim_evidence_aligned(
    *,
    metrics_path: Path,
    run_summary_path: Path,
    audit_report_path: Path,
    attribution_labels_path: Path | None = None,
    manuscript_path: Path,
) -> bool:
    checks = collect_claim_evidence_checks(
        metrics_path=metrics_path,
        run_summary_path=run_summary_path,
        audit_report_path=audit_report_path,
        attribution_labels_path=attribution_labels_path,
        manuscript_path=manuscript_path,
    )
    return all(check.status == "PASS" for check in checks)


def _evidence_summary(metrics: dict, run_summary: list[dict], audit_report: dict) -> dict:
    rows = [
        row
        for actor_rows in metrics.get("by_configuration", {}).values()
        for row in actor_rows.values()
    ]
    mechanism_counts = [
        row.get("mechanism_counts", {})
        for row in rows
    ]
    return {
        "episode_count": len(run_summary),
        "OVR_range": _metric_range(rows, "OVR"),
        "SVR_range": _metric_range(rows, "SVR"),
        "EFR_range": _metric_range(rows, "execution_failure_rate"),
        "audit_agreements": audit_report.get("agreements", 0),
        "audit_n": audit_report.get("n", 0),
        "pilot_valid": overall_pilot_valid(metrics),
        "admit_and_violate": sum(counts.get("a_admission_and_violation", 0) for counts in mechanism_counts),
        "s1_admit_and_safe": sum(
            row.get("mechanism_counts", {}).get("b_admission_and_no_violation", 0)
            for actor_rows in metrics.get("by_configuration", {}).values()
            for system, row in actor_rows.items()
            if system == "S1"
        ),
        "s2_par": max(
            [
                row.get("PAR", 0.0)
                for actor_rows in metrics.get("by_configuration", {}).values()
                for system, row in actor_rows.items()
                if system == "S2"
            ]
            or [0.0]
        ),
    }


def _metric_range(rows: list[dict], key: str) -> tuple[float, float]:
    values = [float(row[key]) for row in rows if key in row]
    if not values:
        return 0.0, 0.0
    return min(values), max(values)


def _range_text(label: str, values: tuple[float, float]) -> str:
    return f"{label} {values[0]:.3f}--{values[1]:.3f}"


def _check_contains(name: str, manuscript: str, expected: str) -> ClaimEvidenceCheck:
    status = "PASS" if expected in manuscript else "FAIL"
    detail = f"found `{expected}`" if status == "PASS" else f"missing `{expected}`"
    return ClaimEvidenceCheck(name, status, detail)


def _check_pilot_framing(manuscript_lower: str, pilot_valid: bool) -> ClaimEvidenceCheck:
    if pilot_valid:
        return ClaimEvidenceCheck("pilot framing", "PASS", "metrics are pilot-valid")
    markers = ("diagnostic", "not submission-ready")
    missing = [marker for marker in markers if marker not in manuscript_lower]
    status = "PASS" if not missing else "FAIL"
    detail = "diagnostic framing present" if status == "PASS" else "missing " + ", ".join(missing)
    return ClaimEvidenceCheck("pilot framing", status, detail)


def _check_mechanism_claims(manuscript_lower: str, summary: dict) -> ClaimEvidenceCheck:
    required = []
    if summary["admit_and_violate"] == 0:
        required.append("zero admit-and-violate")
    if summary["s1_admit_and_safe"] > 0:
        required.append(_count_marker(summary["s1_admit_and_safe"], "admit-and-safe"))
    if summary["s2_par"] == 0.0:
        required.append("s2 has zero poison admissions")
    missing = [marker for marker in required if marker not in manuscript_lower]
    status = "PASS" if not missing else "FAIL"
    detail = "mechanism claims match counts" if status == "PASS" else "missing " + ", ".join(missing)
    return ClaimEvidenceCheck("mechanism claims", status, detail)


def _check_attribution_claims(manuscript_lower: str, attribution_labels_path: Path) -> ClaimEvidenceCheck:
    if not attribution_labels_path.exists():
        return ClaimEvidenceCheck("failure attribution claims", "FAIL", f"missing {attribution_labels_path}")
    labels = _load_json(attribution_labels_path)
    memory_mediated_unsafe = sum(
        1
        for item in labels
        if item.get("unsafe") and item.get("attribution_label") == "memory-mediated"
    )
    if memory_mediated_unsafe == 0:
        markers = ("zero memory-mediated unsafe", "no memory-mediated unsafe")
        if any(marker in manuscript_lower for marker in markers):
            return ClaimEvidenceCheck("failure attribution claims", "PASS", "memory-mediated unsafe count is zero")
        return ClaimEvidenceCheck(
            "failure attribution claims",
            "FAIL",
            "missing zero/no memory-mediated unsafe attribution claim",
        )
    expected = f"{memory_mediated_unsafe} memory-mediated unsafe"
    status = "PASS" if expected in manuscript_lower else "FAIL"
    detail = f"found `{expected}`" if status == "PASS" else f"missing `{expected}`"
    return ClaimEvidenceCheck("failure attribution claims", status, detail)


def _check_audit_reconciliation_claim(manuscript_lower: str, summary: dict) -> ClaimEvidenceCheck:
    expected = f"{summary['audit_agreements']}/{summary['audit_n']}"
    has_count = expected in manuscript_lower
    has_reconciliation = "audit-packet reconciliation" in manuscript_lower or "scorer/audit agreement" in manuscript_lower
    status = "PASS" if has_count and has_reconciliation else "FAIL"
    detail = (
        f"{expected} audit reconciliation claim present"
        if status == "PASS"
        else f"missing retained audit-packet reconciliation claim with {expected}"
    )
    return ClaimEvidenceCheck("audit reconciliation", status, detail)


def _count_marker(count: int, noun: str) -> str:
    words = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
    }
    return f"{words.get(count, str(count))} {noun}"


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
