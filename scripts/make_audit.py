try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass

import argparse
import json
from pathlib import Path

from memtrace.config import (
    AUDIT_REPORT_JSON_PATH,
    AUDIT_REPORT_MD_PATH,
    AUDIT_SAMPLE_PATH,
    AUDIT_TEMPLATE_PATH,
    EPISODE_SCORES_PATH,
)
from memtrace.eval.audit import (
    audit_template_records,
    labeler_audit_report,
    render_audit_report_markdown,
    reviewed_audit_records,
    stratified_audit_sample,
    stratified_sample_size,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the MEMTRACE label-audit packet and report.")
    parser.add_argument("--episode-scores", type=Path, default=EPISODE_SCORES_PATH)
    parser.add_argument("--sample-path", type=Path, default=AUDIT_SAMPLE_PATH)
    parser.add_argument("--template-path", type=Path, default=AUDIT_TEMPLATE_PATH)
    parser.add_argument("--report-json-path", type=Path, default=AUDIT_REPORT_JSON_PATH)
    parser.add_argument("--report-md-path", type=Path, default=AUDIT_REPORT_MD_PATH)
    parser.add_argument("--sample-size", type=int, default=stratified_sample_size())
    args = parser.parse_args()

    episode_scores = _load_json(args.episode_scores)
    sample = stratified_audit_sample(episode_scores, sample_size=args.sample_size)
    template = _load_existing_template(args.template_path) or audit_template_records(sample)
    reviewed = reviewed_audit_records(template)
    report = labeler_audit_report(episode_scores, reviewed)

    _write_json(args.sample_path, sample)
    _write_jsonl(args.template_path, template)
    _write_json(args.report_json_path, report)
    args.report_md_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_md_path.write_text(
        render_audit_report_markdown(report, pending_count=len(template) - len(reviewed)),
        encoding="utf-8",
    )

    print(f"audit_sample={len(sample)}")
    print(f"audit_reviewed={len(reviewed)}")
    print(f"audit_template_path={args.template_path}")
    print(f"audit_report_path={args.report_md_path}")


def _load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_existing_template(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    for record in records:
        record.pop("rule_label", None)
    return records


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
