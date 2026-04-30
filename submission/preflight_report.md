# MEMTRACE Submission Preflight

Date: 2026-04-30
Branch: `neurips-ed-final`

Completed local gates:

- `python scripts/validate_release.py`: PASS
- `python scripts/check_neurips_readiness.py`: PASS, 14/14
- `python -m pytest -q`: PASS, 150 tests
- Fresh unzip of `submission/supplement.zip`: PASS
- Fresh venv install from `requirements.txt`: PASS
- Fresh venv `python scripts/validate_release.py`: PASS
- Croissant validation: PASS
- `python scripts/check_submission_package.py --local-only`: PASS
- PDF page count: 18 total pages, letter paper
- PDF fonts: no Type 3 fonts
- Main-text figures: Figures 1--3 appear before References
- Artifact trace count: 396 total trace files, 324 main plus 72 calibration

External gates still required:

- Host `submission/supplement.zip` at an anonymous reviewer-accessible URL.
- Replace `submission/anonymous_artifact_url.txt` with that URL.
- Run `python scripts/check_submission_package.py` without `--local-only`; it should pass only after the URL is real and reachable.
- Upload `submission/main.pdf`, `submission/supplement.zip`, `submission/croissant.json`, and the hosted URL in OpenReview.
- Fill `submission/final_record_template.md` after submission.
