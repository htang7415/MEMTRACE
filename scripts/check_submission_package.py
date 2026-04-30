#!/usr/bin/env python3
"""Final local checks for the NeurIPS submission directory."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = ROOT / "submission"
REQUIRED_FILES = (
    "main.pdf",
    "supplement.zip",
    "anonymous_artifact_url.txt",
    "croissant.json",
    "checksums.txt",
    "title.txt",
    "abstract.txt",
)
REQUIRED_ZIP_MEMBERS = (
    "README.md",
    "VALIDATION.md",
    "REPRODUCE.md",
    "RELEASE_MANIFEST.md",
    "TRACE_SCHEMA.md",
    "THIRD_PARTY_ASSETS.md",
    "DATASET_CARD.md",
    "EVALUATION_CARD.md",
    "croissant.json",
    "LICENSE",
    "requirements.txt",
    "scripts/validate_release.py",
    "scripts/recompute_metrics.py",
    "scripts/make_figures.py",
    "data/results/run_summary.json",
    "data/calibration/s1_oracle_retrieved_memory/run_summary.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="allow the anonymous artifact URL placeholder while running local preflight checks",
    )
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    for name in REQUIRED_FILES:
        _check((SUBMISSION_DIR / name).is_file(), f"missing submission/{name}", errors)

    if errors:
        return _finish(errors, warnings)

    _check_no_sidecars(SUBMISSION_DIR, errors)
    _check_checksums(errors)
    _check_pdf(errors)
    _check_supplement_zip(errors)
    _check_url(args.local_only, errors, warnings)

    return _finish(errors, warnings)


def _check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _check_no_sidecars(root: Path, errors: list[str]) -> None:
    bad = [
        path.relative_to(ROOT)
        for path in root.rglob("*")
        if path.name == ".DS_Store" or path.name.startswith("._") or path.name == "__pycache__"
    ]
    _check(not bad, f"sidecar/cache files present: {bad[:5]}", errors)


def _check_checksums(errors: list[str]) -> None:
    expected: dict[Path, str] = {}
    for line in (SUBMISSION_DIR / "checksums.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, path_text = line.split(None, 1)
        expected[ROOT / path_text] = digest
    for path in (SUBMISSION_DIR / "main.pdf", SUBMISSION_DIR / "supplement.zip", SUBMISSION_DIR / "croissant.json"):
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        _check(expected.get(path) == actual, f"checksum mismatch for {path.relative_to(ROOT)}", errors)


def _check_pdf(errors: list[str]) -> None:
    pdf = SUBMISSION_DIR / "main.pdf"
    fonts = _run(["pdffonts", str(pdf)], errors, "pdffonts failed")
    if fonts:
        _check("Type 3" not in fonts.stdout, "main.pdf contains Type 3 fonts", errors)

    info = _run(["pdfinfo", str(pdf)], errors, "pdfinfo failed")
    if info:
        _check("Page size:       612 x 792 pts (letter)" in info.stdout, "main.pdf is not letter-sized", errors)

    text = _run(["pdftotext", str(pdf), "-"], errors, "pdftotext failed")
    if text:
        body = text.stdout
        refs = body.find("References")
        checklist = body.find("NeurIPS Paper Checklist")
        _check(refs >= 0, "References heading not found in main.pdf text", errors)
        _check(checklist > refs >= 0, "checklist does not appear after References", errors)
        for label in ("Figure 1:", "Figure 2:", "Figure 3:"):
            pos = body.find(label)
            _check(0 <= pos < refs, f"{label} does not appear before References", errors)
        pages = body.split("\f")
        ref_page = next((index + 1 for index, page in enumerate(pages) if "References" in page), None)
        _check(ref_page is not None and ref_page <= 9, "main content exceeds the 9-page limit before References", errors)


def _check_supplement_zip(errors: list[str]) -> None:
    zip_path = SUBMISSION_DIR / "supplement.zip"
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        for member in REQUIRED_ZIP_MEMBERS:
            _check(member in names, f"supplement.zip missing {member}", errors)
        bad = [
            name
            for name in names
            if Path(name).name == ".DS_Store" or Path(name).name.startswith("._") or "__pycache__" in Path(name).parts
        ]
        _check(not bad, f"supplement.zip contains sidecar/cache files: {bad[:5]}", errors)
        main_traces = [name for name in names if name.startswith("traces/v1_main_324/") and name.endswith(".jsonl")]
        calibration_traces = [
            name for name in names if name.startswith("traces/calibration_oracle_memory_72/") and name.endswith(".jsonl")
        ]
        _check(len(main_traces) == 324, f"supplement.zip has {len(main_traces)} main traces, expected 324", errors)
        _check(
            len(calibration_traces) == 72,
            f"supplement.zip has {len(calibration_traces)} calibration traces, expected 72",
            errors,
        )


def _check_url(local_only: bool, errors: list[str], warnings: list[str]) -> None:
    url = (SUBMISSION_DIR / "anonymous_artifact_url.txt").read_text(encoding="utf-8").strip()
    placeholder = not url or "TODO" in url or "replace" in url.lower()
    if placeholder and local_only:
        warnings.append("anonymous artifact URL is still a placeholder")
        return
    _check(not placeholder, "anonymous artifact URL is still a placeholder", errors)
    if placeholder:
        return
    _check(re.match(r"^https?://", url) is not None, "anonymous artifact URL must be http(s)", errors)
    if not errors:
        _check(_url_reachable(url), "anonymous artifact URL is not reachable", errors)


def _url_reachable(url: str) -> bool:
    for method in ("HEAD", "GET"):
        request = urllib.request.Request(url, method=method)
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return 200 <= response.status < 400
        except (urllib.error.URLError, TimeoutError, ValueError):
            continue
    return False


def _run(command: list[str], errors: list[str], label: str) -> subprocess.CompletedProcess[str] | None:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        errors.append(f"{label}: {result.stderr.strip()}")
        return None
    return result


def _finish(errors: list[str], warnings: list[str]) -> int:
    for warning in warnings:
        print(f"WARN {warning}")
    if errors:
        for error in errors:
            print(f"FAIL {error}")
        return 1
    print("PASS submission package local checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
