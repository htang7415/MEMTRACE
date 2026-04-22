from pathlib import Path

from scripts.export_release import build_manifest


def test_build_manifest_skips_appledouble_files(tmp_path: Path) -> None:
    (tmp_path / "ok.txt").write_text("x", encoding="utf-8")
    (tmp_path / "._skip.txt").write_text("y", encoding="utf-8")
    manifest = build_manifest(tmp_path)
    assert manifest["files"] == [{"path": "ok.txt", "bytes": 1}]
