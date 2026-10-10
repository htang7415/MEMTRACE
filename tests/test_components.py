from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from memtrace import components
from memtrace.components import read_lock, static_checks


def test_lock_parses_and_names_every_digested_image() -> None:
    lock = read_lock()
    assert lock["LLMD_ROUTER_TAG"].startswith("v")
    for key in lock:
        if key.endswith("_DIGEST"):
            assert key.removesuffix("_DIGEST") + "_IMAGE" in lock
            assert lock[key].startswith("sha256:")


def test_repository_matches_the_lock() -> None:
    checks = static_checks(read_lock())
    assert any(c.component.startswith("epp-plugins/go.mod") for c in checks)
    assert len(checks) > 5  # the manifests and compose files were scanned
    assert [c for c in checks if not c.ok] == []


def test_a_drifted_manifest_fails(tmp_path: Path) -> None:
    root = Path(components.ROOT)
    shutil.copytree(root / "deploy", tmp_path / "deploy")
    shutil.copytree(root / "scripts", tmp_path / "scripts")
    (tmp_path / "experiments").mkdir()
    (tmp_path / "epp-plugins").mkdir()
    shutil.copy(root / "epp-plugins" / "go.mod", tmp_path / "epp-plugins" / "go.mod")
    relay = tmp_path / "deploy" / "kind" / "gpu-relay.yaml"
    relay.write_text(relay.read_text().replace("alpine/socat:1.8.0.3", "alpine/socat:1.9.0"))
    failed = [c for c in static_checks(read_lock(), tmp_path) if not c.ok]
    assert [(c.expected, c.found) for c in failed] == [("1.8.0.3", "1.9.0")]


def test_malformed_lock_line_is_an_error(tmp_path: Path) -> None:
    lock = tmp_path / "components.lock"
    lock.write_text("# comment\nA=1\nnot a pair\n")
    with pytest.raises(ValueError, match="expected KEY=value"):
        read_lock(lock)
    lock.write_text("A=1\nA=2\n")
    with pytest.raises(ValueError, match="set twice"):
        read_lock(lock)
