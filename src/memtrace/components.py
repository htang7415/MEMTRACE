"""The external components MEMTRACE runs against: components.lock, and checks that the machine and the repository
match it.

    memtrace components check            # installed vllm-metal, images, models and tools vs the lock
    memtrace components check --static   # only the repository: deploy/ and scripts/ name the locked versions

The lock is a KEY=value file so shell scripts can source it (scripts/images.sh); this module parses the same file.
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "components.lock"
_LINE = re.compile(r"^([A-Z][A-Z0-9_]*)=(\S+)$")
# Images are referenced as repository:tag; the lock pins one tag per repository.
_IMAGE_REF = re.compile(r"(?:image:\s*|IMAGE:-|IMAGE=\")([\w.\-/]+/[\w.\-]+):([\w.\-]+)")
_REVISION_REF = re.compile(r"--revision[\s,]+([0-9a-f]{40})")
_MODELS = {
    "QWEN3_0_6B_REVISION": "models--Qwen--Qwen3-0.6B",
    "QWEN3_4B_4BIT_REVISION": "models--mlx-community--Qwen3-4B-4bit",
    "QWEN3_8B_4BIT_REVISION": "models--mlx-community--Qwen3-8B-4bit",
}


def read_lock(path: Path = LOCK) -> dict[str, str]:
    """KEY -> value from the lock; comments and blank lines are skipped, anything else is an error."""
    out: dict[str, str] = {}
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            raise ValueError(f"{path}:{n}: expected KEY=value, got {raw!r}")
        if m.group(1) in out:
            raise ValueError(f"{path}:{n}: {m.group(1)} set twice")
        out[m.group(1)] = m.group(2)
    return out


def vllm_metal_venv(dev: bool = False) -> Path:
    """The venv holding the locked vllm-metal (or, with `dev`, the development build under test)."""
    if dev:
        return Path(os.environ.get("VLLM_METAL_DEV_VENV", "~/.venv-vllm-metal-dev")).expanduser()
    return Path(os.environ.get("VLLM_METAL_VENV", "~/.venv-vllm-metal")).expanduser()


@dataclass(frozen=True)
class Check:
    component: str
    expected: str
    found: str
    ok: bool | None  # None: not checked (e.g. Docker not running)


def static_checks(lock: dict[str, str], root: Path = ROOT) -> list[Check]:
    """Version references in the repository that must agree with the lock."""
    checks: list[Check] = []
    go_mod = (root / "epp-plugins" / "go.mod").read_text(encoding="utf-8")
    m = re.search(r"github\.com/llm-d/llm-d-router (v[\w.\-]+)", go_mod)
    found = m.group(1) if m else "missing"
    checks.append(
        Check("epp-plugins/go.mod llm-d-router", lock["LLMD_ROUTER_TAG"], found, found == lock["LLMD_ROUTER_TAG"])
    )
    locked = {v.rsplit(":", 1)[0]: v.rsplit(":", 1)[1] for k, v in lock.items() if k.endswith("_IMAGE")}
    for path in sorted([*(root / "deploy").rglob("*.yaml"), *(root / "scripts").glob("*.sh")]):
        if path.name.startswith("._"):
            continue
        for repo, tag in _IMAGE_REF.findall(path.read_text(encoding="utf-8")):
            if repo in locked:
                checks.append(Check(f"{path.relative_to(root)} {repo}", locked[repo], tag, tag == locked[repo]))
    revisions = {v for k, v in lock.items() if k.endswith("_REVISION")}
    for path in sorted([*(root / "experiments").glob("*.yaml"), *(root / "scripts").glob("*.sh")]):
        if path.name.startswith("._"):
            continue
        for rev in _REVISION_REF.findall(path.read_text(encoding="utf-8")):
            checks.append(Check(f"{path.relative_to(root)} model revision", "a locked revision", rev, rev in revisions))
    return checks


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def installed_checks(lock: dict[str, str], hf_home: Path) -> list[Check]:
    """What is installed on this machine against the lock."""
    checks: list[Check] = []
    for prefix, dev in (("VLLM_METAL_", False), ("VLLM_METAL_DEV_", True)):
        python = vllm_metal_venv(dev) / "bin" / "python"
        for key, dist in ((f"{prefix}VERSION", "vllm-metal"), (f"{prefix}VLLM_VERSION", "vllm")):
            found = _run([str(python), "-c", f"import importlib.metadata as m; print(m.version({dist!r}))"])
            # A CPU/Metal build of vLLM reports a local version suffix (0.30.0+cpu).
            version = (found or "not installed").split("+")[0]
            name = f"{dist} ({python.parent.parent})"
            checks.append(Check(name, lock[key], found or "not installed", version == lock[key]))
    docker_up = _run(["docker", "info", "--format", "{{.ServerVersion}}"]) is not None
    for key, value in lock.items():
        if not key.endswith("_DIGEST"):
            continue
        image = lock[key.removesuffix("_DIGEST") + "_IMAGE"]
        if not docker_up:
            checks.append(Check(image, value, "Docker not running", None))
            continue
        digests = _run(["docker", "image", "inspect", image, "--format", '{{join .RepoDigests " "}}']) or ""
        pinned = f"@{value}" in digests
        checks.append(Check(image, value, "pinned" if pinned else "absent or drifted", pinned))
    for key, folder in _MODELS.items():
        present = (hf_home / "hub" / folder / "snapshots" / lock[key]).is_dir()
        name = folder.removeprefix("models--").replace("--", "/")
        checks.append(Check(name, lock[key], "present" if present else "missing", present))
    for key, cmd, pattern in (
        ("KIND_VERSION", ["kind", "version"], r"(v[\d.]+)"),
        ("KUBECTL_VERSION", ["kubectl", "version", "--client"], r"Client Version: (v[\d.]+)"),
        ("HELM_VERSION", ["helm", "version", "--short"], r"(v[\d.]+)"),
    ):
        out = _run(cmd) if shutil.which(cmd[0]) else None
        m = re.search(pattern, out or "")
        found = m.group(1) if m else "not installed"
        checks.append(Check(cmd[0], lock[key], found, found == lock[key]))
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("check",))
    parser.add_argument("--static", action="store_true", help="check only the repository, not this machine")
    args = parser.parse_args(argv)
    lock = read_lock()
    checks = static_checks(lock)
    if not args.static:
        checks += installed_checks(lock, Path(os.environ.get("HF_HOME", ROOT / "models" / "huggingface")))
    width = max(len(c.component) for c in checks)
    for c in checks:
        mark = {True: "ok  ", False: "FAIL", None: "skip"}[c.ok]
        print(f"{mark}  {c.component:<{width}}  expected {c.expected}  found {c.found}")
    failed = [c for c in checks if c.ok is False]
    ok, skipped = sum(c.ok is True for c in checks), sum(c.ok is None for c in checks)
    print(f"components.lock: {ok} match, {len(failed)} differ, {skipped} not checked")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
