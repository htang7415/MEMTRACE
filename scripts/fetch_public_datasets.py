"""Download the public workload and evaluation datasets used by the platform roadmap.

Every source is pinned to an immutable revision. Files land in the git-ignored
``data/public/<dataset>/`` directory, and ``data/public/MANIFEST.json`` records the
URL, license, size, and SHA-256 of each file so runs can cite exactly what they used.

    python scripts/fetch_public_datasets.py              # everything (~690 MB, mostly ShareGPT)
    python scripts/fetch_public_datasets.py mooncake bfcl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

_MOONCAKE = "https://raw.githubusercontent.com/kvcache-ai/Mooncake/245e710604d46a14c46f8882e4381bd87bcd94ce/FAST25-release/traces"
_AZURE = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/215becdacba1ce682c7368642ce97d5a332de7a6/data"
_SHAREGPT = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/192ab2185289094fc556ec8ce5ce1e8e587154ca"
_BFCL = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"
_BFCL_FUNC_DOCS = [
    "gorilla_file_system",
    "math_api",
    "message_api",
    "posting_api",
    "ticket_api",
    "trading_bot",
    "travel_booking",
    "vehicle_control",
]

DATASETS: dict[str, dict] = {
    "mooncake": {
        "license": "Apache-2.0",
        "purpose": "Prefix-heavy production request traces (conversation, tool-agent, synthetic) with prefix-block hash IDs",
        "files": {
            name: f"{_MOONCAKE}/{name}"
            for name in ("conversation_trace.jsonl", "toolagent_trace.jsonl", "synthetic_trace.jsonl")
        },
    },
    "azure_llm": {
        "license": "CC-BY-4.0",
        "purpose": "Azure LLM inference arrival times and token counts (code and conversation services)",
        "files": {
            name: f"{_AZURE}/{name}" for name in ("AzureLLMInferenceTrace_code.csv", "AzureLLMInferenceTrace_conv.csv")
        },
    },
    "sharegpt": {
        "license": "Apache-2.0",
        "purpose": "Chat prompts; the standard input of `vllm bench serve`",
        "files": {
            "ShareGPT_V3_unfiltered_cleaned_split.json": f"{_SHAREGPT}/ShareGPT_V3_unfiltered_cleaned_split.json"
        },
    },
    "bfcl": {
        "license": "Apache-2.0",
        "purpose": "BFCL v3 single-call (simple, multiple) and multi-turn tool-calling cases and answers (quality gate)",
        "files": {
            **{
                f"{prefix}BFCL_v3_{category}.json": f"{_BFCL}/{prefix}BFCL_v3_{category}.json"
                for category in ("simple", "multiple")
                for prefix in ("", "possible_answer/")
            },
            "BFCL_v3_multi_turn_base.json": f"{_BFCL}/BFCL_v3_multi_turn_base.json",
            "possible_answer/BFCL_v3_multi_turn_base.json": f"{_BFCL}/possible_answer/BFCL_v3_multi_turn_base.json",
            **{f"multi_turn_func_doc/{doc}.json": f"{_BFCL}/multi_turn_func_doc/{doc}.json" for doc in _BFCL_FUNC_DOCS},
        },
    },
}


def _download(url: str, dest: Path) -> tuple[int, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as out:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    tmp.replace(dest)
    return dest.stat().st_size, digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", help=f"Subset to fetch: {', '.join(DATASETS)} (default: all)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/public"))
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist")
    args = parser.parse_args(argv)
    unknown = set(args.datasets) - set(DATASETS)
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")

    manifest_path = args.out_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in args.datasets or list(DATASETS):
        spec = DATASETS[name]
        entry = {"license": spec["license"], "purpose": spec["purpose"], "files": {}}
        for rel, url in spec["files"].items():
            dest = args.out_dir / name / rel
            if dest.exists() and not args.force:
                size, sha = dest.stat().st_size, _sha256(dest)
                status = "cached"
            else:
                size, sha = _download(url, dest)
                status = "downloaded"
            entry["files"][rel] = {"url": url, "bytes": size, "sha256": sha}
            print(f"{status:10} {name}/{rel}  {size / 1e6:.1f} MB")
        manifest[name] = entry
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"manifest   {manifest_path}  (free disk: {shutil.disk_usage(args.out_dir).free / 1e9:.0f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
