import os
import sys
from pathlib import Path


STRICT_PROMPT_PATH = Path("paper/reports/strict_refusal_planner_prompt.txt")
OUT_DIR = "data/pilot/qwen25_3b_strict_access_d1_v1"


def main(argv: list[str] | None = None) -> None:
    if not STRICT_PROMPT_PATH.exists():
        raise SystemExit(f"missing strict planner prompt: {STRICT_PROMPT_PATH}")

    os.environ["MEMTRACE_PLANNER_PROMPT_PATH"] = str(STRICT_PROMPT_PATH)

    from scripts import run_pilot

    base_args = [
        "--actor-model",
        "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "--task-id",
        "access-control-rule",
        "--system",
        "S0",
        "--system",
        "S2",
        "--episode-kind",
        "stateful_attack",
        "--horizon",
        "1",
        "--out-dir",
        OUT_DIR,
    ]
    run_pilot.main(base_args + list(argv or []))


if __name__ == "__main__":
    main(sys.argv[1:])
