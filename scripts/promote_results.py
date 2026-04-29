try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass

import argparse
import json
import shutil
from pathlib import Path

from memtrace.config import FIGURES_DIR, RESULTS_DIR


RESULT_FILES = (
    "run_summary.json",
    "episode_scores.json",
    "metrics.json",
    "table1.md",
    "supplementary_tables.md",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a validated result directory to paper-facing outputs.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    summary_count = promote_results(args.source_dir, args.results_dir, args.figures_dir)
    print(f"promoted_source={args.source_dir}")
    print(f"promoted_episodes={summary_count}")
    print(f"results_dir={args.results_dir}")
    print(f"figures_dir={args.figures_dir}")


def promote_results(source_dir: Path, results_dir: Path, figures_dir: Path) -> int:
    _validate_source(source_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for filename in RESULT_FILES:
        shutil.copy2(source_dir / filename, results_dir / filename)

    source_figures = source_dir / "figures"
    if source_figures.exists():
        for source in source_figures.glob("*.svg"):
            if not source.name.startswith("._"):
                shutil.copy2(source, figures_dir / source.name)

    with (source_dir / "run_summary.json").open("r", encoding="utf-8") as handle:
        return len(json.load(handle))


def _validate_source(source_dir: Path) -> None:
    missing = [filename for filename in RESULT_FILES if not (source_dir / filename).exists()]
    if missing:
        raise SystemExit(f"missing required result files in {source_dir}: {', '.join(missing)}")

    with (source_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    validation = metrics.get("pilot_validation_by_configuration", {})
    if not validation:
        raise SystemExit(f"{source_dir} has no pilot validation block")
    invalid = [
        f"{_short_actor_model(actor_model)}:{system}"
        for actor_model, rows in validation.items()
        for system, row in rows.items()
        if not row.get("official_pilot_valid", False)
    ]
    if invalid:
        raise SystemExit(f"cannot promote pilot-invalid result rows: {', '.join(invalid)}")


def _short_actor_model(actor_model: str) -> str:
    if "Qwen2.5-7B" in actor_model:
        return "Qwen2.5-7B"
    if "Qwen2.5-3B" in actor_model:
        return "Qwen2.5-3B"
    if "Llama-3.1-8B" in actor_model:
        return "Llama-3.1-8B"
    if "Llama-3.2-3B" in actor_model:
        return "Llama-3.2-3B"
    return actor_model


if __name__ == "__main__":
    main()
