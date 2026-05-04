import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAUSAL_FIELDS = {
    "poison_in_initial_retrieval",
    "writer_candidate_emitted",
    "writer_candidate_valid_schema",
    "poison_admitted",
    "admitted_memory_retrieved_at_trigger",
    "current_turn_poison_retrieved_at_trigger",
    "unsafe_tool_call_proposed_before_checker",
    "policy_checker_blocked_unsafe_call",
    "unsafe_tool_call_executed",
    "execution_failure",
}
AUTHOR_MARKERS = tuple(
    marker.strip()
    for marker in os.environ.get("MEMTRACE_ANONYMITY_MARKERS", "").split(",")
    if marker.strip()
)
PERSONAL_PATH_MARKERS = ("/" + "Users" + "/", "/" + "Volumes" + "/", "\\" + "Users" + "\\")
IGNORED_SCAN_DIRS = {".git", ".venv", "venv", "__pycache__", "artifacts"}


def main() -> None:
    checks = [
        _required_files_present(),
        _main_traces_present(),
        _calibration_count_matches_reported_scope(),
        _run_summary_episode_ids_unique(),
        _trace_episode_ids_unique(),
        _trace_rows_have_schema_version(),
        _episode_metadata_is_valid(),
        _stateful_scores_have_causal_diagnostics(),
        _generated_table_matches_metrics(),
        _third_party_assets_documented(),
        _artifact_is_anonymous(),
    ]
    failed = [message for ok, message in checks if not ok]
    if failed:
        for message in failed:
            print(f"FAIL {message}")
        raise SystemExit(1)
    for _, message in checks:
        print(f"PASS {message}")


def _required_files_present() -> tuple[bool, str]:
    required = [
        "README.md",
        "VALIDATION.md",
        "REPRODUCE.md",
        "RELEASE_MANIFEST.md",
        "TRACE_SCHEMA.md",
        "DATASET_CARD.md",
        "EVAL_CARD.md",
        "EVALUATION_CARD.md",
        "LICENSE",
        "THIRD_PARTY_ASSETS.md",
        "croissant_metadata.json",
        "croissant.json",
        "environment.yml",
        "requirements.txt",
        "requirements-inference.txt",
        "pyproject.toml",
        "configs/qwen_actor_backend.yaml",
        "configs/scoring.yaml",
        "configs/retrieval.yaml",
        "data/corpus/passages.jsonl",
        "data/allowlist/allowlist.json",
        "data/episodes/episodes.json",
        "data/gold_labels/tasks.json",
        "data/gold_labels/labels.json",
        "results/run_summary.json",
        "results/episode_scores.json",
        "results/metrics.json",
        "results/calibration_oracle_memory_run_summary.json",
        "results/calibration_oracle_memory_episode_scores.json",
        "results/calibration_oracle_memory_metrics.json",
        "tables/main_metrics.json",
        "tables/calibration_metrics.json",
    ]
    missing = [name for name in required if not (ROOT / name).exists()]
    return not missing, "required files present" if not missing else "missing " + ", ".join(missing)


def _main_traces_present() -> tuple[bool, str]:
    traces = _main_trace_paths()
    return len(traces) == 324, f"{len(traces)}/324 main traces present"


def _calibration_count_matches_reported_scope() -> tuple[bool, str]:
    traces = _calibration_trace_paths()
    metrics_path = ROOT / "results" / "calibration_oracle_memory_metrics.json"
    if not metrics_path.exists():
        return len(traces) == 0, f"{len(traces)} calibration traces with no calibration metrics"
    metrics = _load_json(metrics_path)
    rows = metrics.get("calibration_by_condition", {})
    expected = sum(row.get("episodes", 0) for row in rows.values())
    return len(traces) == expected, f"{len(traces)}/{expected} calibration traces present"


def _run_summary_episode_ids_unique() -> tuple[bool, str]:
    rows = _load_all_run_summary_rows()
    episode_ids = [row["episode_id"] for row in rows]
    return len(episode_ids) == len(set(episode_ids)), "run-summary episode IDs are unique"


def _trace_episode_ids_unique() -> tuple[bool, str]:
    seen = {}
    bad = []
    for path in _all_trace_paths():
        rows = _load_jsonl(path)
        ids = {row.get("episode_id") for row in rows}
        if len(ids) != 1 or None in ids:
            bad.append(path.name)
            continue
        episode_id = next(iter(ids))
        if episode_id in seen:
            bad.append(f"{path.name} duplicates {seen[episode_id]}")
            continue
        seen[episode_id] = path.name
    return not bad, f"{len(seen)} trace episode IDs are unique" if not bad else "bad trace episode IDs in " + ", ".join(bad[:5])


def _trace_rows_have_schema_version() -> tuple[bool, str]:
    missing = []
    for path in _all_trace_paths():
        for row in _load_jsonl(path):
            if not row.get("schema_version"):
                missing.append(path.name)
                break
    return not missing, "trace rows have schema_version" if not missing else "schema_version missing in " + ", ".join(missing[:5])


def _episode_metadata_is_valid() -> tuple[bool, str]:
    rows = _load_all_run_summary_rows()
    bad = []
    for row in rows:
        if row["episode_kind"] == "stateful_attack" and row.get("horizon") not in {1, 3, 7}:
            bad.append(row["episode_id"])
        if row["episode_kind"] != "clean_control" and row.get("payload_type") not in {"direct_override", "contextual_drift"}:
            bad.append(row["episode_id"])
    return not bad, "episode horizons and payload types are valid" if not bad else "bad episode metadata in " + ", ".join(bad[:5])


def _stateful_scores_have_causal_diagnostics() -> tuple[bool, str]:
    scores = _load_all_episode_scores()
    missing = [
        score["episode_id"]
        for score in scores
        if score.get("episode_kind") == "stateful_attack" and not CAUSAL_FIELDS.issubset(score.keys())
    ]
    return not missing, "stateful scores include causal-chain diagnostics" if not missing else "missing diagnostics in " + ", ".join(missing[:5])


def _generated_table_matches_metrics() -> tuple[bool, str]:
    metrics = _load_json(ROOT / "results" / "metrics.json")
    table_metrics = _load_json(ROOT / "tables" / "main_metrics.json")
    if metrics != table_metrics:
        return False, "tables/main_metrics.json differs from results/metrics.json"
    calibration_metrics = _load_json(ROOT / "results" / "calibration_oracle_memory_metrics.json")
    table_calibration_metrics = _load_json(ROOT / "tables" / "calibration_metrics.json")
    return (
        calibration_metrics == table_calibration_metrics,
        "generated metric tables match packaged JSON"
        if calibration_metrics == table_calibration_metrics
        else "tables/calibration_metrics.json differs from results/calibration_oracle_memory_metrics.json",
    )


def _third_party_assets_documented() -> tuple[bool, str]:
    assets_path = ROOT / "THIRD_PARTY_ASSETS.md"
    if not assets_path.exists():
        return False, "missing THIRD_PARTY_ASSETS.md"
    assets = assets_path.read_text(encoding="utf-8")
    required = [
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "c26a38f6",
        "mlx-lm",
        "0.18.1",
        "BAAI/bge-small-en-v1.5",
        "5c38ec7c",
        "sentence-transformers",
        "numpy",
        "pydantic",
        "pytest",
        "MEMTRACE synthetic corpus",
        "MEMTRACE traces",
    ]
    missing = [item for item in required if item not in assets]
    return not missing, "third-party assets documented" if not missing else "missing asset rows: " + ", ".join(missing)


def _artifact_is_anonymous() -> tuple[bool, str]:
    offenders = []
    markers = AUTHOR_MARKERS + PERSONAL_PATH_MARKERS
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in IGNORED_SCAN_DIRS for part in relative.parts):
            continue
        if not path.is_file() or path.name.startswith("._"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(marker in text for marker in markers):
            offenders.append(str(path.relative_to(ROOT)))
    return not offenders, "no author markers, personal paths, or local absolute paths" if not offenders else "anonymity markers in " + ", ".join(offenders[:5])


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _main_trace_paths() -> list[Path]:
    return sorted(path for path in (ROOT / "traces" / "v1_main_324").glob("*.jsonl") if not path.name.startswith("._"))


def _calibration_trace_paths() -> list[Path]:
    return sorted(
        path
        for path in (ROOT / "traces" / "calibration_oracle_memory_72").glob("*.jsonl")
        if not path.name.startswith("._")
    )


def _all_trace_paths() -> list[Path]:
    return _main_trace_paths() + _calibration_trace_paths()


def _load_all_run_summary_rows() -> list[dict]:
    rows = _load_json(ROOT / "results" / "run_summary.json")
    calibration_path = ROOT / "results" / "calibration_oracle_memory_run_summary.json"
    if calibration_path.exists():
        rows += _load_json(calibration_path)
    return rows


def _load_all_episode_scores() -> list[dict]:
    scores = _load_json(ROOT / "results" / "episode_scores.json")
    calibration_path = ROOT / "results" / "calibration_oracle_memory_episode_scores.json"
    if calibration_path.exists():
        scores += _load_json(calibration_path)
    return scores


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
