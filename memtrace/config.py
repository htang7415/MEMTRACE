"""Centralized defaults for the MEMTRACE harness."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
EPISODES_DIR = DATA_DIR / "episodes"
GOLD_DIR = DATA_DIR / "gold"
INDICES_DIR = DATA_DIR / "indices"
RESULTS_DIR = DATA_DIR / "results"
TRACES_DIR = DATA_DIR / "traces"
AUDIT_DIR = DATA_DIR / "audit"
FIGURES_DIR = ROOT / "figures"
DOCS_DIR = ROOT / "docs"
PROMPTS_DIR = ROOT / "memtrace" / "prompts"
RELEASE_DIR = ROOT / "release"

SQLITE_PATH = DATA_DIR / "memtrace.sqlite3"
PASSAGES_PATH = CORPUS_DIR / "passages.jsonl"
ALLOWLIST_PATH = CORPUS_DIR / "allowlist.json"
TASKS_PATH = GOLD_DIR / "tasks.json"
LABELS_PATH = GOLD_DIR / "labels.json"
EPISODES_PATH = EPISODES_DIR / "episodes.json"
RETRIEVAL_VERIFICATION_PATH = INDICES_DIR / "verification.json"
RUN_SUMMARY_PATH = RESULTS_DIR / "run_summary.json"
EPISODE_SCORES_PATH = RESULTS_DIR / "episode_scores.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"
TABLE1_MD_PATH = RESULTS_DIR / "table1.md"
SUPPLEMENTARY_TABLES_MD_PATH = RESULTS_DIR / "supplementary_tables.md"
FIGURE1_PATH = FIGURES_DIR / "figure1_pipeline.svg"
FIGURE2_PATH = FIGURES_DIR / "figure2_ovr_vs_svr.svg"
FIGURE3_PATH = FIGURES_DIR / "figure3_par_by_task_family.svg"
FIGURE4_PATH = FIGURES_DIR / "figure4_one_shot_vs_stateful.svg"
FIGURE5_PATH = FIGURES_DIR / "figure5_violation_rate_by_horizon.svg"
RELEASE_MANIFEST_PATH = RELEASE_DIR / "manifest.json"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
ACTOR_MODELS = (
    "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
)
TEMPERATURE = 0
TOP_P = 1
TOP_K = 5
MAX_MEMORY_CANDIDATES = 3
MAX_MEMORY_CONTENT_CHARS = 200
