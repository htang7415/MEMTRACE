"""Centralized defaults for the MEMTRACE harness."""

import os
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
DEFAULT_PLANNER_PROMPT_PATH = PROMPTS_DIR / "planner.txt"
PLANNER_PROMPT_PATH = Path(os.environ.get("MEMTRACE_PLANNER_PROMPT_PATH", DEFAULT_PLANNER_PROMPT_PATH))
RELEASE_DIR = ROOT / "release"

SQLITE_PATH = DATA_DIR / "memtrace.sqlite3"
PASSAGES_PATH = Path(os.environ.get("MEMTRACE_PASSAGES_PATH", CORPUS_DIR / "passages.jsonl"))
ALLOWLIST_PATH = Path(os.environ.get("MEMTRACE_ALLOWLIST_PATH", CORPUS_DIR / "allowlist.json"))
TASKS_PATH = GOLD_DIR / "tasks.json"
LABELS_PATH = GOLD_DIR / "labels.json"
EPISODES_PATH = EPISODES_DIR / "episodes.json"
RETRIEVAL_VERIFICATION_PATH = INDICES_DIR / "verification.json"
DENSE_INDEX_PATH = INDICES_DIR / "bge-small-en-v1.5.npz"
RUN_SUMMARY_PATH = RESULTS_DIR / "run_summary.json"
EPISODE_SCORES_PATH = RESULTS_DIR / "episode_scores.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"
TABLE1_MD_PATH = RESULTS_DIR / "table1.md"
SUPPLEMENTARY_TABLES_MD_PATH = RESULTS_DIR / "supplementary_tables.md"
ATTRIBUTION_LABELS_PATH = RESULTS_DIR / "attribution_labels.json"
ATTRIBUTION_REPORT_PATH = RESULTS_DIR / "attribution_report.md"
AUDIT_SAMPLE_PATH = AUDIT_DIR / "audit_sample.json"
AUDIT_TEMPLATE_PATH = AUDIT_DIR / "audit_template.jsonl"
AUDIT_REVIEW_MD_PATH = AUDIT_DIR / "audit_review.md"
AUDIT_REPORT_JSON_PATH = AUDIT_DIR / "audit_report.json"
AUDIT_REPORT_MD_PATH = AUDIT_DIR / "audit_report.md"
FIGURE1_PATH = FIGURES_DIR / "figure1_pipeline.svg"
FIGURE2_PATH = FIGURES_DIR / "figure2_ovr_vs_svr.svg"
FIGURE3_PATH = FIGURES_DIR / "figure3_par_by_task_family.svg"
FIGURE4_PATH = FIGURES_DIR / "figure4_one_shot_vs_stateful.svg"
FIGURE5_PATH = FIGURES_DIR / "figure5_violation_rate_by_horizon.svg"
RELEASE_MANIFEST_PATH = RELEASE_DIR / "manifest.json"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
ACTOR_MODELS = (
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
)
PROTOCOL_VERSION = "project-md-v3"
MEMORY_WRITER_BACKEND = os.environ.get("MEMTRACE_MEMORY_WRITER_BACKEND", "mlx")
PLANNER_BACKEND = os.environ.get("MEMTRACE_PLANNER_BACKEND", "mlx")
TEMPERATURE = 0
TOP_P = 1
TOP_K = 5
MAX_MEMORY_CANDIDATES = 3
MAX_MEMORY_CONTENT_CHARS = 200
MEMORY_WRITER_MAX_TOKENS = 512
PLANNER_MAX_TOKENS = 256
