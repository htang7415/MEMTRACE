"""Typed runtime configuration for MEMTRACE."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_ACTOR_MODELS = (
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
)

_ENVIRONMENT_KEYS = {
    "root": "MEMTRACE_ROOT",
    "data_dir": "MEMTRACE_DATA_DIR",
    "figures_dir": "MEMTRACE_FIGURES_DIR",
    "planner_prompt_path": "MEMTRACE_PLANNER_PROMPT_PATH",
    "passages_path": "MEMTRACE_PASSAGES_PATH",
    "allowlist_path": "MEMTRACE_ALLOWLIST_PATH",
    "actor_models": "MEMTRACE_ACTOR_MODELS",
    "embedding_model": "MEMTRACE_EMBEDDING_MODEL",
    "retrieval_backend": "MEMTRACE_RETRIEVAL_BACKEND",
    "protocol_version": "MEMTRACE_PROTOCOL_VERSION",
    "memory_writer_backend": "MEMTRACE_MEMORY_WRITER_BACKEND",
    "planner_backend": "MEMTRACE_PLANNER_BACKEND",
    "temperature": "MEMTRACE_TEMPERATURE",
    "top_p": "MEMTRACE_TOP_P",
    "top_k": "MEMTRACE_TOP_K",
    "max_memory_candidates": "MEMTRACE_MAX_MEMORY_CANDIDATES",
    "max_memory_content_chars": "MEMTRACE_MAX_MEMORY_CONTENT_CHARS",
    "memory_writer_max_tokens": "MEMTRACE_MEMORY_WRITER_MAX_TOKENS",
    "planner_max_tokens": "MEMTRACE_PLANNER_MAX_TOKENS",
}


@dataclass(frozen=True)
class Settings:
    """Validated settings shared by CLI workflows and library code."""

    root: Path
    data_dir: Path
    figures_dir: Path
    planner_prompt_path: Path
    passages_path: Path
    allowlist_path: Path
    actor_models: tuple[str, ...]
    embedding_model: str
    retrieval_backend: str
    protocol_version: str
    memory_writer_backend: str
    planner_backend: str
    temperature: float
    top_p: float
    top_k: int
    max_memory_candidates: int
    max_memory_content_chars: int
    memory_writer_max_tokens: int
    planner_max_tokens: int

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "corpus"

    @property
    def episodes_dir(self) -> Path:
        return self.data_dir / "episodes"

    @property
    def gold_dir(self) -> Path:
        return self.data_dir / "gold"

    @property
    def indices_dir(self) -> Path:
        return self.data_dir / "indices"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def traces_dir(self) -> Path:
        return self.data_dir / "traces"

    @property
    def audit_dir(self) -> Path:
        return self.data_dir / "audit"

    @classmethod
    def load(
        cls,
        config_path: Path | None = None,
        *,
        environ: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> "Settings":
        environment = os.environ if environ is None else environ
        working_dir = (cwd or Path.cwd()).resolve()
        configured = _load_toml(config_path)

        def value(name: str, default: Any) -> Any:
            environment_key = _ENVIRONMENT_KEYS[name]
            if environment_key in environment:
                return environment[environment_key]
            return configured.get(name, default)

        root = _resolve_path(value("root", working_dir), working_dir)
        data_dir = _resolve_path(value("data_dir", "data"), root)
        figures_dir = _resolve_path(value("figures_dir", "figures"), root)
        prompts_dir = Path(__file__).resolve().parent / "prompts"
        planner_prompt_path = _resolve_path(value("planner_prompt_path", prompts_dir / "planner.txt"), root)
        passages_path = _resolve_path(value("passages_path", data_dir / "corpus" / "passages.jsonl"), root)
        allowlist_path = _resolve_path(value("allowlist_path", data_dir / "corpus" / "allowlist.json"), root)
        actor_models = _actor_models(value("actor_models", DEFAULT_ACTOR_MODELS))
        if not actor_models:
            raise ValueError("actor_models must contain at least one model")
        retrieval_backend = str(value("retrieval_backend", "dense"))
        if retrieval_backend not in {"dense", "lexical"}:
            raise ValueError("retrieval_backend must be 'dense' or 'lexical'")

        return cls(
            root=root,
            data_dir=data_dir,
            figures_dir=figures_dir,
            planner_prompt_path=planner_prompt_path,
            passages_path=passages_path,
            allowlist_path=allowlist_path,
            actor_models=actor_models,
            embedding_model=str(value("embedding_model", "BAAI/bge-small-en-v1.5")),
            retrieval_backend=retrieval_backend,
            protocol_version=str(value("protocol_version", "project-md-v3")),
            memory_writer_backend=str(value("memory_writer_backend", "mlx")),
            planner_backend=str(value("planner_backend", "mlx")),
            temperature=float(value("temperature", 0)),
            top_p=float(value("top_p", 1)),
            top_k=int(value("top_k", 5)),
            max_memory_candidates=int(value("max_memory_candidates", 3)),
            max_memory_content_chars=int(value("max_memory_content_chars", 200)),
            memory_writer_max_tokens=int(value("memory_writer_max_tokens", 512)),
            planner_max_tokens=int(value("planner_max_tokens", 256)),
        )


def _load_toml(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    path = config_path.expanduser().resolve()
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    configured = payload.get("memtrace", {})
    if not isinstance(configured, dict):
        raise ValueError(f"{path}: [memtrace] must be a TOML table")
    unknown = sorted(set(configured) - set(_ENVIRONMENT_KEYS))
    if unknown:
        raise ValueError(f"{path}: unknown MEMTRACE settings: {', '.join(unknown)}")
    return configured


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _actor_models(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item) for item in value)


SETTINGS: Settings
ROOT: Path
DATA_DIR: Path
CORPUS_DIR: Path
EPISODES_DIR: Path
GOLD_DIR: Path
INDICES_DIR: Path
RESULTS_DIR: Path
TRACES_DIR: Path
AUDIT_DIR: Path
FIGURES_DIR: Path
PROMPTS_DIR: Path
DEFAULT_PLANNER_PROMPT_PATH: Path
PLANNER_PROMPT_PATH: Path
SQLITE_PATH: Path
PASSAGES_PATH: Path
ALLOWLIST_PATH: Path
TASKS_PATH: Path
LABELS_PATH: Path
EPISODES_PATH: Path
RETRIEVAL_VERIFICATION_PATH: Path
DENSE_INDEX_PATH: Path
RUN_SUMMARY_PATH: Path
EPISODE_SCORES_PATH: Path
METRICS_PATH: Path
TABLE1_MD_PATH: Path
SUPPLEMENTARY_TABLES_MD_PATH: Path
ATTRIBUTION_LABELS_PATH: Path
ATTRIBUTION_REPORT_PATH: Path
AUDIT_SAMPLE_PATH: Path
AUDIT_TEMPLATE_PATH: Path
AUDIT_REVIEW_MD_PATH: Path
AUDIT_REPORT_JSON_PATH: Path
AUDIT_REPORT_MD_PATH: Path
EMBEDDING_MODEL: str
RETRIEVAL_BACKEND: str
ACTOR_MODELS: tuple[str, ...]
PROTOCOL_VERSION: str
MEMORY_WRITER_BACKEND: str
PLANNER_BACKEND: str
TEMPERATURE: float
TOP_P: float
TOP_K: int
MAX_MEMORY_CANDIDATES: int
MAX_MEMORY_CONTENT_CHARS: int
MEMORY_WRITER_MAX_TOKENS: int
PLANNER_MAX_TOKENS: int


def activate(settings: Settings) -> None:
    """Activate settings before importing a CLI workflow module."""

    global SETTINGS
    global ROOT, DATA_DIR, CORPUS_DIR, EPISODES_DIR, GOLD_DIR, INDICES_DIR, RESULTS_DIR, TRACES_DIR, AUDIT_DIR
    global FIGURES_DIR, PROMPTS_DIR, DEFAULT_PLANNER_PROMPT_PATH, PLANNER_PROMPT_PATH
    global SQLITE_PATH, PASSAGES_PATH, ALLOWLIST_PATH, TASKS_PATH, LABELS_PATH, EPISODES_PATH
    global RETRIEVAL_VERIFICATION_PATH, DENSE_INDEX_PATH, RUN_SUMMARY_PATH, EPISODE_SCORES_PATH, METRICS_PATH
    global TABLE1_MD_PATH, SUPPLEMENTARY_TABLES_MD_PATH, ATTRIBUTION_LABELS_PATH, ATTRIBUTION_REPORT_PATH
    global AUDIT_SAMPLE_PATH, AUDIT_TEMPLATE_PATH, AUDIT_REVIEW_MD_PATH, AUDIT_REPORT_JSON_PATH, AUDIT_REPORT_MD_PATH
    global EMBEDDING_MODEL, RETRIEVAL_BACKEND, ACTOR_MODELS, PROTOCOL_VERSION
    global MEMORY_WRITER_BACKEND, PLANNER_BACKEND
    global TEMPERATURE, TOP_P, TOP_K, MAX_MEMORY_CANDIDATES, MAX_MEMORY_CONTENT_CHARS
    global MEMORY_WRITER_MAX_TOKENS, PLANNER_MAX_TOKENS

    SETTINGS = settings
    ROOT = settings.root
    DATA_DIR = settings.data_dir
    CORPUS_DIR = settings.corpus_dir
    EPISODES_DIR = settings.episodes_dir
    GOLD_DIR = settings.gold_dir
    INDICES_DIR = settings.indices_dir
    RESULTS_DIR = settings.results_dir
    TRACES_DIR = settings.traces_dir
    AUDIT_DIR = settings.audit_dir
    FIGURES_DIR = settings.figures_dir
    PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
    DEFAULT_PLANNER_PROMPT_PATH = PROMPTS_DIR / "planner.txt"
    PLANNER_PROMPT_PATH = settings.planner_prompt_path

    SQLITE_PATH = DATA_DIR / "memtrace.sqlite3"
    PASSAGES_PATH = settings.passages_path
    ALLOWLIST_PATH = settings.allowlist_path
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

    EMBEDDING_MODEL = settings.embedding_model
    RETRIEVAL_BACKEND = settings.retrieval_backend
    ACTOR_MODELS = settings.actor_models
    PROTOCOL_VERSION = settings.protocol_version
    MEMORY_WRITER_BACKEND = settings.memory_writer_backend
    PLANNER_BACKEND = settings.planner_backend
    TEMPERATURE = settings.temperature
    TOP_P = settings.top_p
    TOP_K = settings.top_k
    MAX_MEMORY_CANDIDATES = settings.max_memory_candidates
    MAX_MEMORY_CONTENT_CHARS = settings.max_memory_content_chars
    MEMORY_WRITER_MAX_TOKENS = settings.memory_writer_max_tokens
    PLANNER_MAX_TOKENS = settings.planner_max_tokens


activate(Settings.load())
