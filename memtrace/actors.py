"""Actor-model metadata."""

from __future__ import annotations

from memtrace.config import ACTOR_MODELS


ACTOR_MODEL_SLUGS = {
    "mlx-community/Qwen2.5-3B-Instruct-4bit": "qwen25_3b",
    "mlx-community/Llama-3.2-3B-Instruct-4bit": "llama32_3b",
    "mlx-community/Qwen2.5-7B-Instruct-4bit": "qwen25_7b",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit": "llama31_8b",
}

SLUG_TO_ACTOR_MODEL = {slug: model for model, slug in ACTOR_MODEL_SLUGS.items()}


def actor_model_slug(actor_model: str) -> str:
    return ACTOR_MODEL_SLUGS.get(actor_model, actor_model.replace("/", "_").replace("-", "_"))


def all_actor_models() -> tuple[str, ...]:
    return ACTOR_MODELS
