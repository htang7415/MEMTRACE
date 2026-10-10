"""MEMTRACE: serving LLM agents efficiently: engines, routing, gateway, agent context, and KV-cache memory."""

import os
from pathlib import Path

__version__ = "0.6.0"

# In a source checkout, model weights live in the repository's git-ignored models/ folder rather than the user's
# Hugging Face cache; engines this package starts inherit the setting. An explicit HF_HOME wins.
_MODELS = Path(__file__).resolve().parents[2] / "models"
if _MODELS.is_dir():
    os.environ.setdefault("HF_HOME", str(_MODELS / "huggingface"))
