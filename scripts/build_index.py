import _bootstrap  # noqa: F401

import json

from memtrace.config import DENSE_INDEX_PATH, EMBEDDING_MODEL, PASSAGES_PATH
from memtrace.retrieval import load_passages


def main() -> None:
    passages = load_passages(PASSAGES_PATH)
    if not passages:
        raise SystemExit(f"no passages found at {PASSAGES_PATH}")

    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "building the dense index requires `sentence-transformers` and `numpy`; "
            "install project dependencies before running this script"
        ) from exc

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [item["text"] for item in passages]
    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    DENSE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DENSE_INDEX_PATH,
        source_ids=np.array([item["source_id"] for item in passages]),
        embeddings=embeddings,
        metadata=np.array([json.dumps(item) for item in passages]),
        model=np.array([EMBEDDING_MODEL]),
    )
    print(f"loaded_passages={len(passages)}")
    print(f"embedding_model={EMBEDDING_MODEL}")
    print(f"index_path={DENSE_INDEX_PATH}")


if __name__ == "__main__":
    main()
