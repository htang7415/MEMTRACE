import _bootstrap  # noqa: F401

from memtrace.config import PASSAGES_PATH
from memtrace.retrieval import load_passages


def main() -> None:
    passages = load_passages(PASSAGES_PATH)
    print(f"loaded_passages={len(passages)}")


if __name__ == "__main__":
    main()
