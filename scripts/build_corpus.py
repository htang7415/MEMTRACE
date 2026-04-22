import _bootstrap  # noqa: F401

from memtrace.config import ALLOWLIST_PATH, PASSAGES_PATH
from memtrace.corpus import build_allowlist, build_corpus, save_json, save_jsonl


def main() -> None:
    corpus = build_corpus()
    allowlist = build_allowlist(corpus)
    save_jsonl(PASSAGES_PATH, corpus)
    save_json(ALLOWLIST_PATH, allowlist)
    print(f"wrote_passages={len(corpus)}")
    print(f"wrote_allowlist={len(allowlist)}")


if __name__ == "__main__":
    main()
