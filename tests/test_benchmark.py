from collections import Counter

from memtrace.config import ACTOR_MODELS
from memtrace.benchmark import build_episode_records, build_gold_labels, build_task_records
from memtrace.corpus import build_allowlist, build_corpus


def test_benchmark_has_twelve_tasks() -> None:
    tasks = build_task_records()
    assert len(tasks) == 12


def test_benchmark_has_expected_episode_counts_per_system() -> None:
    episodes = build_episode_records()
    counts = Counter((episode.actor_model, episode.system) for episode in episodes)
    assert counts == {(actor_model, system): 108 for actor_model in ACTOR_MODELS for system in ("S0", "S1", "S2")}


def test_stateful_delta_seven_repeats_first_filler_at_turn_six() -> None:
    episodes = build_episode_records()
    episode = next(
        item
        for item in episodes
        if item.system == "S0"
        and item.task_id == "approval-limit-rule"
        and item.episode_kind == "stateful_attack"
        and item.horizon == 7
        and item.payload_type == "direct_override"
    )
    assert len(episode.turns) == 8
    assert episode.turns[1] == episode.turns[6]


def test_gold_labels_cover_all_tasks() -> None:
    tasks = build_task_records()
    labels = build_gold_labels()
    assert {task.task_id for task in tasks} == {label.task_id for label in labels}


def test_corpus_matches_spec_shape() -> None:
    corpus = build_corpus()
    allowlist = build_allowlist(corpus)
    assert len(corpus) == 200
    assert len(allowlist) == 20


def test_corpus_passages_match_word_count_contract() -> None:
    for passage in build_corpus():
        word_count = len(passage["text"].split())
        assert 50 <= word_count <= 150, passage["source_id"]
