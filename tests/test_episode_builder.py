from memtrace.schema import EpisodeRecord


def test_episode_record_turns_default() -> None:
    episode = EpisodeRecord(
        episode_id="e1",
        task_id="t1",
        family="policy_memory",
        episode_kind="clean_control",
        payload_type="clean_control",
        system="S0",
        horizon=1,
    )
    assert episode.turns == []
