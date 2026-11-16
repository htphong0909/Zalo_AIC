"""Unit tests for GiGPO Group-in-Group advantage calculations."""
from src.agent.lfm3_gigpo import (
    TrajectoryStepInfo,
    EpisodeRollout,
    compute_gigpo_advantages,
    compute_official_episode_reward,
)
from src.core.dataset import load_catalog


def test_gigpo_advantages_distinguishes_good_from_bad_rollouts():
    # AnchorState: (sample_id, state_hash, executed_reads)
    anchor_0 = ("s1", "hash_init", frozenset())
    anchor_1 = ("s1", "hash_read_done", frozenset([("contacts.search", '{"query":"Lan"}')]))

    # Episode 1: Successful rollout (reward = 2.35)
    ep1 = EpisodeRollout(
        sample_id="s1",
        steps=[
            TrajectoryStepInfo(step_idx=0, anchor_state=anchor_0),
            TrajectoryStepInfo(step_idx=1, anchor_state=anchor_1),
        ],
        episode_reward=2.35,
    )

    # Episode 2: Failed rollout on step 1 (reward = -0.5)
    ep2 = EpisodeRollout(
        sample_id="s1",
        steps=[
            TrajectoryStepInfo(step_idx=0, anchor_state=anchor_0),
            TrajectoryStepInfo(step_idx=1, anchor_state=anchor_1),
        ],
        episode_reward=-0.5,
    )

    episodes = compute_gigpo_advantages([ep1, ep2], omega=1.0)

    # Episode 1 should have positive macro advantage, Episode 2 negative
    assert ep1.steps[0].macro_advantage > 0.0
    assert ep2.steps[0].macro_advantage < 0.0

    # Step 1 micro advantage distinguishes the good action from the bad action from anchor_1
    assert ep1.steps[1].micro_advantage > 0.0
    assert ep2.steps[1].micro_advantage < 0.0

    # Combined advantage aligns with success
    assert ep1.steps[1].combined_advantage > ep2.steps[1].combined_advantage


def test_compute_official_episode_reward():
    catalog = load_catalog("train", "data")
    pred = [{"name": "alarm.create", "arguments": {"time": "06:30"}}]
    gold = [{"name": "alarm.create", "arguments": {"time": "06:30"}}]

    # Exact state match -> high reward
    score, f1, state_match, ep_reward = compute_official_episode_reward(
        pred, "hash_match", gold, "hash_match", catalog
    )
    assert score == 1.0
    assert f1 == 1.0
    assert state_match is True
    assert ep_reward >= 2.35

    # State mismatch -> zero official score
    score_m, f1_m, state_match_m, ep_reward_m = compute_official_episode_reward(
        pred, "hash_wrong", gold, "hash_gold", catalog
    )
    assert score_m == 0.0
    assert f1_m == 1.0
    assert state_match_m is False
    assert ep_reward_m < ep_reward
