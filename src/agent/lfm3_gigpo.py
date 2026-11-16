"""GiGPO (NeurIPS 2025) Group-in-Group Policy Optimization for multi-turn SafeEnv tool calling."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.eval.metrics import score_sample


@dataclass
class TrajectoryStepInfo:
    step_idx: int
    anchor_state: tuple[str, str, frozenset[tuple[str, str]]]
    step_reward: float = 0.0
    macro_advantage: float = 0.0
    micro_advantage: float = 0.0
    combined_advantage: float = 0.0


@dataclass
class EpisodeRollout:
    sample_id: str
    steps: list[TrajectoryStepInfo] = field(default_factory=list)
    pred_calls: list[dict[str, Any]] = field(default_factory=list)
    pred_hash: str = ""
    gold_calls: list[dict[str, Any]] = field(default_factory=list)
    gold_hash: str = ""
    f1: float = 0.0
    state_match: bool = False
    official_score: float = 0.0
    episode_reward: float = 0.0


def compute_official_episode_reward(
    pred_calls: list[dict[str, Any]],
    pred_hash: str,
    gold_calls: list[dict[str, Any]],
    gold_hash: str,
    catalog: dict[str, dict[str, Any]],
    read_calls_count: int = 0,
) -> tuple[float, float, bool, float]:
    """Compute official LifeMate competition score and dense episode reward.

    Returns:
        (official_score, f1, state_match, episode_reward)
    """
    res = score_sample(pred_calls, pred_hash, gold_calls, gold_hash, catalog)
    f1 = float(res.f1)
    state_match = bool(res.state_match)
    official_score = float(res.score)

    # Dense reward guiding policy toward exact state_hash match
    # Reward = 2.0 * official_score + 0.35 * f1 - 0.04 * unmatched_reads
    unmatched_reads = max(0, res.unmatched_pred)
    episode_reward = 2.0 * official_score + 0.35 * f1 - 0.04 * unmatched_reads

    return official_score, f1, state_match, episode_reward


def compute_gigpo_advantages(
    episodes: list[EpisodeRollout],
    omega: float = 1.0,
    gamma: float = 0.95,
    eps: float = 1e-6,
) -> list[EpisodeRollout]:
    """Compute Group-in-Group advantages (Macro Episode Advantage + Micro AnchorState Advantage).

    Adapted directly from GiGPO (NeurIPS 2025, core_gigpo.py:138-195).
    """
    if not episodes:
        return episodes

    # 1. Macro Episode Advantage
    ep_rewards = [ep.episode_reward for ep in episodes]
    mean_ep = sum(ep_rewards) / len(ep_rewards)
    var_ep = sum((r - mean_ep) ** 2 for r in ep_rewards) / len(ep_rewards)
    std_ep = math.sqrt(var_ep) + eps

    for ep in episodes:
        macro_adv = (ep.episode_reward - mean_ep) / std_ep
        for st in ep.steps:
            st.macro_advantage = macro_adv

    # 2. Micro AnchorState Advantage
    # Group discounted step returns by AnchorState
    anchor_groups: dict[tuple, list[tuple[TrajectoryStepInfo, float]]] = defaultdict(list)
    for ep in episodes:
        T = len(ep.steps)
        for t_idx, st in enumerate(ep.steps):
            disc_ret = (gamma ** (T - 1 - t_idx)) * ep.episode_reward
            anchor_groups[st.anchor_state].append((st, disc_ret))

    for anchor, items in anchor_groups.items():
        if len(items) >= 2:
            rets = [ret for _, ret in items]
            mean_ret = sum(rets) / len(rets)
            var_ret = sum((r - mean_ret) ** 2 for r in rets) / len(rets)
            std_ret = math.sqrt(var_ret) + eps
            for st, ret in items:
                st.micro_advantage = (ret - mean_ret) / std_ret
        else:
            for st, _ in items:
                st.micro_advantage = 0.0

    # 3. Combine advantages: A = A^E + omega * A^S
    for ep in episodes:
        for st in ep.steps:
            st.combined_advantage = st.macro_advantage + omega * st.micro_advantage

    return episodes
