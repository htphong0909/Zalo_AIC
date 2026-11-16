"""Exact RL policy, KL, MiniCPM5 reverse-KL OPD, and DiaTool turn-discounted DPO losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_gspo_clip_higher_loss(
    cur_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    clip_low: float = 0.20,
    clip_high: float = 0.28,
) -> tuple[torch.Tensor, dict[str, float]]:
    """GSPO sequence-level importance ratio (arXiv:2507.18071) + DAPO Clip-Higher + seq-mean-token-mean."""
    mask = loss_mask.float()
    seq_lens = mask.sum(dim=-1).clamp(min=1.0)
    log_diff = (cur_logprobs - old_logprobs.detach()) * mask
    seq_log_ratio = log_diff.sum(dim=-1) / seq_lens

    # Straight-through per-token gradient at sequence ratio scale
    st_log_ratio = cur_logprobs - cur_logprobs.detach() + seq_log_ratio.detach().unsqueeze(-1)
    st_log_ratio = torch.clamp(st_log_ratio, min=-10.0, max=10.0)
    ratio = torch.exp(st_log_ratio)

    adv = advantages.detach()
    pg1 = -adv * ratio
    clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_low), 1.0 + float(clip_high))
    pg2 = -adv * clipped_ratio
    pg_losses = torch.maximum(pg1, pg2)

    # seq-mean-token-mean matching LoopTool/verl core_algos.py:792-794
    per_seq = (pg_losses * mask).sum(dim=-1) / seq_lens
    loss = per_seq.mean()

    with torch.no_grad():
        was_clipped = ((pg2 > pg1).float() * mask).sum() / mask.sum().clamp(min=1.0)
        approx_kl = (-(cur_logprobs - old_logprobs) * mask).sum() / mask.sum().clamp(min=1.0)
    return loss, {"clipped_frac": float(was_clipped.item()), "approx_kl": float(approx_kl.item())}


def compute_low_var_kl_loss(
    cur_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Non-negative k3 low-variance KL penalty (always ADDED with +kl_coef to total loss)."""
    mask = loss_mask.float()
    seq_lens = mask.sum(dim=-1).clamp(min=1.0)
    delta = torch.clamp(ref_logprobs.detach() - cur_logprobs, min=-10.0, max=10.0)
    k3 = torch.clamp(torch.exp(delta) - delta - 1.0, min=0.0, max=100.0)
    return ((k3 * mask).sum(dim=-1) / seq_lens).mean()


def compute_opd_reverse_kl_advantages(
    student_token_logprobs: torch.Tensor,
    teacher_token_logprobs: torch.Tensor,
    episode_advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    alpha_ep: float = 0.5,
    clip_val: float = 2.0,
) -> torch.Tensor:
    """MiniCPM5 reverse-KL advantage: pull student up where teacher logprob > student logprob."""
    rkl_adv = torch.clamp(teacher_token_logprobs.detach() - student_token_logprobs.detach(), min=-clip_val, max=clip_val)
    total = (rkl_adv + float(alpha_ep) * episode_advantages.detach()) * loss_mask.float()
    return total


def _diatool_discounted_proxy(
    pi_turn_lps: list[torch.Tensor],
    ref_turn_lps: list[torch.Tensor],
    gamma: float,
) -> torch.Tensor:
    T = len(pi_turn_lps)
    if T == 0:
        return torch.tensor(0.0)
    denom = max(1e-8, 1.0 - (gamma ** T))
    weights = [(1.0 - (gamma ** (T - t))) / denom for t in range(T)]
    psi = max(1e-8, sum(weights))
    weighted = sum(w * (p - r.detach()) for w, p, r in zip(weights, pi_turn_lps, ref_turn_lps))
    return weighted / psi


def compute_diatool_step_dpo_loss(
    chosen_turn_lps: list[torch.Tensor],
    chosen_turn_ref_lps: list[torch.Tensor],
    rejected_turn_lps: list[torch.Tensor],
    rejected_turn_ref_lps: list[torch.Tensor],
    *,
    beta: float = 0.1,
    margin: float = 0.5,
    gamma: float = 0.5,
) -> torch.Tensor:
    """DiaTool-DPO temporal turn-normalized margin loss over variable-length multi-turn calls."""
    r_c = _diatool_discounted_proxy(chosen_turn_lps, chosen_turn_ref_lps, gamma)
    r_r = _diatool_discounted_proxy(rejected_turn_lps, rejected_turn_ref_lps, gamma)
    logits = float(beta) * (r_c - r_r - float(margin))
    return -F.logsigmoid(logits)
