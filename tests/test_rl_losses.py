"""Test GSPO+Clip-Higher, positive k3 KL, MiniCPM5 reverse-KL OPD, and DiaTool temporal Step-DPO."""
import math
import torch
from src.agent.rl_losses import (
    compute_diatool_step_dpo_loss,
    compute_gspo_clip_higher_loss,
    compute_low_var_kl_loss,
    compute_opd_reverse_kl_advantages,
)


def test_gspo_clip_higher_masks_non_assistant_tokens_and_asymmetrically_clips():
    cur_lp = torch.tensor([[-1.0, -1.0, -0.5, -0.5]], requires_grad=True)
    old_lp = torch.tensor([[-1.0, -1.0, -0.9, -0.9]])  # exp(delta) = exp(0.4) = 1.49 > 1.28
    adv = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

    loss, stats = compute_gspo_clip_higher_loss(cur_lp, old_lp, adv, mask, clip_low=0.20, clip_high=0.28)
    loss.backward()
    assert cur_lp.grad[0, 0].item() == 0.0
    assert cur_lp.grad[0, 1].item() == 0.0
    assert stats["clipped_frac"] > 0.0


def test_low_var_kl_is_nonnegative_and_zero_when_equal():
    lp = torch.tensor([[-1.2, -0.8, -0.4]], requires_grad=True)
    kl = compute_low_var_kl_loss(lp, lp.detach(), torch.tensor([[0.0, 1.0, 1.0]]))
    assert abs(kl.item()) < 1e-6

    cur_shifted = torch.tensor([[-1.2, -1.5, -0.1]], requires_grad=True)
    kl2 = compute_low_var_kl_loss(cur_shifted, torch.tensor([[-1.2, -0.8, -0.4]]), torch.tensor([[0.0, 1.0, 1.0]]))
    assert kl2.item() > 0.0


def test_opd_reverse_kl_rewards_tokens_aligned_with_teacher():
    s_lp = torch.tensor([[-2.0, -0.5, -0.1]])
    t_lp = torch.tensor([[-0.5, -0.5, -1.5]])
    ep_adv = torch.tensor([[0.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    adv = compute_opd_reverse_kl_advantages(s_lp, t_lp, ep_adv, mask, alpha_ep=0.0, clip_val=2.0)
    # Token 0: teacher has higher prob than student -> positive advantage (+1.5) to pull student UP!
    assert abs(adv[0, 0].item() - 1.5) < 1e-5
    # Token 2: student overconfident vs teacher -> negative advantage (-1.4)
    assert abs(adv[0, 2].item() - (-1.4)) < 1e-5


def test_diatool_step_dpo_weights_early_turns_higher_even_with_unequal_lengths():
    c_pi = [torch.tensor(-1.0, requires_grad=True), torch.tensor(-1.0, requires_grad=True)]
    c_ref = [torch.tensor(-1.5), torch.tensor(-1.0)]
    r_pi = [torch.tensor(-2.0, requires_grad=True), torch.tensor(-1.0, requires_grad=True), torch.tensor(-1.0, requires_grad=True)]
    r_ref = [torch.tensor(-1.5), torch.tensor(-1.0), torch.tensor(-1.0)]

    loss = compute_diatool_step_dpo_loss(c_pi, c_ref, r_pi, r_ref, beta=0.1, margin=0.5, gamma=0.5)
    loss.backward()
    assert math.isfinite(loss.item())
    assert abs(c_pi[0].grad.item()) > abs(c_pi[1].grad.item())


def test_trainer_chunked_logprobs_and_dry_run_configs():
    from types import SimpleNamespace
    import scripts.train_qwen35_rl as trainer

    # Tiny dummy causal LM checking chunked logsumexp matches full float log_softmax 1-to-1
    class _ToyLM(torch.nn.Module):
        def __init__(self, vocab: int = 31):
            super().__init__()
            self.emb = torch.nn.Embedding(vocab, 8)
            self.head = torch.nn.Linear(8, vocab, bias=False)

        def forward(self, input_ids, use_cache=False):
            logits = self.head(self.emb(input_ids))
            return SimpleNamespace(logits=logits)

    torch.manual_seed(0)
    m = _ToyLM()
    ids = [1, 5, 9, 12, 3, 7, 2]
    lp_chunked = trainer._compute_sequence_token_logprobs(m, ids, chunk_size=2)
    x = torch.tensor([ids], dtype=torch.long)
    lp_ref = torch.gather(
        torch.log_softmax(m(x).logits[:, :-1, :].float(), dim=-1), dim=-1, index=x[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    lp_ref = torch.nn.functional.pad(lp_ref, (1, 0), value=0.0)
    assert torch.allclose(lp_chunked, lp_ref, atol=1e-5)

    # Dry-run validates YAML schemas, split isolation, and public RL HF config without CUDA
    assert trainer.main(["--config", "configs/qwen35_v2_rl_stage2.yaml", "--dry-run"]) == 0
    assert trainer.main(["--config", "configs/qwen35_v2_rl_stage3_opd.yaml", "--dry-run"]) == 0


def test_streaming_entropy_tracker_handles_neginf_and_token_budget_packing():
    import scripts.train_qwen35_rl as trainer

    tracker = trainer._StreamingEntropyTracker()
    logits_t0 = torch.tensor([[1.0, 2.0, float("-inf"), 0.0], [3.0, 3.0, 3.0, 3.0]], dtype=torch.float32)
    logits_t1 = torch.tensor([[4.0, float("-inf"), float("-inf"), float("-inf")], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    tracker(torch.zeros((2, 3), dtype=torch.long), logits_t0)
    tracker(torch.zeros((2, 4), dtype=torch.long), logits_t1)

    ents = trainer._reduce_streaming_entropies(tracker.step_entropies, gen_lens=[1, 2])
    assert all(math.isfinite(e) and e >= 0.0 for e in ents)
    # Row 0 had len=1 -> only step 0 (3 valid classes with weights [e^1, e^2, e^0])
    p0 = torch.softmax(torch.tensor([1.0, 2.0, 0.0]), dim=-1)
    ref0 = float(-(p0 * p0.log()).sum().item())
    assert abs(ents[0] - ref0) < 1e-5
    # Row 1 had uniform 4-class at t0 (ln 4) + [0, 1, 0, 0] at t1
    p1 = torch.softmax(torch.tensor([0.0, 1.0, 0.0, 0.0]), dim=-1)
    ref1 = 0.5 * (math.log(4.0) + float(-(p1 * p1.log()).sum().item()))
    assert abs(ents[1] - ref1) < 1e-5

    order = [0, 1, 2, 3]
    tok_lens = [1000, 2000, 5000, 8000]
    c0 = trainer._pack_sorted_token_budget(order, tok_lens, cursor=0, max_rows=10, max_tokens=10000, group_size=1)
    # idx 0..2: 3 * 5000 = 15000 > 10000, so stops at [0, 1] (2 * 2000 = 4000 <= 10000)
    assert c0 == [0, 1]
    c1 = trainer._pack_sorted_token_budget(order, tok_lens, cursor=2, max_rows=10, max_tokens=10000, group_size=1)
    assert c1 == [2]
    c2 = trainer._pack_sorted_token_budget(order, tok_lens, cursor=3, max_rows=10, max_tokens=10000, group_size=1)
    assert c2 == [3]


def test_vllm_entropy_and_stop_token_resolution(tmp_path):
    from types import SimpleNamespace
    import json
    import scripts.train_qwen35_rl as trainer

    # Confident token (p1=0.999) stays far below entropy threshold 0.35
    h_conf = trainer._vllm_entropy_from_top_logprobs(
        [{1: SimpleNamespace(logprob=math.log(0.999)), 2: SimpleNamespace(logprob=math.log(0.0005))}]
    )
    assert 0.0 < h_conf < 0.05

    # Uncertain token (p1=0.50, p2=0.40) stays above boundary 0.35 and finite
    h_unc = trainer._vllm_entropy_from_top_logprobs(
        [
            {
                1: SimpleNamespace(logprob=math.log(0.50)),
                2: SimpleNamespace(logprob=math.log(0.40)),
                3: SimpleNamespace(logprob=math.log(0.05)),
            }
        ]
    )
    assert math.isfinite(h_unc) and h_unc > 0.65

    # Empty / malformed returns safe 0.0
    assert trainer._vllm_entropy_from_top_logprobs(None) == 0.0
    assert trainer._vllm_entropy_from_top_logprobs([{}]) == 0.0

    # Resolves generation_config + tokenizer eos/im_end tokens cleanly
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [248046, 248044]}), encoding="utf-8")

    class _DummyTok:
        eos_token_id = 248046

        def convert_tokens_to_ids(self, tok: str):
            return {"<|im_end|>": 248046, "<|endoftext|>": 248044}.get(tok, -1)

    assert trainer._resolve_stop_token_ids(_DummyTok(), tmp_path) == [248044, 248046]


def test_batched_right_padded_logprobs_and_grad_equivalence():
    from types import SimpleNamespace
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import scripts.train_qwen35_rl as trainer
    from src.agent.rl_losses import compute_gspo_clip_higher_loss, compute_low_var_kl_loss

    torch.manual_seed(7)

    class _TinyBackbone(nn.Module):
        def __init__(self, vocab: int = 32, dim: int = 16):
            super().__init__()
            self.emb = nn.Embedding(vocab, dim)
            self.attn = nn.MultiheadAttention(dim, num_heads=2, batch_first=True)

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            h = self.emb(input_ids)
            L = input_ids.shape[1]
            causal = torch.triu(torch.ones((L, L), dtype=torch.bool, device=h.device), diagonal=1)
            out, _ = self.attn(h, h, h, attn_mask=causal, is_causal=False, need_weights=False)
            return SimpleNamespace(last_hidden_state=h + out)

    class _CausalTinyLM(nn.Module):
        def __init__(self, vocab: int = 32, dim: int = 16):
            super().__init__()
            self.model = _TinyBackbone(vocab=vocab, dim=dim)
            self.lm_head = nn.Linear(dim, vocab, bias=False)

        def get_base_model(self):
            return self

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            hs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache).last_hidden_state
            return SimpleNamespace(logits=self.lm_head(hs))

    m1 = _CausalTinyLM()
    m1._fast_backbone_ok = False
    m1._batch_pad_ok = True
    seqs = [
        [1, 5, 9, 3, 7, 2],
        [4, 8, 12, 6, 10, 14, 15, 2],
        [3, 11, 7, 19],
    ]
    masks = [
        [0, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 1],
        [0, 1, 1, 1],
    ]

    single_lps = [trainer._compute_sequence_token_logprobs(m1, s, loss_mask=mk, chunk_size=2) for s, mk in zip(seqs, masks)]
    batched_lps = trainer._compute_batch_token_logprobs(m1, seqs, loss_masks=masks, pad_token_id=0, chunk_size=2)
    for s_lp, b_lp in zip(single_lps, batched_lps):
        assert s_lp.shape == b_lp.shape
        assert torch.allclose(s_lp, b_lp, atol=1e-5)

    # Also test packed budget helper
    batches = trainer._pack_token_budget_batches([len(s) for s in seqs], max_tokens=16, max_rows=2)
    assert sorted(i for b in batches for i in b) == [0, 1, 2]
    assert all(len(b) <= 2 and len(b) * max(len(seqs[i]) for i in b) <= 16 for b in batches)

    # Verify micro-batch gradient accumulation matches exact single-episode gradients
    m2 = _CausalTinyLM()
    m2.load_state_dict(m1.state_dict())
    m2._fast_backbone_ok = True
    # Monkeypatch model property to separate backbone output vs full output
    old_ref = [lp.detach().clone() - 0.1 for lp in single_lps]
    advs = [torch.full((1, len(s)), 0.5 if idx != 1 else -0.4) * torch.tensor([mk], dtype=torch.float32) for idx, (s, mk) in enumerate(zip(seqs, masks))]

    m1.zero_grad()
    n_eps = len(seqs)
    for s, mk, o_lp, adv in zip(seqs, masks, old_ref, advs):
        cur = trainer._compute_sequence_token_logprobs(m1, s, loss_mask=mk)
        msk = torch.tensor([mk], dtype=torch.float32)
        pg, _ = compute_gspo_clip_higher_loss(cur, o_lp, adv, msk)
        kl = compute_low_var_kl_loss(cur, o_lp, msk)
        ((pg + 0.01 * kl) / n_eps).backward()

    m2.zero_grad()
    packed = trainer._pack_token_budget_batches([len(s) for s in seqs], max_tokens=16, max_rows=2)
    for idxs in packed:
        cur_list = trainer._compute_batch_token_logprobs(m2, [seqs[i] for i in idxs], loss_masks=[masks[i] for i in idxs], pad_token_id=0, chunk_size=2)
        mb_max = max(lp.shape[1] for lp in cur_list)
        cur_b = torch.stack([F.pad(lp[0], (0, mb_max - lp.shape[1])) for lp in cur_list], dim=0)
        old_b = torch.stack([F.pad(old_ref[i][0], (0, mb_max - old_ref[i].shape[1])) for i in idxs], dim=0)
        adv_b = torch.stack([F.pad(advs[i][0], (0, mb_max - advs[i].shape[1])) for i in idxs], dim=0)
        msk_b = torch.tensor([masks[i] + [0] * (mb_max - len(masks[i])) for i in idxs], dtype=torch.float32)
        pg, _ = compute_gspo_clip_higher_loss(cur_b, old_b, adv_b, msk_b)
        kl = compute_low_var_kl_loss(cur_b, old_b, msk_b)
        ((pg + 0.01 * kl) * (len(idxs) / n_eps)).backward()

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5)



