"""Multi-turn Qwen3.5-4B RL & Voice-OPD trainer reusing frozen SFT V2 base + PEFT LoRA adapters.

Guarantees:
1. HFSyncer.preflight() before GPU allocation, LoRA-only checkpoints (~260 MB) pushed every 10m & save_steps.
2. Automatic resume from HF <hf_prefix>/checkpoints/latest/ (never restarts at step 0) + cached GCP buckets.
3. Zero extra base-model VRAM: uses model.disable_adapter() for SFT V2 ref KL and named PEFT adapters
   ('default' + 'teacher') for Stage 3 Privileged Voice/Error Reverse-KL OPD/SDPO.
4. Chunked exact logsumexp log-probs (<0.5 GB peak for vocab=248k) and Turn-0 batched rollouts.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    get_cosine_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.history import build_messages
from src.agent.qwen35_sft import QWEN35_LORA_TARGET_MODULES
from src.agent.rl_losses import (
    compute_gspo_clip_higher_loss,
    compute_low_var_kl_loss,
    compute_opd_reverse_kl_advantages,
)
from src.agent.rl_rollout import (
    MultiTurnEpisode,
    RolloutRequest,
    assign_group_gigpo_advantages,
    build_sample_with_asr,
    execute_batched_rollouts,
    filter_active_groups,
    sample_weighted_prompt_ids,
    select_arpo_anchor,
)
from src.agent.tool_selector import select_tools
from src.core.dataset import load_catalog, load_split
from src.core.hf_sync import ENV_TOKEN, BackgroundPusher, HFSyncer, resolve_repo_id
from src.env.safe_env import SafeEnv

# Persist shrunk prompt micro-batch cap & token budget after an OOM so future generate calls don't repeat OOM retries.
_ADAPTIVE_PROMPT_CAP: dict[tuple[int, bool, int], int] = {}
_ADAPTIVE_TOKEN_BUDGET: dict[tuple[int, bool, int], int] = {}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _unwrap_backbone_and_head(model: Any) -> tuple[Any | None, Any | None]:
    base_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(base_lm, "model", None)
    lm_head = getattr(base_lm, "lm_head", None)
    if backbone is None or lm_head is None or not callable(backbone) or not callable(lm_head):
        return None, None
    return backbone, lm_head


def _probe_fast_backbone_projection(model: Any) -> bool:
    """Verify backbone(x) -> slice -> lm_head matches model(x).logits to save ~7GB VRAM on long RL sequences."""
    bb, head = _unwrap_backbone_and_head(model)
    if bb is None or head is None:
        model._fast_backbone_ok = False
        return False
    dev = next(model.parameters()).device
    test_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long, device=dev)
    nz_test = torch.tensor([2, 5, 6], dtype=torch.long, device=dev)
    try:
        model.eval()
        with torch.no_grad():
            full_logits = model(input_ids=test_ids, use_cache=False).logits[:, :-1, :][:, nz_test, :]
            bb_out = bb(input_ids=test_ids, use_cache=False)
            hs = bb_out[0] if isinstance(bb_out, tuple) else bb_out.last_hidden_state
            sel_logits = head(hs[:, :-1, :][:, nz_test, :])
            max_diff = float((full_logits.float() - sel_logits.float()).abs().max().item())
            lp_full = F.log_softmax(full_logits.float(), dim=-1)
            lp_sel = F.log_softmax(sel_logits.float(), dim=-1)
            lp_diff = float((lp_full - lp_sel).abs().max().item())
        p_dtype = next(model.parameters()).dtype
        tol = 0.1 if p_dtype in (torch.bfloat16, torch.float16) else 1e-3
        lp_tol = 0.05 if p_dtype in (torch.bfloat16, torch.float16) else 1e-3
        if max_diff > tol or lp_diff > lp_tol:
            model._fast_backbone_ok = False
            print(
                f"[RL] Fast backbone projection diff={max_diff:.5f} (lp_diff={lp_diff:.5f}); using full forward.",
                flush=True,
            )
            return False

        # Verify gradients flow cleanly through sliced hidden states with active adapter
        model.train()
        bb_out = bb(input_ids=test_ids, use_cache=False)
        hs = bb_out[0] if isinstance(bb_out, tuple) else bb_out.last_hidden_state
        loss = head(hs[:, :-1, :][:, nz_test, :]).float().sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        model.zero_grad(set_to_none=True)
        model.eval()
        if not has_grad:
            model._fast_backbone_ok = False
            model._batch_pad_ok = False
            print("[RL] Fast backbone projection produced no parameter grads; using full forward.", flush=True)
            return False

        batch_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 0], [1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long, device=dev)
        batch_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long, device=dev)
        nz0 = torch.tensor([1, 3, 4], dtype=torch.long, device=dev)
        with torch.no_grad():
            ref0 = model(input_ids=batch_ids[:1, :6], use_cache=False).logits[:, :-1, :][:, nz0, :].float()
            bat_bb = bb(input_ids=batch_ids, attention_mask=batch_mask, use_cache=False)
            bat_hs = bat_bb[0] if isinstance(bat_bb, tuple) else bat_bb.last_hidden_state
            bat0 = head(bat_hs[:1, :-1, :][:, nz0, :]).float()
            lp_r = F.log_softmax(ref0, dim=-1)
            lp_b = F.log_softmax(bat0, dim=-1)
            tgt0 = batch_ids[:1, 1:6][:, nz0].unsqueeze(-1)
            tgt_lp_diff = float((lp_r.gather(-1, tgt0) - lp_b.gather(-1, tgt0)).abs().max().item())
            batch_lp_diff = float((lp_r - lp_b).abs().max().item())
        all_tol = 0.30 if p_dtype in (torch.bfloat16, torch.float16) else 1e-3
        tgt_tol = 0.12 if p_dtype in (torch.bfloat16, torch.float16) else 1e-3
        model._batch_pad_ok = (tgt_lp_diff <= tgt_tol) and (batch_lp_diff <= all_tol)
        model._fast_backbone_ok = True
        print(
            f"[RL] Fast backbone+lm_head projection ENABLED (diff={max_diff:.6f}, tgt_lp_diff={tgt_lp_diff:.6f}, batch_lp_diff={batch_lp_diff:.6f}, batch_pad={model._batch_pad_ok})",
            flush=True,
        )
        return True
    except Exception as exc:
        model.zero_grad(set_to_none=True)
        model.eval()
        model._fast_backbone_ok = False
        model._batch_pad_ok = False
        print(f"[RL] Fast backbone projection fallback ({exc})", flush=True)
        return False


def _chunked_selected_logprobs(
    logits_or_hs: torch.Tensor,
    targets_sel: torch.Tensor,
    *,
    head: Any | None = None,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Compute exact next-token log P(target) over flattened [N_active, D] rows in bounded chunks."""
    n_eval = logits_or_hs.shape[0]
    if n_eval == 0:
        return torch.empty((0,), dtype=torch.float32, device=logits_or_hs.device)
    chunks: list[torch.Tensor] = []
    for start in range(0, n_eval, chunk_size):
        sl = logits_or_hs[start : start + chunk_size]
        if head is not None:
            sl = head(sl)
        sl_f = sl.float()
        tg = targets_sel[start : start + chunk_size]
        tgt_logits = torch.gather(sl_f, dim=-1, index=tg.unsqueeze(-1)).squeeze(-1)
        lse = torch.logsumexp(sl_f, dim=-1)
        chunks.append(tgt_logits - lse)
    return torch.cat(chunks, dim=0)


def _compute_sequence_token_logprobs(
    model: Any,
    input_ids: list[int],
    *,
    loss_mask: list[int] | None = None,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Compute exact next-token log P(x_t | x_<t) over [1, T] on active masked positions via chunked logsumexp."""
    dev = next(model.parameters()).device
    seq_len = len(input_ids)
    if seq_len <= 1:
        return torch.zeros((1, seq_len), dtype=torch.float32, device=dev)

    nz: torch.Tensor | None = None
    if loss_mask is not None:
        m_shift = torch.tensor(loss_mask[1:seq_len], dtype=torch.bool, device=dev)
        nz = m_shift.nonzero(as_tuple=True)[0]
        if nz.numel() == 0:
            return torch.zeros((1, seq_len), dtype=torch.float32, device=dev)

    x = torch.tensor([input_ids], dtype=torch.long, device=dev)
    targets = x[:, 1:]
    if nz is not None:
        targets = targets[:, nz]

    if nz is not None and getattr(model, "_fast_backbone_ok", False):
        bb, head = _unwrap_backbone_and_head(model)
        assert bb is not None and head is not None
        out_bb = bb(input_ids=x, use_cache=False)
        hs = out_bb[0] if isinstance(out_bb, tuple) else out_bb.last_hidden_state
        del out_bb
        hs_sel = hs[0, :-1, :][nz]
        del hs
        sel_lp = _chunked_selected_logprobs(hs_sel, targets[0], head=head, chunk_size=chunk_size)
        del hs_sel
    else:
        out = model(input_ids=x, use_cache=False)
        logits_shift = out.logits[0, :-1, :]
        del out
        if nz is not None:
            logits_shift = logits_shift[nz]
        sel_lp = _chunked_selected_logprobs(logits_shift, targets[0], head=None, chunk_size=chunk_size)
        del logits_shift

    if nz is not None:
        full_lp = torch.zeros((1, seq_len), dtype=sel_lp.dtype, device=dev)
        return full_lp.index_put((torch.zeros_like(nz), nz + 1), sel_lp)
    return F.pad(sel_lp.unsqueeze(0), (1, 0), value=0.0)


def _compute_batch_token_logprobs(
    model: Any,
    sequences: list[list[int]],
    *,
    loss_masks: list[list[int] | None] | None = None,
    pad_token_id: int = 0,
    chunk_size: int = 128,
) -> list[torch.Tensor]:
    """Compute exact [1, L_i] token logprobs using right-padded causal batching."""
    if not sequences:
        return []
    masks = list(loss_masks) if loss_masks is not None else [None] * len(sequences)
    if len(sequences) == 1 or not getattr(model, "_batch_pad_ok", True):
        return [
            _compute_sequence_token_logprobs(model, seq, loss_mask=m, chunk_size=chunk_size)
            for seq, m in zip(sequences, masks)
        ]

    dev = next(model.parameters()).device
    bsz = len(sequences)
    max_len = max(len(s) for s in sequences)
    if max_len <= 1:
        return [torch.zeros((1, len(s)), dtype=torch.float32, device=dev) for s in sequences]

    x = torch.full((bsz, max_len), int(pad_token_id), dtype=torch.long, device=dev)
    attn_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=dev)
    shift_mask = torch.zeros((bsz, max_len - 1), dtype=torch.bool, device=dev)
    for i, (seq, m) in enumerate(zip(sequences, masks)):
        ln = len(seq)
        if ln == 0:
            continue
        x[i, :ln] = torch.tensor(seq, dtype=torch.long, device=dev)
        attn_mask[i, :ln] = 1
        if ln > 1:
            if m is None:
                shift_mask[i, : ln - 1] = True
            else:
                shift_mask[i, : ln - 1] = torch.tensor(m[1:ln], dtype=torch.bool, device=dev)

    if not bool(shift_mask.any().item()):
        return [torch.zeros((1, len(s)), dtype=torch.float32, device=dev) for s in sequences]

    targets_sel = x[:, 1:][shift_mask]
    if getattr(model, "_fast_backbone_ok", False):
        bb, head = _unwrap_backbone_and_head(model)
        assert bb is not None and head is not None
        out_bb = bb(input_ids=x, attention_mask=attn_mask, use_cache=False)
        hs = out_bb[0] if isinstance(out_bb, tuple) else out_bb.last_hidden_state
        del out_bb
        act_sel = hs[:, :-1, :][shift_mask]
        del hs
        sel_lp = _chunked_selected_logprobs(act_sel, targets_sel, head=head, chunk_size=chunk_size)
        del act_sel
    else:
        out = model(input_ids=x, attention_mask=attn_mask, use_cache=False)
        logits_sel = out.logits[:, :-1, :][shift_mask]
        del out
        sel_lp = _chunked_selected_logprobs(logits_sel, targets_sel, head=None, chunk_size=chunk_size)
        del logits_sel

    b_idx, t_idx = shift_mask.nonzero(as_tuple=True)
    full_lp = torch.zeros((bsz, max_len), dtype=sel_lp.dtype, device=dev).index_put((b_idx, t_idx + 1), sel_lp)
    return [full_lp[i : i + 1, : len(sequences[i])] for i in range(bsz)]


def _pack_token_budget_batches(
    lengths: list[int],
    *,
    max_tokens: int,
    max_rows: int,
) -> list[list[int]]:
    """Pack length-sorted indices into micro-batches respecting padded (rows * max_len) <= max_tokens."""
    if not lengths:
        return []
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for idx in order:
        ln = lengths[idx]
        new_max = max(cur_max, ln)
        new_rows = len(cur) + 1
        if cur and (new_rows > max_rows or (new_rows * new_max) > max_tokens):
            batches.append(cur)
            cur = [idx]
            cur_max = ln
        else:
            cur.append(idx)
            cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def _compute_logprobs_packed(
    model: Any,
    items: list[tuple[list[int], list[int] | None]],
    *,
    max_tokens: int,
    max_rows: int,
    pad_token_id: int = 0,
    to_cpu: bool = True,
) -> list[torch.Tensor]:
    """Compute per-sequence [1, T] token logprobs across length-packed micro-batches with OOM splitting."""
    out: list[torch.Tensor | None] = [None] * len(items)
    queue: list[list[int]] = _pack_token_budget_batches(
        [len(seq) for seq, _ in items],
        max_tokens=max_tokens,
        max_rows=max_rows,
    )
    while queue:
        idxs = queue.pop(0)
        seqs = [items[i][0] for i in idxs]
        masks = [items[i][1] for i in idxs]
        try:
            lps = _compute_batch_token_logprobs(model, seqs, loss_masks=masks, pad_token_id=pad_token_id)
            for idx, lp in zip(idxs, lps):
                out[idx] = lp.cpu() if to_cpu else lp
        except torch.cuda.OutOfMemoryError:
            if len(idxs) <= 1:
                raise
            torch.cuda.empty_cache()
            mid = max(1, len(idxs) // 2)
            queue = [idxs[:mid], idxs[mid:]] + queue
    return [t for t in out if t is not None]


def _build_token_advantage_tensor(ep: MultiTurnEpisode, seq_len: int, device: torch.device) -> torch.Tensor:
    step_adv_map = {st.step_idx: float(st.combined_advantage) for st in ep.steps}
    vals = [step_adv_map.get(sidx, 0.0) if m == 1 else 0.0 for sidx, m in zip(ep.step_ids_per_token, ep.loss_mask)]
    if len(vals) < seq_len:
        vals.extend([0.0] * (seq_len - len(vals)))
    return torch.tensor([vals[:seq_len]], dtype=torch.float32, device=device)


def _compute_teacher_batch_aligned_logprobs(
    model: Any,
    episodes: list[MultiTurnEpisode],
    *,
    max_tokens: int = 16384,
    max_rows: int = 8,
    pad_token_id: int = 0,
    to_cpu: bool = True,
) -> list[torch.Tensor]:
    """Evaluate teacher adapter over (teacher_prompt_ids + student_completion_ids) aligned 1-to-1."""
    items: list[tuple[list[int], list[int] | None]] = []
    for ep in episodes:
        if not ep.teacher_prompt_ids:
            items.append((ep.input_ids, ep.loss_mask))
        else:
            comp_ids = ep.input_ids[ep.prompt_token_len :]
            comp_mask = ep.loss_mask[ep.prompt_token_len :]
            items.append((ep.teacher_prompt_ids + comp_ids, ([0] * ep.teacher_prompt_len) + list(comp_mask)))

    raw_lps = _compute_logprobs_packed(
        model,
        items,
        max_tokens=max_tokens,
        max_rows=max_rows,
        pad_token_id=pad_token_id,
        to_cpu=to_cpu,
    )
    aligned_list: list[torch.Tensor] = []
    for ep, t_lp_full in zip(episodes, raw_lps):
        if not ep.teacher_prompt_ids:
            aligned_list.append(t_lp_full)
            continue
        t_comp_lp = t_lp_full[:, ep.teacher_prompt_len :]
        aligned = torch.zeros((1, len(ep.input_ids)), dtype=t_comp_lp.dtype, device=t_comp_lp.device)
        n = min(len(ep.input_ids) - ep.prompt_token_len, t_comp_lp.shape[1])
        if n > 0:
            aligned[:, ep.prompt_token_len : ep.prompt_token_len + n] = t_comp_lp[:, :n]
        aligned_list.append(aligned)
    return aligned_list


def _compute_teacher_aligned_logprobs(
    model: Any,
    ep: MultiTurnEpisode,
) -> torch.Tensor:
    return _compute_teacher_batch_aligned_logprobs(model, [ep], to_cpu=False)[0]


def _load_asr_cache(path: str | None) -> dict[str, dict[str, Any]]:
    if not path or not Path(path).exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[row["id"]] = row
    return out


def _precompute_gold_hashes(samples: list[Any], split: str, root: str) -> dict[str, tuple[list[dict[str, Any]], str]]:
    cache: dict[str, tuple[list[dict[str, Any]], str]] = {}
    for s in samples:
        gold = [c.model_dump() for c in (s.label.tool_calls if s.label else [])]
        env = SafeEnv(split, s.id, root)
        for gc in gold:
            env.call(gc["name"], gc["arguments"])
        cache[s.id] = (gold, env.get_state_hash())
    return cache


def _ensure_base_model(cfg: dict[str, Any], out_dir: Path, token: str | None) -> Path:
    local = Path(cfg["base_model_id"])
    if (local / "config.json").exists():
        return local
    base_hf_repo = cfg.get("base_hf_repo")
    base_hf_path = cfg.get("base_hf_path")
    if not base_hf_repo or not base_hf_path:
        return local

    from huggingface_hub import snapshot_download

    cache_root = out_dir.parent / "_base_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    print(f"[RL] Downloading base SFT V2 from {base_hf_repo}/{base_hf_path}...", flush=True)
    snapshot_download(
        repo_id=base_hf_repo,
        repo_type="model",
        allow_patterns=[f"{base_hf_path}/*", f"{base_hf_path}/**"],
        local_dir=str(cache_root),
        token=token,
    )
    downloaded = cache_root / base_hf_path
    if not (downloaded / "config.json").exists():
        raise FileNotFoundError(f"Base model config.json missing after HF download at {downloaded}")
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists():
        shutil.rmtree(local)
    shutil.move(str(downloaded), str(local))
    return local


def _ensure_asr_cache(cfg: dict[str, Any], syncer: HFSyncer) -> str | None:
    asr_path = cfg.get("asr_cache")
    if asr_path and Path(asr_path).exists():
        return asr_path
    remote = cfg.get("asr_cache_hf")
    if asr_path and remote and syncer.exists(remote):
        dest = Path(asr_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dl = syncer.api.hf_hub_download(
            repo_id=syncer.repo_id,
            filename=remote,
            repo_type=syncer.repo_type,
            local_dir=str(dest.parent / "_hf_asr"),
        )
        shutil.copy2(dl, dest)
        print(f"[RL] Downloaded ASR cache from {syncer.repo_id}/{remote} -> {dest}", flush=True)
        return str(dest)
    return asr_path


def _load_base_causal_lm(base_path: str | Path) -> Any:
    try:
        return AutoModelForCausalLM.from_pretrained(
            str(base_path), dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}, trust_remote_code=True
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            str(base_path), torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}, trust_remote_code=True
        )


class _StreamingEntropyTracker(LogitsProcessor):
    """Computes masked-token Shannon entropy on the fly without retaining [B, T, V] scores in VRAM."""

    def __init__(self) -> None:
        self.step_entropies: list[torch.Tensor] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        with torch.no_grad():
            s = scores.float()
            lse = torch.logsumexp(s, dim=-1, keepdim=True)
            lp = s - lse
            p = lp.exp()
            # Guard 0.0 * -inf = NaN when any upstream processor/warper sets -inf.
            ent = torch.where(torch.isfinite(lp), -p * lp, torch.zeros_like(lp)).sum(dim=-1)
            self.step_entropies.append(ent)
        return scores


def _reduce_streaming_entropies(step_entropies: list[torch.Tensor], gen_lens: list[int]) -> list[float]:
    total_rows = len(gen_lens)
    if total_rows == 0 or not step_entropies:
        return [0.0] * total_rows
    stacked = torch.stack(step_entropies, dim=0)  # [T, B]
    dev = stacked.device
    len_tensor = torch.tensor(gen_lens, dtype=torch.long, device=dev)
    steps_vec = torch.arange(stacked.shape[0], device=dev).unsqueeze(1)
    active = (steps_vec < len_tensor.unsqueeze(0)).float()
    ent_per_row = (stacked.float() * active).sum(dim=0) / active.sum(dim=0).clamp(min=1.0)
    return [float(x) for x in ent_per_row.tolist()]


def _collect_eos_ids(tokenizer: Any, model: Any) -> set[int]:
    eos_set: set[int] = set()
    for src in (
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
    ):
        if isinstance(src, int):
            eos_set.add(src)
        elif isinstance(src, (list, tuple, set)):
            eos_set.update(int(x) for x in src)
    return eos_set


def _run_generate_subbatch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    num_return_sequences: int,
) -> list[tuple[str, list[int], float, float]]:
    dev = next(model.parameters()).device
    orig_pad_side = getattr(tokenizer, "padding_side", "right")
    eos_raw = getattr(tokenizer, "eos_token_id", None)
    eos_first = eos_raw[0] if isinstance(eos_raw, (list, tuple)) and eos_raw else eos_raw
    if getattr(tokenizer, "pad_token_id", None) is None and eos_first is not None:
        tokenizer.pad_token_id = eos_first
    pad_id = getattr(tokenizer, "pad_token_id", eos_first)
    eos_set = _collect_eos_ids(tokenizer, model)

    try:
        tokenizer.padding_side = "left"
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(dev)
    finally:
        tokenizer.padding_side = orig_pad_side

    tracker = _StreamingEntropyTracker()
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "num_return_sequences": int(num_return_sequences),
        "return_dict_in_generate": False,
        "output_scores": False,
        "logits_processor": LogitsProcessorList([tracker]),
        "pad_token_id": pad_id,
        "use_cache": True,  # Override gradient_checkpointing use_cache=False for 20x faster rollouts
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)

    with torch.no_grad():
        gen = model.generate(**enc, **gen_kwargs)

    seqs_t = gen.sequences if hasattr(gen, "sequences") else gen
    prompt_width = int(enc["input_ids"].shape[1])
    total_rows = int(seqs_t.shape[0])
    row_seqs: list[list[int]] = []
    for r in range(total_rows):
        seq = seqs_t[r, prompt_width:].tolist()
        cut = next((i for i, tid in enumerate(seq) if tid in eos_set), None)
        if cut is not None:
            seq = seq[: cut + 1]
        elif pad_id is not None:
            while seq and seq[-1] == pad_id:
                seq.pop()
        row_seqs.append(seq)

    entropies = _reduce_streaming_entropies(tracker.step_entropies, [max(1, len(s)) for s in row_seqs])
    del gen, seqs_t, enc, tracker

    out: list[tuple[str, list[int], float, float]] = []
    for seq, ent in zip(row_seqs, entropies):
        text = tokenizer.decode(seq, skip_special_tokens=True)
        out.append((text, seq, 0.0, ent))
    return out


def _pack_sorted_token_budget(
    order: list[int], tok_lens: list[int], cursor: int, max_rows: int, max_tokens: int, group_size: int
) -> list[int]:
    chunk: list[int] = []
    for idx in order[cursor:]:
        new_rows = len(chunk) + 1
        padded_tokens = new_rows * tok_lens[idx] * group_size
        if chunk and (new_rows > max_rows or padded_tokens > max_tokens):
            break
        chunk.append(idx)
    return chunk


def _generate_batch_prompts(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float = 1.0,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
    micro_batch_size: int = 12,
) -> list[tuple[str, list[int], float, float]]:
    if not prompts:
        return []

    g = max(1, int(num_return_sequences))
    prompt_cap = max(1, int(micro_batch_size) // g)
    cap_key = (id(model), bool(do_sample), g)
    try:
        tok_lens = [max(1, len(ids)) for ids in tokenizer(prompts, add_special_tokens=False)["input_ids"]]
    except Exception:
        tok_lens = [max(1, len(p) // 3) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: tok_lens[i])
    results: list[tuple[str, list[int], float, float] | None] = [None] * (len(prompts) * g)

    default_budget = 88000
    if torch.cuda.is_available():
        try:
            dev = next(model.parameters()).device
            if dev.type == "cuda":
                vram_gb = torch.cuda.get_device_properties(dev).total_memory / 1e9
                default_budget = 88000 if vram_gb >= 35.0 else 32000
        except StopIteration:
            pass

    cursor = 0
    cur_bs = min(prompt_cap, _ADAPTIVE_PROMPT_CAP.get(cap_key, prompt_cap))
    cur_budget = _ADAPTIVE_TOKEN_BUDGET.get(cap_key, default_budget)
    while cursor < len(order):
        idx_chunk = _pack_sorted_token_budget(order, tok_lens, cursor, cur_bs, cur_budget, g)
        sub_prompts = [prompts[i] for i in idx_chunk]
        try:
            sub_out = _run_generate_subbatch(
                model,
                tokenizer,
                sub_prompts,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=g,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(idx_chunk) == 1:
                raise
            cur_bs = max(1, min(cur_bs, len(idx_chunk)) // 2)
            cur_budget = max(16384, int(cur_budget * 0.65))
            _ADAPTIVE_PROMPT_CAP[cap_key] = cur_bs
            _ADAPTIVE_TOKEN_BUDGET[cap_key] = cur_budget
            print(
                f"[RL] CUDA OOM during batched generate -> shrinking prompt micro_batch={cur_bs}, token_budget={cur_budget}",
                flush=True,
            )
            continue

        for local_i, orig_idx in enumerate(idx_chunk):
            for k in range(g):
                results[orig_idx * g + k] = sub_out[local_i * g + k]
        cursor += len(idx_chunk)

    return results  # type: ignore[return-value]


def _save_and_push_ckpt(
    model: PeftModel,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    out_dir: Path,
    step: int,
    metrics: dict[str, Any],
    pusher: BackgroundPusher,
    hf_prefix: str,
) -> None:
    snap = out_dir / f"checkpoint-{step}"
    if snap.exists():
        shutil.rmtree(snap)
    snap.mkdir(parents=True, exist_ok=True)
    tmp_save = snap / "_peft_tmp"
    has_teacher = "teacher" in getattr(model, "peft_config", {})
    if has_teacher:
        model.save_pretrained(str(tmp_save), selected_adapters=["default"])
        src_dir = tmp_save / "default" if (tmp_save / "default").is_dir() else tmp_save
        for p in src_dir.iterdir():
            shutil.move(str(p), str(snap / p.name))
        shutil.rmtree(tmp_save, ignore_errors=True)
    else:
        model.save_pretrained(str(snap))
    tokenizer.save_pretrained(str(snap))
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        snap / "trainer_state.pt",
    )
    (snap / "README.md").unlink(missing_ok=True)
    (snap / "metrics.json").write_text(json.dumps({"step": step, **metrics}, indent=2), encoding="utf-8")
    pusher.submit(snap, f"{hf_prefix}/checkpoints/checkpoint-{step}", f"RL step {step}")
    pusher.submit(snap, f"{hf_prefix}/checkpoints/latest", f"RL latest step {step}")


def _prompt_for_sample(
    sample: Any,
    catalog: dict[str, dict[str, Any]],
    tokenizer: Any,
    asr_diff: str | None,
    max_prompt_tokens: int = 15200,
) -> str:
    messages = build_messages(
        sample,
        current_text=sample.audio_text,
        asr_phonetic_diff=asr_diff,
        prompt_style="qwen35",
        max_prompt_tokens=max_prompt_tokens,
    )
    tools = select_tools(sample, catalog)
    return tokenizer.apply_chat_template(
        messages, tools=tools or None, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def _vllm_entropy_from_top_logprobs(
    token_logprobs: list[dict[Any, Any]] | None,
    vocab_size: int = 248320,
) -> float:
    """Compute bounded Shannon entropy estimate from vLLM per-token top-K logprobs."""
    if not token_logprobs:
        return 0.0
    vals: list[float] = []
    for lp_dict in token_logprobs:
        if not lp_dict:
            continue
        lps = [float(getattr(v, "logprob", v)) for v in lp_dict.values()]
        finite_lps = [lp for lp in lps if math.isfinite(lp)]
        if not finite_lps:
            continue
        ps = [math.exp(lp) for lp in finite_lps]
        h_top = sum(-p * lp for p, lp in zip(ps, finite_lps) if p > 0.0)
        rem = max(0.0, 1.0 - sum(ps))
        h_tail = 0.0
        if rem > 1e-9:
            p_min = max(min(ps), 1e-12)
            tail_slots = max(1, int(vocab_size) - len(ps))
            min_tail = -rem * math.log(p_min)
            max_tail = -rem * math.log(max(rem / tail_slots, 1e-12))
            p_eff = max(p_min * 0.25, rem / tail_slots, 1e-12)
            h_tail = max(min_tail, min(max_tail, -rem * math.log(p_eff)))
        vals.append(h_top + h_tail)
    return float(sum(vals) / len(vals)) if vals else 0.0


def _resolve_stop_token_ids(tokenizer: Any, model_dir: str | Path | None = None) -> list[int]:
    stop_ids = _collect_eos_ids(tokenizer, None)
    if model_dir is not None:
        gen_cfg = Path(model_dir) / "generation_config.json"
        if gen_cfg.exists():
            try:
                raw_eos = json.loads(gen_cfg.read_text(encoding="utf-8")).get("eos_token_id")
                if isinstance(raw_eos, int):
                    stop_ids.add(raw_eos)
                elif isinstance(raw_eos, (list, tuple)):
                    stop_ids.update(int(x) for x in raw_eos)
            except Exception:
                pass
    for tok_str in ("<|im_end|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok_str)
            if isinstance(tid, int) and tid >= 0:
                stop_ids.add(tid)
        except Exception:
            pass
    return sorted(stop_ids)


class _VLLMRolloutEngine:
    """Accelerated vLLM rollout wrapper supporting standalone GCP mining and sleep-mode LoRA RL."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.72,
        enable_lora: bool = False,
        max_lora_rank: int = 32,
        enable_sleep_mode: bool = False,
        seed: int = 42,
        max_num_seqs: int = 48,
        vocab_size: int = 248320,
        stop_token_ids: list[int] | None = None,
    ) -> None:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
        from vllm import LLM  # type: ignore

        self.vocab_size = int(vocab_size)
        self.enable_lora = bool(enable_lora)
        self.enable_sleep_mode = bool(enable_sleep_mode)
        self.stop_token_ids = list(stop_token_ids or [])
        self._sleeping = False
        self.llm = LLM(
            model=str(model_path),
            dtype="bfloat16",
            max_model_len=int(max_model_len),
            max_num_seqs=int(max_num_seqs),
            gpu_memory_utilization=float(gpu_memory_utilization),
            enable_lora=self.enable_lora,
            max_loras=1,
            max_lora_rank=int(max_lora_rank),
            enable_prefix_caching=True,
            enable_sleep_mode=self.enable_sleep_mode,
            enforce_eager=True,
            trust_remote_code=True,
            seed=int(seed),
        )

    def sleep(self) -> None:
        if self.enable_sleep_mode and not self._sleeping:
            self.llm.sleep(level=1)
            self._sleeping = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def wake_up(self) -> None:
        if self.enable_sleep_mode and self._sleeping:
            self.llm.wake_up()
            self._sleeping = False

    def close(self) -> None:
        llm = getattr(self, "llm", None)
        if llm is not None:
            self.llm = None
            del llm
        try:
            from vllm.distributed.parallel_state import (  # type: ignore
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_prompts(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        lora_path: str | Path | None = None,
        lora_id: int = 1,
    ) -> list[tuple[str, list[int], float, float]]:
        if not prompts:
            return []
        from vllm import SamplingParams  # type: ignore

        sp = SamplingParams(
            temperature=float(temperature) if do_sample else 0.0,
            top_p=float(top_p) if do_sample else 1.0,
            n=max(1, int(num_return_sequences)),
            max_tokens=int(max_new_tokens),
            logprobs=20,
            stop_token_ids=self.stop_token_ids or None,
        )
        lora_req = None
        if lora_path is not None and self.enable_lora:
            from vllm.lora.request import LoRARequest  # type: ignore

            lora_req = LoRARequest(f"rl_{int(lora_id)}", max(1, int(lora_id)), str(lora_path))
        req_outs = self.llm.generate(prompts, sp, lora_request=lora_req, use_tqdm=False)
        out: list[tuple[str, list[int], float, float]] = []
        for req_out in req_outs:
            for comp in req_out.outputs:
                ent = _vllm_entropy_from_top_logprobs(comp.logprobs, vocab_size=self.vocab_size)
                out.append((comp.text, list(comp.token_ids), 0.0, ent))
        return out


def _save_adapter_snapshot_for_vllm(model: PeftModel, snap_dir: Path) -> None:
    if snap_dir.exists():
        shutil.rmtree(snap_dir, ignore_errors=True)
    snap_dir.mkdir(parents=True, exist_ok=True)
    has_teacher = "teacher" in getattr(model, "peft_config", {})
    if has_teacher:
        tmp = snap_dir / "_tmp"
        model.save_pretrained(str(tmp), selected_adapters=["default"])
        src_dir = tmp / "default" if (tmp / "default").is_dir() else tmp
        for p in src_dir.iterdir():
            shutil.move(str(p), str(snap_dir / p.name))
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        model.save_pretrained(str(snap_dir))


def _vllm_backend_enabled(cfg: dict[str, Any], phase: str) -> bool:
    if not torch.cuda.is_available() or importlib.util.find_spec("vllm") is None:
        return False
    roll_cfg = cfg.get("rollout", {})
    if phase == "gcp":
        return bool(roll_cfg.get("use_vllm_gcp", roll_cfg.get("use_vllm", True)))
    if phase == "rl":
        return bool(roll_cfg.get("use_vllm_rl", False))
    return False


def _make_vllm_step_gen_fn(
    engine: _VLLMRolloutEngine,
    tokenizer: Any,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float = 1.0,
    top_p: float = 1.0,
    lora_path_fn: Any | None = None,
    lora_id_fn: Any | None = None,
):
    def _batch_gen(reqs: list[tuple[list[dict[str, Any]], int, list[dict[str, Any]]]]):
        prompts = [
            tokenizer.apply_chat_template(
                msgs, tools=tools or None, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for msgs, _sidx, tools in reqs
        ]
        lora_path = lora_path_fn() if lora_path_fn is not None else None
        lora_id = int(lora_id_fn()) if lora_id_fn is not None else 1
        return engine.generate_prompts(
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            lora_path=lora_path,
            lora_id=lora_id,
        )

    return _batch_gen


def _make_batch_step_gen_fn(
    model: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float = 1.0,
    top_p: float = 1.0,
    micro_batch_size: int = 12,
):
    def _batch_gen(reqs: list[tuple[list[dict[str, Any]], int, list[dict[str, Any]]]]):
        prompts = [
            tokenizer.apply_chat_template(
                msgs, tools=tools or None, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for msgs, _sidx, tools in reqs
        ]
        return _generate_batch_prompts(
            model,
            tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            micro_batch_size=micro_batch_size,
        )

    return _batch_gen


def _mine_gcp_buckets(
    model: PeftModel | None,
    tokenizer: Any,
    samples: list[Any],
    asr_by_id: dict[str, dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
    gold_map: dict[str, tuple[list[dict[str, Any]], str]],
    cfg: dict[str, Any],
    *,
    out_dir: Path,
    pusher: BackgroundPusher,
    syncer: HFSyncer,
    hf_prefix: str,
    wall_deadline_t: float,
    batch_gen_override: Any | None = None,
) -> tuple[dict[str, list[str]], bool]:
    """Run resumable, batched multi-turn GCP probe splitting samples into Tier A/B/C."""
    if model is not None:
        model.eval()
    partial_file = out_dir / "gcp_buckets_partial.json"
    remote_partial = f"{hf_prefix}/gcp_buckets_partial.json"

    buckets: dict[str, list[str]] = {"tier_a_hard": [], "tier_b_boundary": [], "tier_c_anchor": []}
    if not partial_file.exists() and syncer.exists(remote_partial):
        dl_p = syncer.api.hf_hub_download(
            repo_id=syncer.repo_id,
            filename=remote_partial,
            repo_type=syncer.repo_type,
            local_dir=str(out_dir / "_hf_buckets"),
        )
        shutil.copy2(dl_p, partial_file)
    if partial_file.exists():
        buckets = json.loads(partial_file.read_text(encoding="utf-8"))

    done_ids = {sid for lst in buckets.values() for sid in lst}
    remaining = [s for s in samples if s.id not in done_ids]
    if done_ids:
        print(f"[GCP Mine] Resuming from {len(done_ids)}/{len(samples)} already probed samples.", flush=True)

    gcp_bs = max(1, int(cfg["rollout"].get("gcp_batch_size", 16)))
    if torch.cuda.is_available() and (torch.cuda.get_device_properties(0).total_memory / 1e9) < 28.0:
        gcp_bs = min(gcp_bs, 16 if batch_gen_override is not None else 12)
    max_new = int(cfg["rollout"]["max_new_tokens"])
    max_steps = int(cfg["rollout"]["max_steps"])
    chunk_sz = max(8, gcp_bs)
    gcp_sync_s = float(cfg["optim"].get("gcp_sync_interval_s", 120.0))
    if batch_gen_override is not None:
        batch_greedy_fn = batch_gen_override
    else:
        assert model is not None, "PyTorch GCP mining requires loaded model"
        batch_greedy_fn = _make_batch_step_gen_fn(
            model, tokenizer, max_new_tokens=max_new, do_sample=False, micro_batch_size=gcp_bs
        )
    last_push_t = time.monotonic()

    for start in range(0, len(remaining), chunk_sz):
        chunk_t0 = time.monotonic()
        batch = remaining[start : start + chunk_sz]
        roll_reqs: list[RolloutRequest] = []
        layout: list[tuple[str, int, int | None]] = []  # (sample_id, clean_idx, asr_idx_or_None)
        for s in batch:
            g_calls, g_hash = gold_map[s.id]
            clean_idx = len(roll_reqs)
            roll_reqs.append(
                RolloutRequest(sample=s, gold_calls=g_calls, gold_hash=g_hash, clean_sample=s, use_asr=False)
            )
            asr_idx: int | None = None
            if s.id in asr_by_id:
                s_noisy, diff = build_sample_with_asr(s, asr_by_id[s.id], inject_asr=True)
                asr_idx = len(roll_reqs)
                roll_reqs.append(
                    RolloutRequest(
                        sample=s_noisy,
                        gold_calls=g_calls,
                        gold_hash=g_hash,
                        clean_sample=s,
                        asr_phonetic_diff=diff,
                        use_asr=True,
                    )
                )
            layout.append((s.id, clean_idx, asr_idx))

        episodes = execute_batched_rollouts(
            roll_reqs,
            catalog,
            cfg["train_split"],
            cfg["root"],
            batch_greedy_fn,
            tokenizer,
            max_steps=max_steps,
        )
        for sid, c_idx, a_idx in layout:
            ep_clean = episodes[c_idx]
            ep_asr_ok = True
            if a_idx is not None:
                ep_asr_ok = episodes[a_idx].reward_breakdown.official_score >= 1.0 - 1e-6
            if ep_clean.reward_breakdown.official_score < 1.0 - 1e-6 or not ep_asr_ok:
                buckets["tier_a_hard"].append(sid)
            elif any(e >= 0.35 for e in ep_clean.step_entropies):
                buckets["tier_b_boundary"].append(sid)
            else:
                buckets["tier_c_anchor"].append(sid)

        partial_file.write_text(json.dumps(buckets, indent=2), encoding="utf-8")
        probed = sum(len(v) for v in buckets.values())
        print(
            f"[GCP Mine] {probed}/{len(samples)} hard={len(buckets['tier_a_hard'])} "
            f"boundary={len(buckets['tier_b_boundary'])} anchor={len(buckets['tier_c_anchor'])} "
            f"chunk_s={time.monotonic() - chunk_t0:.1f}s",
            flush=True,
        )
        if (time.monotonic() - last_push_t) >= gcp_sync_s or time.monotonic() >= wall_deadline_t:
            last_push_t = time.monotonic()
            pusher.submit(partial_file, remote_partial, f"GCP partial ({probed}/{len(samples)})")
        if time.monotonic() >= wall_deadline_t and (start + len(batch)) < len(remaining):
            pusher.flush(timeout=600)
            print("[GCP Mine] Wall-cap hit during GCP probing; saved partial buckets to HF.", flush=True)
            return buckets, False

    if remaining:
        pusher.submit(partial_file, remote_partial, f"GCP partial ({sum(len(v) for v in buckets.values())}/{len(samples)})")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        f"[GCP Mine] Complete: hard={len(buckets['tier_a_hard'])} "
        f"boundary={len(buckets['tier_b_boundary'])} anchor={len(buckets['tier_c_anchor'])}",
        flush=True,
    )
    return buckets, True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--override-total-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate config, splits, and CPU imports without CUDA.")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.override_total_steps is not None:
        cfg["optim"]["total_steps"] = int(args.override_total_steps)
    _seed_everything(int(cfg.get("seed", 42)))

    if args.dry_run:
        train_ids = set(json.loads(Path(cfg["train_ids"]).read_text(encoding="utf-8"))["ids"])
        val_ids = set(json.loads(Path(cfg["val_ids"]).read_text(encoding="utf-8"))["ids"])
        assert not (train_ids & val_ids), "Train/val split overlap detected!"
        print(
            f"[DRY-RUN] OK stage={cfg['stage']} hf_repo={args.hf_repo or cfg.get('hf_repo')} "
            f"private={cfg.get('hf_private', False)} train={len(train_ids)} val={len(val_ids)}",
            flush=True,
        )
        return 0

    from huggingface_hub import get_token

    token = os.environ.get(ENV_TOKEN) or get_token()
    repo_id = resolve_repo_id(args.hf_repo or cfg.get("hf_repo"))
    hf_private = bool(cfg.get("hf_private", False))
    syncer = HFSyncer(repo_id, token=token, private=hf_private)
    syncer.preflight()

    hf_prefix = cfg["hf_prefix"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pusher = BackgroundPusher(syncer)

    try:
        rl_vllm: _VLLMRolloutEngine | None = None
        base_model_path = _ensure_base_model(cfg, out_dir, token)
        asr_cache_path = _ensure_asr_cache(cfg, syncer)

        # Mandatory HF Auto-Resume check (download_dir returns root/path_in_repo)
        resume_step = 0
        resume_dir: Path | None = None
        latest_remote = f"{hf_prefix}/checkpoints/latest"
        if syncer.exists(f"{latest_remote}/adapter_config.json"):
            resume_dir = syncer.download_dir(latest_remote, out_dir / "resumed_cache")
            state_file = resume_dir / "trainer_state.pt"
            if state_file.exists():
                resume_step = int(torch.load(state_file, map_location="cpu", weights_only=False)["step"])
                print(f"[RL] Resuming from HF {latest_remote} at step {resume_step}", flush=True)

        if resume_step >= int(cfg["optim"]["total_steps"]):
            print(f"[RL] Already reached target total_steps={cfg['optim']['total_steps']} (resume_step={resume_step}).", flush=True)
            return 0

        tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), trust_remote_code=True)
        catalog = load_catalog(cfg["train_split"], cfg["root"])
        train_ids = set(json.loads(Path(cfg["train_ids"]).read_text(encoding="utf-8"))["ids"])
        samples = [s for s in load_split(cfg["train_split"], cfg["root"]) if s.id in train_ids]
        sample_by_id = {s.id: s for s in samples}
        gold_map = _precompute_gold_hashes(samples, cfg["train_split"], cfg["root"])
        asr_by_id = _load_asr_cache(asr_cache_path)

        t0 = time.monotonic()
        last_sync_t = time.monotonic()
        max_wall_s = float(cfg["optim"].get("max_wall_minutes", 45)) * 60.0
        wall_deadline_t = t0 + max_wall_s

        init_dir: Path | None = None
        if cfg.get("init_student_adapter_hf") and not (
            resume_dir is not None and (resume_dir / "adapter_config.json").exists()
        ):
            init_dir = syncer.download_dir(cfg["init_student_adapter_hf"], out_dir / "init_cache")

        bucket_file = out_dir / "gcp_buckets.json"
        remote_bucket = f"{hf_prefix}/gcp_buckets.json"
        buckets: dict[str, list[str]] | None = None
        if bucket_file.exists():
            buckets = json.loads(bucket_file.read_text(encoding="utf-8"))
        elif syncer.exists(remote_bucket):
            dl_b = syncer.api.hf_hub_download(
                repo_id=syncer.repo_id,
                filename=remote_bucket,
                repo_type=syncer.repo_type,
                local_dir=str(out_dir / "_hf_buckets"),
            )
            shutil.copy2(dl_b, bucket_file)
            buckets = json.loads(bucket_file.read_text(encoding="utf-8"))
        elif _vllm_backend_enabled(cfg, "gcp"):
            max_new_gcp = int(cfg["rollout"]["max_new_tokens"])
            gcp_bs = max(1, int(cfg["rollout"].get("gcp_batch_size", 16)))
            vllm_util = float(cfg["rollout"].get("vllm_gcp_util", 0.72))
            gcp_vllm = _VLLMRolloutEngine(
                base_model_path,
                gpu_memory_utilization=vllm_util,
                enable_lora=bool(init_dir is not None),
                max_lora_rank=int(cfg["lora"]["r"]),
                enable_sleep_mode=False,
                seed=int(cfg.get("seed", 42)),
                max_num_seqs=max(32, gcp_bs * 2),
                vocab_size=len(tokenizer),
                stop_token_ids=_resolve_stop_token_ids(tokenizer, base_model_path),
            )
            try:
                vllm_gcp_fn = _make_vllm_step_gen_fn(
                    gcp_vllm,
                    tokenizer,
                    max_new_tokens=max_new_gcp,
                    do_sample=False,
                    lora_path_fn=(lambda: init_dir) if init_dir is not None else None,
                    lora_id_fn=lambda: 1,
                )
                buckets, gcp_done = _mine_gcp_buckets(
                    None,
                    tokenizer,
                    samples,
                    asr_by_id,
                    catalog,
                    gold_map,
                    cfg,
                    out_dir=out_dir,
                    pusher=pusher,
                    syncer=syncer,
                    hf_prefix=hf_prefix,
                    wall_deadline_t=wall_deadline_t,
                    batch_gen_override=vllm_gcp_fn,
                )
            finally:
                gcp_vllm.close()
            if not gcp_done:
                return 0
            bucket_file.write_text(json.dumps(buckets, indent=2), encoding="utf-8")
            pusher.submit(bucket_file, remote_bucket, "Save mined GCP buckets")
            pusher.flush(timeout=180)
            if _vllm_backend_enabled(cfg, "rl"):
                print(
                    "[GCP Mine] Pushed gcp_buckets.json; exiting cleanly so RL starts with a fresh vLLM engine.",
                    flush=True,
                )
                return 0

        rl_vllm: _VLLMRolloutEngine | None = None
        vram_gb = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
            if torch.cuda.is_available()
            else 0.0
        )
        if _vllm_backend_enabled(cfg, "rl"):
            offload_peft_default = vram_gb < 28.0
            default_rl_util = 0.62 if (vram_gb < 28.0 and cfg["rollout"].get("vllm_offload_peft", offload_peft_default)) else (0.44 if vram_gb < 28.0 else 0.45)
            rl_vllm = _VLLMRolloutEngine(
                base_model_path,
                gpu_memory_utilization=float(cfg["rollout"].get("vllm_rl_util", default_rl_util)),
                enable_lora=True,
                max_lora_rank=int(cfg["lora"]["r"]),
                enable_sleep_mode=True,
                seed=int(cfg.get("seed", 42)),
                max_num_seqs=max(24, int(cfg["rollout"].get("gen_micro_batch_size", 24))),
                vocab_size=len(tokenizer),
                stop_token_ids=_resolve_stop_token_ids(tokenizer, base_model_path),
            )
            rl_vllm.sleep()

        base = _load_base_causal_lm(base_model_path)

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(cfg["lora"]["r"]),
            lora_alpha=int(cfg["lora"]["alpha"]),
            lora_dropout=float(cfg["lora"]["dropout"]),
            target_modules=list(QWEN35_LORA_TARGET_MODULES),
            bias="none",
        )

        if resume_dir is not None and (resume_dir / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(base, str(resume_dir), adapter_name="default", is_trainable=True)
        elif init_dir is not None:
            model = PeftModel.from_pretrained(base, str(init_dir), adapter_name="default", is_trainable=True)
        else:
            model = get_peft_model(base, lora_cfg, adapter_name="default")

        if cfg["stage"] == "voice_opd":
            t_dir = syncer.download_dir(cfg["teacher_adapter_hf"], out_dir / "teacher_cache")
            model.load_adapter(str(t_dir), adapter_name="teacher", is_trainable=False)
        model.set_adapter("default")
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        _probe_fast_backbone_projection(model)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params, lr=float(cfg["optim"]["lr"]), weight_decay=float(cfg["optim"]["weight_decay"])
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(cfg["optim"]["warmup_steps"]),
            num_training_steps=int(cfg["optim"]["total_steps"]),
        )
        if resume_dir is not None and resume_step > 0 and (resume_dir / "trainer_state.pt").exists():
            st = torch.load(resume_dir / "trainer_state.pt", map_location="cpu", weights_only=False)
            optimizer.load_state_dict(st["optimizer"])
            scheduler.load_state_dict(st["scheduler"])

        if buckets is None:
            buckets, gcp_done = _mine_gcp_buckets(
                model,
                tokenizer,
                samples,
                asr_by_id,
                catalog,
                gold_map,
                cfg,
                out_dir=out_dir,
                pusher=pusher,
                syncer=syncer,
                hf_prefix=hf_prefix,
                wall_deadline_t=wall_deadline_t,
            )
            if not gcp_done:
                return 0
            bucket_file.write_text(json.dumps(buckets, indent=2), encoding="utf-8")
            pusher.submit(bucket_file, remote_bucket, "Save mined GCP buckets")
            pusher.flush(timeout=180)
            if _vllm_backend_enabled(cfg, "rl"):
                print(
                    "[GCP Mine] Pushed gcp_buckets.json; exiting cleanly so RL starts with a fresh vLLM engine.",
                    flush=True,
                )
                return 0

        rng = random.Random(int(cfg.get("seed", 42)) + resume_step)
        max_new = int(cfg["rollout"]["max_new_tokens"])
        temp = float(cfg["rollout"]["temperature"])
        top_p = float(cfg["rollout"]["top_p"])
        group_size = int(cfg["rollout"]["group_size"])
        gen_micro_bs = int(cfg["rollout"].get("gen_micro_batch_size", 12))
        max_steps = int(cfg["rollout"]["max_steps"])
        arpo_thresh = float(cfg["rollout"]["arpo_entropy_thresh"])
        reuse_on_policy_old_lp = float(cfg["lora"]["dropout"]) == 0.0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram_gb < 28.0:
                gen_micro_bs = min(gen_micro_bs, 12)
        log_path = out_dir / "metrics_history.jsonl"
        remote_metrics = f"{hf_prefix}/metrics_history.jsonl"
        if not log_path.exists() and syncer.exists(remote_metrics):
            dl_m = syncer.api.hf_hub_download(
                repo_id=syncer.repo_id,
                filename=remote_metrics,
                repo_type=syncer.repo_type,
                local_dir=str(out_dir / "_hf_metrics"),
            )
            shutil.copy2(dl_m, log_path)

        cur_step_ref = [0]
        cur_lora_dir_ref: list[Path | None] = [None]
        offload_peft = bool(cfg["rollout"].get("vllm_offload_peft", vram_gb < 28.0))
        if rl_vllm is not None:
            batch_sample_fn = _make_vllm_step_gen_fn(
                rl_vllm,
                tokenizer,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temp,
                top_p=top_p,
                lora_path_fn=lambda: cur_lora_dir_ref[0],
                lora_id_fn=lambda: cur_step_ref[0],
            )
        else:
            batch_sample_fn = _make_batch_step_gen_fn(
                model,
                tokenizer,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temp,
                top_p=top_p,
                micro_batch_size=gen_micro_bs,
            )

        for step in range(resume_step + 1, int(cfg["optim"]["total_steps"]) + 1):
            step_t0 = time.monotonic()
            model.set_adapter("default")
            model.eval()
            train_dev = next(model.parameters()).device
            active_groups: list[list[MultiTurnEpisode]] = []
            fallback_groups: list[list[MultiTurnEpisode]] = []
            target_bs = int(cfg["rollout"]["batch_size"])

            if rl_vllm is not None:
                cur_step_ref[0] = step
                step_lora_dir = out_dir / "_vllm_lora" / f"s{step}"
                _save_adapter_snapshot_for_vllm(model, step_lora_dir)
                cur_lora_dir_ref[0] = step_lora_dir
                old_snap = out_dir / "_vllm_lora" / f"s{step - 2}"
                if old_snap.exists():
                    shutil.rmtree(old_snap, ignore_errors=True)
                if offload_peft and train_dev.type == "cuda":
                    model.to("cpu")
                    gc.collect()
                    torch.cuda.empty_cache()
                rl_vllm.wake_up()

            try:
                for attempt in range(int(cfg["rollout"]["max_gen_batches"])):
                    need = max(1, target_bs - len(active_groups))
                    n_cand = need if attempt == 0 else min(target_bs, max(2, need * 2))
                    cand_ids = sample_weighted_prompt_ids(buckets, rng, n_cand)

                    cand_meta: list[tuple[Any, Any, str | None, bool, list[dict[str, Any]], str]] = []
                    turn0_prompts: list[str] = []
                    for sid in cand_ids:
                        s_clean = sample_by_id[sid]
                        use_asr = rng.random() < float(cfg["rollout"]["asr_mix_prob"]) and (sid in asr_by_id)
                        s_eff, diff = build_sample_with_asr(s_clean, asr_by_id.get(sid), inject_asr=use_asr)
                        g_calls, g_hash = gold_map[sid]
                        cand_meta.append((s_eff, s_clean, diff, use_asr, g_calls, g_hash))
                        turn0_prompts.append(_prompt_for_sample(s_eff, catalog, tokenizer, diff))

                    if rl_vllm is not None:
                        turn0_resps = rl_vllm.generate_prompts(
                            turn0_prompts,
                            max_new_tokens=max_new,
                            do_sample=True,
                            temperature=temp,
                            top_p=top_p,
                            num_return_sequences=group_size,
                            lora_path=cur_lora_dir_ref[0],
                            lora_id=step,
                        )
                    else:
                        turn0_resps = _generate_batch_prompts(
                            model,
                            tokenizer,
                            turn0_prompts,
                            max_new_tokens=max_new,
                            do_sample=True,
                            temperature=temp,
                            top_p=top_p,
                            num_return_sequences=group_size,
                            micro_batch_size=gen_micro_bs,
                        )

                    roll_reqs: list[RolloutRequest] = []
                    for c_idx, (s_eff, s_clean, diff, use_asr, g_calls, g_hash) in enumerate(cand_meta):
                        for k in range(group_size):
                            roll_reqs.append(
                                RolloutRequest(
                                    sample=s_eff,
                                    gold_calls=g_calls,
                                    gold_hash=g_hash,
                                    clean_sample=s_clean,
                                    asr_phonetic_diff=diff,
                                    use_asr=use_asr,
                                    prefilled_step0=turn0_resps[c_idx * group_size + k],
                                )
                            )

                    flat_eps = execute_batched_rollouts(
                        roll_reqs,
                        catalog,
                        cfg["train_split"],
                        cfg["root"],
                        batch_sample_fn,
                        tokenizer,
                        max_steps=max_steps,
                    )
                    batch_groups = [flat_eps[i * group_size : (i + 1) * group_size] for i in range(len(cand_meta))]

                    # Batched ARPO post-tool branch sampling across all groups in this attempt
                    fork_reqs: list[RolloutRequest] = []
                    fork_targets: list[tuple[int, int]] = []
                    max_forks = 2
                    for grp_idx, (grp, meta) in enumerate(zip(batch_groups, cand_meta)):
                        anchor_ep = select_arpo_anchor(grp, entropy_thresh=arpo_thresh)
                        if anchor_ep is None:
                            continue
                        s_eff, s_clean, diff, use_asr, g_calls, g_hash = meta
                        fork_targets.append((grp_idx, len(fork_reqs)))
                        for _ in range(max_forks):
                            fork_reqs.append(
                                RolloutRequest(
                                    sample=s_eff,
                                    gold_calls=g_calls,
                                    gold_hash=g_hash,
                                    clean_sample=s_clean,
                                    asr_phonetic_diff=diff,
                                    use_asr=use_asr,
                                    forced_prefix_calls=anchor_ep.first_step_calls,
                                )
                            )
                    if fork_reqs:
                        fork_eps = execute_batched_rollouts(
                            fork_reqs,
                            catalog,
                            cfg["train_split"],
                            cfg["root"],
                            batch_sample_fn,
                            tokenizer,
                            max_steps=max_steps,
                        )
                        for grp_idx, start_pos in fork_targets:
                            batch_groups[grp_idx].extend(fork_eps[start_pos : start_pos + max_forks])

                    for grp, meta in zip(batch_groups, cand_meta):
                        g_calls, g_hash = meta[4], meta[5]
                        assign_group_gigpo_advantages(
                            grp,
                            g_calls,
                            g_hash,
                            omega=float(cfg["rollout"]["omega"]),
                            gamma=float(cfg["rollout"]["gamma"]),
                            remove_std=bool(cfg["rollout"]["remove_std"]),
                        )
                    fallback_groups.extend(batch_groups)
                    active_groups.extend(filter_active_groups(batch_groups))
                    if len(active_groups) >= target_bs:
                        active_groups = active_groups[:target_bs]
                        break
            finally:
                if rl_vllm is not None:
                    rl_vllm.sleep()
                    if offload_peft and train_dev.type == "cuda":
                        model.to(train_dev)

            if not active_groups:
                active_groups = fallback_groups[:target_bs] if cfg["stage"] == "voice_opd" else []
            episodes = [ep for grp in active_groups for ep in grp]
            if not episodes:
                print(
                    f"[step {step}] skipped update (0 non-degenerate groups across {cfg['rollout']['max_gen_batches']} attempts)",
                    flush=True,
                )
                if step == int(cfg["optim"]["total_steps"]):
                    _save_and_push_ckpt(
                        model, tokenizer, optimizer, scheduler, out_dir, step, {"skipped_final_step": True}, pusher, hf_prefix
                    )
                    return 0
                if (time.monotonic() - t0) >= max_wall_s and step - 1 > resume_step:
                    _save_and_push_ckpt(
                        model, tokenizer, optimizer, scheduler, out_dir, step - 1, {"skipped_step": step}, pusher, hf_prefix
                    )
                    print(f"[RL] Wall-cap reached on skipped step {step}; next Colab chunk will auto-resume.", flush=True)
                    return 0
                continue

            pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0))
            nograd_max_tokens = int(cfg["optim"].get("nograd_micro_max_tokens", 20480 if vram_gb >= 35.0 else 8192))
            nograd_max_rows = int(cfg["optim"].get("nograd_micro_max_rows", 4))
            grad_max_tokens = int(cfg["optim"].get("grad_micro_max_tokens", 14336 if vram_gb >= 35.0 else 4096))
            grad_max_rows = int(cfg["optim"].get("grad_micro_max_rows", 2))

            with torch.no_grad():
                ep_items = [(ep.input_ids, ep.loss_mask) for ep in episodes]
                if not reuse_on_policy_old_lp:
                    model.set_adapter("default")
                    old_lps: list[torch.Tensor | None] = list(
                        _compute_logprobs_packed(
                            model,
                            ep_items,
                            max_tokens=nograd_max_tokens,
                            max_rows=nograd_max_rows,
                            pad_token_id=pad_id,
                            to_cpu=True,
                        )
                    )
                else:
                    old_lps = [None] * len(episodes)

                if cfg["stage"] == "voice_opd":
                    model.set_adapter("teacher")
                    ref_or_teach_lps = _compute_teacher_batch_aligned_logprobs(
                        model,
                        episodes,
                        max_tokens=nograd_max_tokens,
                        max_rows=nograd_max_rows,
                        pad_token_id=pad_id,
                        to_cpu=True,
                    )
                else:
                    with model.disable_adapter():
                        ref_or_teach_lps = _compute_logprobs_packed(
                            model,
                            ep_items,
                            max_tokens=nograd_max_tokens,
                            max_rows=nograd_max_rows,
                            pad_token_id=pad_id,
                            to_cpu=True,
                        )

            model.set_adapter("default")
            model.train()
            optimizer.zero_grad()
            dev = next(model.parameters()).device
            loss_acc, kl_acc, clip_acc = [], [], []
            n_eps = len(episodes)

            grad_queue: list[list[int]] = _pack_token_budget_batches(
                [len(ep.input_ids) for ep in episodes],
                max_tokens=grad_max_tokens,
                max_rows=grad_max_rows,
            )
            while grad_queue:
                mb_idxs = grad_queue.pop(0)
                mb_eps = [episodes[i] for i in mb_idxs]
                try:
                    cur_lps = _compute_batch_token_logprobs(
                        model,
                        [ep.input_ids for ep in mb_eps],
                        loss_masks=[ep.loss_mask for ep in mb_eps],
                        pad_token_id=pad_id,
                    )
                except torch.cuda.OutOfMemoryError:
                    if len(mb_idxs) <= 1:
                        raise
                    torch.cuda.empty_cache()
                    mid = max(1, len(mb_idxs) // 2)
                    grad_queue = [mb_idxs[:mid], mb_idxs[mid:]] + grad_queue
                    continue

                m = len(mb_idxs)
                mb_max_len = max(lp.shape[1] for lp in cur_lps)
                cur_lp = torch.stack([F.pad(lp[0], (0, mb_max_len - lp.shape[1]), value=0.0) for lp in cur_lps], dim=0)
                if old_lps[mb_idxs[0]] is None:
                    o_lp = cur_lp.detach()
                else:
                    o_lp = torch.stack(
                        [F.pad(old_lps[i][0].to(dev), (0, mb_max_len - old_lps[i].shape[1]), value=0.0) for i in mb_idxs],
                        dim=0,
                    )
                rt_lp = torch.stack(
                    [F.pad(ref_or_teach_lps[i][0].to(dev), (0, mb_max_len - ref_or_teach_lps[i].shape[1]), value=0.0) for i in mb_idxs],
                    dim=0,
                )
                mask = torch.tensor(
                    [
                        ep.loss_mask[: lp.shape[1]] + ([0] * (mb_max_len - lp.shape[1]))
                        for ep, lp in zip(mb_eps, cur_lps)
                    ],
                    dtype=torch.float32,
                    device=dev,
                )
                adv = torch.cat(
                    [
                        F.pad(
                            _build_token_advantage_tensor(ep, lp.shape[1], dev),
                            (0, mb_max_len - lp.shape[1]),
                            value=0.0,
                        )
                        for ep, lp in zip(mb_eps, cur_lps)
                    ],
                    dim=0,
                )

                if cfg["stage"] == "voice_opd":
                    adv = compute_opd_reverse_kl_advantages(
                        cur_lp,
                        rt_lp,
                        adv,
                        mask,
                        alpha_ep=float(cfg["opd"]["alpha_ep"]),
                        clip_val=float(cfg["opd"]["rkl_clip"]),
                    )

                pg_loss, st = compute_gspo_clip_higher_loss(
                    cur_lp,
                    o_lp,
                    adv,
                    mask,
                    clip_low=float(cfg["optim"]["clip_low"]),
                    clip_high=float(cfg["optim"]["clip_high"]),
                )
                kl_loss = compute_low_var_kl_loss(cur_lp, rt_lp, mask)
                total = (pg_loss + float(cfg["optim"]["kl_coef"]) * kl_loss) * (float(m) / max(1, n_eps))
                total.backward()
                loss_acc.append(float(pg_loss.item()) * m)
                kl_acc.append(float(kl_loss.item()) * m)
                clip_acc.append(float(st["clipped_frac"]) * m)

            torch.nn.utils.clip_grad_norm_(trainable_params, float(cfg["optim"]["max_grad_norm"]))
            optimizer.step()
            scheduler.step()

            mean_rew = sum(e.episode_reward for e in episodes) / n_eps
            mean_off = sum(e.reward_breakdown.official_score for e in episodes) / n_eps
            mean_seq_len = sum(len(e.input_ids) for e in episodes) / n_eps
            metrics = {
                "step": step,
                "mean_reward": round(mean_rew, 4),
                "mean_official": round(mean_off, 4),
                "pg_loss": round(sum(loss_acc) / n_eps, 5),
                "kl": round(sum(kl_acc) / n_eps, 5),
                "clip_frac": round(sum(clip_acc) / n_eps, 4),
                "active_groups": len(active_groups),
                "n_episodes": n_eps,
                "mean_seq_len": round(mean_seq_len, 1),
                "step_s": round(time.monotonic() - step_t0, 2),
            }
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(metrics) + "\n")
            print(f"[step {step}/{cfg['optim']['total_steps']}] {json.dumps(metrics)}", flush=True)

            due_step = (step % int(cfg["optim"]["save_steps"]) == 0) or (step == int(cfg["optim"]["total_steps"]))
            due_time = (time.monotonic() - last_sync_t) >= float(cfg["optim"]["sync_interval_s"])
            due_wall = (time.monotonic() - t0) >= max_wall_s
            if due_step or due_time or due_wall:
                last_sync_t = time.monotonic()
                _save_and_push_ckpt(model, tokenizer, optimizer, scheduler, out_dir, step, metrics, pusher, hf_prefix)
                pusher.submit(log_path, f"{hf_prefix}/metrics_history.jsonl", f"RL metrics step {step}")
                if due_wall and step < int(cfg["optim"]["total_steps"]):
                    print(f"[RL] Wall-cap reached cleanly at step {step}; next Colab chunk will auto-resume.", flush=True)
                    return 0
            elif step % 5 == 0:
                pusher.submit(log_path, f"{hf_prefix}/metrics_history.jsonl", f"RL metrics step {step}")
    finally:
        if locals().get("rl_vllm") is not None:
            try:
                rl_vllm.close()  # type: ignore[union-attr]
            except Exception:
                pass
        pusher.flush(timeout=900)
        pusher.close(timeout=30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
