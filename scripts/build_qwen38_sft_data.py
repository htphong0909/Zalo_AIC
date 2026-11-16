#!/usr/bin/env python3
"""Build leak-free SFT dataset v2 for Qwen3.5-4B (and Qwen3.8-4B-Distill).

Incorporates verified paper-aligned data engineering and fixes all V1 audit bugs:
1. Native Qwen3.5 XML with zero MiniCPM5 `<function name=...>` history leakage.
2. APIGen-MT multi-intent chaining with `include_step_1=True` (trains both Step 0 and Step 1).
3. Clean When2Call Step-0 ASK (no `*_id` / conditional / multi-call leaks) + Post-Lookup ASK.
4. Schema-validated Recovery (empty search retry + xref repair matching runtime `hint` format).
5. Hammer (ICLR 2025) & Tool-N1 (ICLR 2026) Irrelevance OOS counterfactuals + schema masking.
6. AURA (ASRU 2025) real/simulated ASR transcript augmentation for spoken Vietnamese resilience.
7. Balanced terminal stop sampling + zero Dev leakage assertions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gzip
import json
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_lfm3_data import apply_hammer_masking, simulate_ctc_transcript
from scripts.build_minicpm5_dataset_v2 import resolve_step_tools
from src.agent.prompts import build_system_prompt
from src.agent.qwen35_sft import _rewrite_system_for_qwen35, step_to_qwen35_example
from src.agent.sft_dataset import build_sft_trajectory, unroll_trajectory_steps
from src.core.dataset import load_catalog, load_split
from src.core.schemas import ASK_MISSING_INFO, Sample
from src.data_gen.anchor_binding_gen import generate_anchor_binding_samples
from src.data_gen.apigen_mt_gen import generate_apigen_multi_intent_samples
from src.data_gen.execution_verifier import verify_step_validity
from src.data_gen.prompt_utils import build_sample_prefix_messages
from src.data_gen.recovery_augmenter import (
    create_empty_search_recovery_step,
    create_xref_repair_recovery_step,
)
from src.data_gen.when2call_gen import (
    generate_post_lookup_ask_samples,
    generate_when2call_contrastive_samples,
)

# Overlapping functional domains removed together during Hammer irrelevance OOS synthesis
# so no remaining distractor app can plausibly satisfy the user's request.
_APP_CONFLICT_CLUSTERS: tuple[frozenset[str], ...] = (
    frozenset({"sms", "chat_ott", "email"}),
    frozenset({"banking", "ewallet", "bill_payment"}),
    frozenset({"reminders", "tasks", "notes", "calendar"}),
    frozenset({"music", "podcast_radio", "tv"}),
    frozenset({"smart_home", "home_automation", "door_lock", "home_security", "robot_vacuum", "appliances", "home_energy"}),
)


def _load_exclude_ids(dev_split_path: str | Path | None) -> set[str]:
    if not dev_split_path:
        return set()
    p = Path(dev_split_path)
    if not p.exists():
        return set()
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return {str(x) for x in raw}
    return {str(x) for x in (raw.get("dev_ids") or raw.get("ids") or [])}


def _load_asr_map(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row.get("id"):
                    out[str(row["id"])] = row
    return out


def _is_leaked_sample_id(sid: str, exclude_ids: set[str]) -> bool:
    if not sid or not exclude_ids:
        return False
    if sid in exclude_ids:
        return True
    parts = sid.split("_")
    return any(f"train_{p}" in exclude_ids or p in exclude_ids for p in parts if p.isdigit())


def _expand_conflict_apps(used_apps: set[str]) -> set[str]:
    expanded = set(used_apps)
    for cluster in _APP_CONFLICT_CLUSTERS:
        if expanded & cluster:
            expanded |= cluster
    return expanded


def generate_irrelevance_oos_samples(
    train_samples: list[Sample],
    catalog: dict[str, dict[str, Any]],
    *,
    count: int = 220,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Hammer / Tool-N1 irrelevance augmentation: drop needed app cluster -> verified OOS refusal."""
    if count <= 0:
        return []
    rng = random.Random(seed)
    candidates: list[Sample] = []
    for s in train_samples:
        if not s.label or not s.label.tool_calls or s.history:
            continue
        if any(c.name == ASK_MISSING_INFO for c in s.label.tool_calls):
            continue
        candidates.append(s)

    rng.shuffle(candidates)
    out: list[dict[str, Any]] = []
    for idx, s in enumerate(candidates):
        if len(out) >= count:
            break
        ctx = copy.deepcopy(s.context or {})
        installed = dict(ctx.get("installed_apps") or {})
        used_apps = {c.name.split(".", 1)[0] for c in s.label.tool_calls if "." in c.name}
        drop_apps = _expand_conflict_apps(used_apps)
        remaining = {k: v for k, v in installed.items() if k not in drop_apps}
        if len(remaining) < 3 or len(remaining) == len(installed):
            continue

        ctx["installed_apps"] = remaining
        masked_sample = Sample(
            id=f"irr_{s.id}_{idx}",
            audio_text=s.audio_text,
            context=ctx,
            history=[],
        )
        prompt_msgs = build_sample_prefix_messages(masked_sample, catalog)
        out.append(
            {
                "id": f"irr_oos_{s.id}_{idx}",
                "sample_id": s.id,
                "step_index": 0,
                "step_category": "oos",
                "prompt_messages": prompt_msgs,
                "step_calls": [],
                "tools": sorted(remaining.keys()),
                "sample_context": ctx,
            }
        )
    return out


def _maybe_apply_asr_text(
    step: dict[str, Any],
    sid: str,
    src_tag: str,
    *,
    asr_map: dict[str, dict[str, Any]],
    sim_ratio: float,
    rng: random.Random,
) -> tuple[dict[str, Any], bool]:
    """Apply real or simulated spoken ASR transcript noise to eligible step-0 user utterances."""
    if src_tag not in ("base_train", "hammer_irrelevance_oos") or int(step.get("step_index") or 0) != 0:
        return step, False
    pmsgs = step.get("prompt_messages") or []
    if not pmsgs or pmsgs[-1].get("role") != "user":
        return step, False
    orig_u = str(pmsgs[-1].get("content") or "").strip()
    if not orig_u:
        return step, False

    noisy_u: str | None = None
    entry = asr_map.get(sid)
    if entry and entry.get("audio_text"):
        cand = str(entry["audio_text"]).strip()
        if cand and cand != orig_u:
            noisy_u = cand
    elif sim_ratio > 0 and rng.random() < sim_ratio:
        cand = simulate_ctc_transcript(orig_u)
        if cand and cand != orig_u:
            noisy_u = cand

    if not noisy_u:
        return step, False

    out_step = copy.deepcopy(step)
    out_step["prompt_messages"][-1]["content"] = noisy_u
    out_step["id"] = f"{step.get('id', sid)}_asr"
    return out_step, True


def _prepare_hammer_masked_step(
    step: dict[str, Any],
    ctx: dict[str, Any] | None,
    catalog: dict[str, dict[str, Any]],
    *,
    rng: random.Random,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Rewrite prompt system message to qwen35 first, then apply Hammer schema masking."""
    step_copy = copy.deepcopy(step)
    tools = resolve_step_tools(step_copy, ctx, catalog)
    for m in step_copy.get("prompt_messages") or []:
        if m.get("role") == "system":
            m["content"] = _rewrite_system_for_qwen35(str(m.get("content") or ""), ctx)
    if not any(m.get("role") == "system" for m in (step_copy.get("prompt_messages") or [])) and ctx:
        step_copy.setdefault("prompt_messages", []).insert(
            0,
            {"role": "system", "content": build_system_prompt(Sample(id="ctx", context=ctx), prompt_style="qwen35")},
        )
    step_copy["tools"] = tools
    masked = apply_hammer_masking(step_copy, rng=rng)
    masked["id"] = f"{step.get('id', 'step')}_hmask"
    return masked, list(masked.get("tools") or [])


def build_qwen38_sft_dataset(
    split_name: str = "train",
    root: str | Path = "data",
    exclude_ids_path: str | Path = "configs/splits/dev_v1.json",
    out_dir: str | Path = "data/sft",
    out_stem: str = "qwen35_train_sft_v2",
    *,
    enable_thinking: bool = False,
    when2call_step0_count: int = 250,
    when2call_post_count: int = 450,
    anchor_count: int = 400,
    apigen_count: int = 550,
    recovery_count: int = 250,
    irrelevance_oos_count: int = 220,
    keep_single_stop_ratio: float = 0.25,
    hammer_mask_prob: float = 0.12,
    simulated_asr_ratio: float = 0.15,
    asr_transcripts_path: str | Path | None = None,
    seed: int = 42,
) -> tuple[int, Path]:
    """Assemble and verify leak-free SFT examples in Qwen3.5/Qwen3.8 format."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_path / f"{out_stem}.jsonl"
    out_gz = out_path / f"{out_stem}.jsonl.gz"

    catalog = load_catalog(split_name, root)
    all_samples = load_split(split_name, root)
    exclude_ids = _load_exclude_ids(exclude_ids_path)
    asr_map = _load_asr_map(asr_transcripts_path)
    train_samples = [s for s in all_samples if s.id not in exclude_ids]
    sample_map = {s.id: s for s in train_samples}
    rng = random.Random(seed)

    verified: list[tuple[dict[str, Any], dict[str, Any] | None, str, str]] = []
    act_pool: list[tuple[dict[str, Any], dict[str, Any] | None, str]] = []

    def _add_if_valid(step: dict[str, Any], source_tag: str) -> None:
        if not verify_step_validity(step, catalog):
            return
        sid = str(step.get("sample_id") or step.get("id") or "")
        if _is_leaked_sample_id(sid, exclude_ids):
            return
        ctx = step.get("sample_context") or (sample_map[sid].context if sid in sample_map else None)
        verified.append((step, ctx, sid, source_tag))

    for sample in train_samples:
        traj = build_sft_trajectory(sample, catalog)
        gold_len = len(sample.label.tool_calls) if sample.label else 0
        for step in unroll_trajectory_steps(traj, split=split_name, root=str(root)):
            if not verify_step_validity(step, catalog):
                continue
            is_trivial_stop = (
                not step.get("step_calls")
                and step.get("step_category") != "oos"
                and step.get("step_index", 0) == 1
                and gold_len == 1
            )
            if is_trivial_stop and rng.random() > keep_single_stop_ratio:
                continue
            verified.append((step, sample.context, sample.id, "base_train"))
            if step.get("step_calls") and step.get("step_category") == "act":
                act_pool.append((step, sample.context, sample.id))

    for st in generate_when2call_contrastive_samples(train_samples, catalog, count=when2call_step0_count, seed=seed):
        _add_if_valid(st, "when2call_step0")
    for st in generate_post_lookup_ask_samples(train_samples, catalog, count=when2call_post_count, seed=seed + 11):
        _add_if_valid(st, "when2call_post_lookup")
    for st in generate_anchor_binding_samples(train_samples, catalog, count=anchor_count, seed=seed):
        _add_if_valid(st, "anchor_binding")
    for st in generate_apigen_multi_intent_samples(
        train_samples, catalog, count=apigen_count, seed=seed, include_step_1=True
    ):
        _add_if_valid(st, "apigen_mt")
    for st in generate_irrelevance_oos_samples(train_samples, catalog, count=irrelevance_oos_count, seed=seed + 23):
        _add_if_valid(st, "hammer_irrelevance_oos")

    # Balanced Recovery synthesis (empty_search + catalog-verified xref_repair)
    rng.shuffle(act_pool)
    empty_cap = max(1, recovery_count // 2)
    xref_cap = recovery_count - empty_cap
    empty_added = 0
    xref_added = 0
    used_xref_ids: set[str] = set()
    for st, ctx, sid in act_pool:
        if empty_added + xref_added >= recovery_count:
            break
        step_id = str(st.get("id") or sid)
        if empty_added < empty_cap:
            rec_empty = create_empty_search_recovery_step(st)
            if rec_empty is not None and verify_step_validity(rec_empty, catalog):
                verified.append((rec_empty, ctx, sid, "recovery_empty_search"))
                empty_added += 1
                continue
        if xref_added < xref_cap and step_id not in used_xref_ids:
            rec_xref = create_xref_repair_recovery_step(st, require_grounded_context=True, catalog=catalog)
            if rec_xref is not None and verify_step_validity(rec_xref, catalog):
                verified.append((rec_xref, ctx, sid, "recovery_xref"))
                used_xref_ids.add(step_id)
                xref_added += 1
    # Top up remaining recovery budget without duplicating already-augmented steps
    for st, ctx, sid in act_pool:
        if empty_added + xref_added >= recovery_count:
            break
        step_id = str(st.get("id") or sid)
        if step_id in used_xref_ids:
            continue
        rec_xref = create_xref_repair_recovery_step(st, require_grounded_context=True, catalog=catalog)
        if rec_xref is not None and verify_step_validity(rec_xref, catalog):
            verified.append((rec_xref, ctx, sid, "recovery_xref"))
            used_xref_ids.add(step_id)
            xref_added += 1

    # Add optional Hammer schema-masked counterfactuals & AURA ASR-noisy variants
    final_entries: list[tuple[dict[str, Any], dict[str, Any] | None, str, str, list[dict[str, Any]] | None]] = []
    asr_count = 0
    hmask_count = 0
    for step, ctx, sid, src_tag in verified:
        noisy_step, used_asr = _maybe_apply_asr_text(
            step,
            sid,
            src_tag,
            asr_map=asr_map,
            sim_ratio=simulated_asr_ratio,
            rng=rng,
        )
        if used_asr:
            asr_count += 1
        final_entries.append((noisy_step, ctx, sid, src_tag, None))

        if (
            hammer_mask_prob > 0
            and src_tag == "base_train"
            and step.get("step_category") == "act"
            and step.get("step_calls")
            and rng.random() < hammer_mask_prob
        ):
            masked_step, masked_tools = _prepare_hammer_masked_step(step, ctx, catalog, rng=rng)
            final_entries.append((masked_step, ctx, sid, "hammer_masked", masked_tools))
            hmask_count += 1

    cat_counts: Counter[str] = Counter()
    src_counts: Counter[str] = Counter()
    rng.shuffle(final_entries)

    with open(out_jsonl, "w", encoding="utf-8") as fj, gzip.open(out_gz, "wt", encoding="utf-8") as fgz:
        for idx, (step, ctx, sid, src_tag, pre_tools) in enumerate(final_entries):
            assert not _is_leaked_sample_id(sid, exclude_ids), f"Dev leakage detected: {sid}"
            tools = pre_tools if pre_tools is not None else resolve_step_tools(step, ctx, catalog)
            q_ex = step_to_qwen35_example(
                step,
                ctx,
                tools,
                catalog,
                enable_thinking=enable_thinking,
                rng=random.Random(seed + idx),
            )
            for pm in q_ex["prompt_messages"]:
                if pm.get("role") == "assistant":
                    assert "<function name=" not in str(pm.get("content") or ""), (
                        f"MiniCPM5 XML leaked into prompt assistant turn for {step.get('id')}"
                    )

            row = {
                "id": str(step.get("id") or f"q35_{idx}"),
                "sample_id": sid,
                "stratum": str(step.get("stratum") or src_tag),
                "step_category": str(step.get("step_category") or "act"),
                "source": src_tag,
                **q_ex,
            }
            line = json.dumps(row, ensure_ascii=False) + "\n"
            fj.write(line)
            fgz.write(line)
            cat_counts[row["step_category"]] += 1
            src_counts[src_tag] += 1

    print(
        json.dumps(
            {
                "total_examples": len(final_entries),
                "excluded_ids": len(exclude_ids),
                "enable_thinking": enable_thinking,
                "asr_augmented": asr_count,
                "hammer_masked": hmask_count,
                "categories": dict(cat_counts),
                "sources": dict(src_counts),
                "out_gz": str(out_gz),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return len(final_entries), out_gz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--root", default="data")
    parser.add_argument("--exclude-ids", default="configs/splits/dev_v1.json")
    parser.add_argument("--out-dir", default="data/sft")
    parser.add_argument("--out-stem", default="qwen35_train_sft_v2")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--when2call-step0", type=int, default=250)
    parser.add_argument("--when2call-post", type=int, default=450)
    parser.add_argument("--anchor", type=int, default=400)
    parser.add_argument("--apigen", type=int, default=550)
    parser.add_argument("--recovery", type=int, default=250)
    parser.add_argument("--irrelevance-oos", type=int, default=220)
    parser.add_argument("--keep-single-stop-ratio", type=float, default=0.25)
    parser.add_argument("--hammer-mask-prob", type=float, default=0.12)
    parser.add_argument("--simulated-asr-ratio", type=float, default=0.15)
    parser.add_argument("--asr-transcripts", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_qwen38_sft_dataset(
        split_name=args.split,
        root=args.root,
        exclude_ids_path=args.exclude_ids,
        out_dir=args.out_dir,
        out_stem=args.out_stem,
        enable_thinking=args.enable_thinking,
        when2call_step0_count=args.when2call_step0,
        when2call_post_count=args.when2call_post,
        anchor_count=args.anchor,
        apigen_count=args.apigen,
        recovery_count=args.recovery,
        irrelevance_oos_count=args.irrelevance_oos,
        keep_single_stop_ratio=args.keep_single_stop_ratio,
        hammer_mask_prob=args.hammer_mask_prob,
        simulated_asr_ratio=args.simulated_asr_ratio,
        asr_transcripts_path=args.asr_transcripts,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
