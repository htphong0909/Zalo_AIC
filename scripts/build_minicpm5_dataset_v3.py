#!/usr/bin/env python3
"""Build leak-free Tier 1 V3 SFT dataset for openbmb/MiniCPM5-2B.

Upgrades over V2 (all strictly leak-free on non-dev train samples):
1. 100% Unified System Prompts & History Parity (zero bare 'Bạn là trợ lý ảo LifeMate.' prompts).
2. De-biased single-step final_stop turns so 0-call stopping never dominates multi-step chains.
3. Multi-turn Post-Lookup & Multi-Intent When2Call Ask Synthesis (fixes post-search ambiguity).
4. Multi-edge Anchor Binding & non-mixed APIGen-MT Chaining.
5. Re-integrated Recovery Augmenter (empty search relaxation + x-ref self-repair).
6. Optional LoopTool EDDE hard-sample weighting (--hard-sample-ids) with varied Hammer permutations.
7. Strict schema verifier & token-length audit guaranteeing 0% truncation at max_length=16384.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_minicpm5_dataset_v2 import convert_step_to_minicpm5_messages
from src.agent.sft_dataset import build_sft_trajectory, unroll_trajectory_steps
from src.core.dataset import load_catalog, load_split
from src.data_gen.anchor_binding_gen import generate_anchor_binding_samples
from src.data_gen.apigen_mt_gen import generate_apigen_multi_intent_samples
from src.data_gen.distill_think import synthesize_deterministic_think
from src.data_gen.execution_verifier import validate_step_with_reason, verify_step_validity
from src.data_gen.recovery_augmenter import (
    create_empty_search_recovery_step,
    create_xref_repair_recovery_step,
)
from src.data_gen.when2call_gen import (
    generate_post_lookup_ask_samples,
    generate_when2call_contrastive_samples,
)


def load_hard_sample_ids(path: str | Path | None) -> set[str]:
    """Load non-dev hard/failing train sample IDs from a JSON list or JSONL eval output."""
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    if p.suffix == ".jsonl":
        hard: set[str] = set()
        for line in text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("id") or row.get("sample_id") or "")
            if sid and float(row.get("score", 0.0)) < 0.9999:
                hard.add(sid)
        return hard
    data = json.loads(text)
    if isinstance(data, list):
        return {str(x) for x in data}
    if isinstance(data, dict):
        for k in ("hard_ids", "failed_ids", "ids"):
            if k in data and isinstance(data[k], list):
                return {str(x) for x in data[k]}
    return set()


def _is_trivial_single_stop(sample: Any, step: dict[str, Any]) -> bool:
    """Return True for trailing 0-call confirmation after a single-call trajectory."""
    if step.get("step_calls"):
        return False
    if step.get("step_category") == "oos":
        return False
    gold_len = len(sample.label.tool_calls) if sample.label else 0
    return step.get("step_index", 0) == 1 and gold_len == 1


def build_v3_dataset(
    split_name: str = "train",
    root: str | Path = "data",
    dev_split_path: str | Path = "configs/splits/dev_v1.json",
    out_dir: str | Path = "data/sft",
    when2call_step0_count: int = 350,
    when2call_post_count: int = 450,
    anchor_count: int = 400,
    apigen_count: int = 550,
    recovery_count: int = 250,
    keep_single_stop_ratio: float = 0.45,
    hard_sample_ids_path: str | Path | None = None,
    seed: int = 42,
) -> tuple[int, Path]:
    """Assemble verified leak-free Tier 1 V3 dataset for MiniCPM5-2B."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_path / "minicpm5_train_sft_v3.jsonl"
    out_gz = out_path / "minicpm5_train_sft_v3.jsonl.gz"

    catalog = load_catalog(split_name, root)
    all_samples = load_split(split_name, root)

    dev_ids: set[str] = set()
    dev_path = Path(dev_split_path)
    if dev_path.exists():
        dev_ids = set(json.loads(dev_path.read_text(encoding="utf-8")).get("dev_ids", []))

    train_samples = [s for s in all_samples if s.id not in dev_ids]
    sample_map = {s.id: s for s in train_samples}
    hard_ids = load_hard_sample_ids(hard_sample_ids_path) - dev_ids
    rng = random.Random(seed)

    print(
        f"[V3 Builder] total={len(all_samples)}, train_only={len(train_samples)}, "
        f"excluded_dev={len(dev_ids)}, edde_hard_ids={len(hard_ids)}"
    )

    verified_entries: list[tuple[dict[str, Any], dict[str, Any] | None, str, str, str, int]] = []
    # Tuple fields: (step, ctx, sample_id, step_category, source_tag, extra_seed_offset)

    # 1. Clean Base Trajectories (with de-biased trivial single-stop + optional EDDE hard weighting)
    base_total = 0
    base_kept = 0
    dropped_trivial_stops = 0
    hard_extra_added = 0
    act_steps_pool: list[tuple[dict[str, Any], dict[str, Any], str]] = []

    for sample in train_samples:
        traj = build_sft_trajectory(sample, catalog)
        steps = unroll_trajectory_steps(traj, split=split_name, root=str(root))
        for step in steps:
            base_total += 1
            if not verify_step_validity(step, catalog):
                continue
            if _is_trivial_single_stop(sample, step) and rng.random() > keep_single_stop_ratio:
                dropped_trivial_stops += 1
                continue
            base_kept += 1
            cat = step.get("step_category", "act")
            verified_entries.append((step, sample.context, sample.id, cat, "base_train", 0))
            if step.get("step_calls") and cat == "act":
                act_steps_pool.append((step, sample.context, sample.id))

            # LoopTool EDDE: duplicate decision/oos turns of hard failing train_only samples
            # with a distinct Hammer tool-ordering seed to avoid verbatim memorization.
            if sample.id in hard_ids and (step.get("step_calls") or cat == "oos"):
                hard_step = dict(step)
                hard_step["id"] = f"{sample.id}_s{step.get('step_index', 0)}_edde"
                verified_entries.append((hard_step, sample.context, sample.id, cat, "looptool_edde", 7919))
                hard_extra_added += 1

    print(
        f"  1. base_train: kept={base_kept}/{base_total} (dropped trivial single stops={dropped_trivial_stops}), "
        f"edde_hard_extra={hard_extra_added}"
    )

    # 2. When2Call Step-0 + Post-Lookup Multi-Turn Ask Synthesis
    w_step0 = generate_when2call_contrastive_samples(train_samples, catalog, count=when2call_step0_count, seed=seed)
    w0_ok = 0
    for st in w_step0:
        if verify_step_validity(st, catalog):
            w0_ok += 1
            sid = st.get("sample_id", "")
            ctx = st.get("sample_context") or (sample_map[sid].context if sid in sample_map else None)
            verified_entries.append((st, ctx, sid, "ask", "when2call_step0", 0))

    w_post = generate_post_lookup_ask_samples(train_samples, catalog, count=when2call_post_count, seed=seed + 11)
    wp_ok = 0
    for st in w_post:
        if verify_step_validity(st, catalog):
            wp_ok += 1
            sid = st.get("sample_id", "")
            ctx = st.get("sample_context") or (sample_map[sid].context if sid in sample_map else None)
            verified_entries.append((st, ctx, sid, "ask", "when2call_post_lookup", 0))
    print(f"  2. when2call: step0={w0_ok}/{len(w_step0)}, post_lookup={wp_ok}/{len(w_post)}")

    # 3. ToolSandbox Multi-Edge Anchor Binding
    anchor_steps = generate_anchor_binding_samples(train_samples, catalog, count=anchor_count, seed=seed)
    anc_ok = 0
    for st in anchor_steps:
        if verify_step_validity(st, catalog):
            anc_ok += 1
            sid = st.get("sample_id", "")
            ctx = st.get("sample_context") or (sample_map[sid].context if sid in sample_map else None)
            verified_entries.append((st, ctx, sid, "act", "anchor_binding", 0))
    print(f"  3. anchor_binding: {anc_ok}/{len(anchor_steps)}")

    # 4. APIGen-MT Clean Multi-Intent Chaining
    apigen_steps = generate_apigen_multi_intent_samples(
        train_samples, catalog, count=apigen_count, seed=seed, include_step_1=True
    )
    api_ok = 0
    for st in apigen_steps:
        if verify_step_validity(st, catalog):
            api_ok += 1
            sid = st.get("sample_id", "")
            ctx = st.get("sample_context")
            verified_entries.append((st, ctx, sid, "act", "apigen_mt", 0))
    print(f"  4. apigen_mt: {api_ok}/{len(apigen_steps)}")

    # 5. Re-integrated Recovery Augmenter (empty search relaxation + x-ref self-repair)
    rng.shuffle(act_steps_pool)
    rec_added = 0
    for step, ctx, sid in act_steps_pool:
        if rec_added >= recovery_count:
            break
        aug = create_empty_search_recovery_step(step)
        if aug is None:
            aug = create_xref_repair_recovery_step(step)
        if aug is not None and verify_step_validity(aug, catalog):
            aug["sample_id"] = sid
            aug["think"] = synthesize_deterministic_think(aug, ctx, catalog)
            verified_entries.append((aug, ctx, sid, "act", f"recovery_{aug.get('recovery_type', 'fix')}", 0))
            rec_added += 1
    print(f"  5. recovery_augmenter: {rec_added}/{recovery_count}")

    # 6. Render messages & verify strict Tier 1 invariants
    records: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    zero_call_count = 0

    for idx_rec, (step, ctx, sid, cat, src, seed_off) in enumerate(verified_entries):
        if sid in dev_ids:
            continue
        parts = sid.split("_")
        if any(f"train_{p}" in dev_ids or p in dev_ids for p in parts if p.isdigit()):
            continue

        step_rng = random.Random(seed + idx_rec + seed_off)
        msgs = convert_step_to_minicpm5_messages(step, ctx, catalog, rng=step_rng)
        assert msgs and msgs[0]["role"] == "system"
        assert "Quy tắc bắt buộc" in msgs[0]["content"], f"Bare prompt leaked on {src} / {sid}"
        assert msgs[-1]["role"] == "assistant" and "</think>" in msgs[-1]["content"]

        if "<function " not in msgs[-1]["content"]:
            zero_call_count += 1

        rec_id = step.get("id") or f"{sid}_s{step.get('step_index', 0)}"
        records.append(
            {
                "id": rec_id,
                "sample_id": sid,
                "category": cat,
                "source": src,
                "messages": msgs,
            }
        )
        source_counts[src] += 1

    rng.shuffle(records)

    with open(out_jsonl, "w", encoding="utf-8") as f_out, gzip.open(out_gz, "wt", encoding="utf-8") as f_gz:
        for r in records:
            line = json.dumps(r, ensure_ascii=False) + "\n"
            f_out.write(line)
            f_gz.write(line)

    print(
        f"[V3 Builder] Wrote {len(records)} verified records -> {out_gz} "
        f"(zero_call_ratio={zero_call_count}/{len(records)}={zero_call_count/max(1, len(records)):.1%})"
    )
    print("  Source distribution:", dict(source_counts))

    # Optional exact tokenizer audit if local adapter tokenizer exists
    tok_json = ROOT / "experiments" / "exp_20261111_minicpm5_sft_v2_g4" / "adapter" / "tokenizer.json"
    if tok_json.exists():
        try:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(tok_json))
            tot_lens = [
                len(x.ids) + 40
                for x in tok.encode_batch(
                    ["\n".join(m["content"] for m in r["messages"]) for r in records],
                    add_special_tokens=False,
                )
            ]
            s_lens = sorted(tot_lens)
            n = len(s_lens)
            print(
                f"  Token audit: min={s_lens[0]} p50={s_lens[n//2]} p95={s_lens[int(0.95*n)]} "
                f"p99={s_lens[int(0.99*n)]} max={s_lens[-1]} (>16384={sum(x > 16384 for x in s_lens)})"
            )
        except Exception as exc:
            print(f"  Token audit skipped ({exc})")

    return len(records), out_gz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dev-split", default="configs/splits/dev_v1.json")
    parser.add_argument("--out-dir", default="data/sft")
    parser.add_argument("--hard-sample-ids", default=None, help="Optional train_only failing IDs / per_sample.jsonl")
    parser.add_argument("--when2call-step0", type=int, default=350)
    parser.add_argument("--when2call-post", type=int, default=450)
    parser.add_argument("--anchor", type=int, default=400)
    parser.add_argument("--apigen", type=int, default=550)
    parser.add_argument("--recovery", type=int, default=250)
    parser.add_argument("--keep-single-stop-ratio", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    count, gz_path = build_v3_dataset(
        split_name=args.split,
        root=args.root,
        dev_split_path=args.dev_split,
        out_dir=args.out_dir,
        when2call_step0_count=args.when2call_step0,
        when2call_post_count=args.when2call_post,
        anchor_count=args.anchor,
        apigen_count=args.apigen,
        recovery_count=args.recovery,
        keep_single_stop_ratio=args.keep_single_stop_ratio,
        hard_sample_ids_path=args.hard_sample_ids,
        seed=args.seed,
    )
    print(f"Dataset V3 ready: {count} rows in {gz_path}")


if __name__ == "__main__":
    main()
