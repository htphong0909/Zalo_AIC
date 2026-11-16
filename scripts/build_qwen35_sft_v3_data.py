"""Build clean Fold-0/Dev-excluded SFT V3 dataset for Qwen3.5-4B with 4-slot <think>, EDDE & real ASR."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import random
from pathlib import Path
from typing import Any

from scripts.build_lfm3_data import simulate_ctc_transcript
from scripts.build_minicpm5_dataset_v2 import resolve_step_tools
from src.agent.qwen35_sft import step_to_qwen35_example
from src.agent.sft_dataset import build_sft_trajectory, unroll_trajectory_steps
from src.core.dataset import load_catalog, load_split
from src.core.schemas import Sample
from src.data_gen.anchor_binding_gen import generate_anchor_binding_samples
from src.data_gen.apigen_mt_gen import generate_apigen_multi_intent_samples
from src.data_gen.edde_augmenter import (
    assert_not_excluded,
    generate_fuzzy_fidelity_edde,
    generate_long_history_coref_edde,
    generate_self_correction_and_oos_edde,
    generate_time_and_math_edde,
    mask_tools_and_params_hammer,
)
from src.data_gen.execution_verifier import verify_step_validity
from src.data_gen.prompt_utils import build_sample_prefix_messages, resolve_installed_apps
from src.data_gen.recovery_augmenter import (
    create_empty_search_recovery_step,
    create_xref_repair_recovery_step,
)
from src.data_gen.when2call_gen import (
    generate_post_lookup_ask_samples,
    generate_when2call_contrastive_samples,
)


def _load_dev_ids(path: str | Path | None) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {str(x) for x in data}
    if isinstance(data, dict):
        if "dev_ids" in data:
            return {str(x) for x in data["dev_ids"]}
        if "ids" in data:
            return {str(x) for x in data["ids"]}
    return set()


def _load_fold_splits(
    folds_file: str | Path | None,
    include_folds: list[str] | None,
) -> tuple[set[str] | None, set[str]]:
    """Return (included_ids_or_None, excluded_fold_ids)."""
    if not folds_file:
        return None, set()
    p = Path(folds_file)
    if not p.exists():
        raise FileNotFoundError(f"Folds file not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    folds_map = payload.get("folds") if isinstance(payload, dict) else None
    if not isinstance(folds_map, dict):
        raise ValueError(f"Invalid folds file schema in {p}")

    target_folds = list(include_folds or [k for k in folds_map.keys() if k != "fold_0"])
    inc_ids: set[str] = set()
    exc_ids: set[str] = set()
    for fname, id_list in folds_map.items():
        sids = {str(x) for x in (id_list or [])}
        if fname in target_folds:
            inc_ids.update(sids)
        else:
            exc_ids.update(sids)
    # Fail loudly if any sample overlaps between included and excluded folds
    overlap = inc_ids & exc_ids
    if overlap:
        raise ValueError(f"Fold overlap detected: {sorted(overlap)[:5]}")
    return inc_ids, exc_ids


def _load_asr_map(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ASR transcripts file not found: {p}")
    out: dict[str, dict[str, Any]] = {}
    if p.suffix == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
        rows = raw if isinstance(raw, list) else raw.get("rows", [])
    else:
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    for row in rows:
        sid = str(row.get("id") or row.get("sample_id") or "").strip()
        if not sid:
            continue
        raw_hist = row.get("history_texts") or {}
        hist_int: dict[int, str] = {}
        if isinstance(raw_hist, dict):
            for k, v in raw_hist.items():
                try:
                    hist_int[int(k)] = str(v)
                except (TypeError, ValueError):
                    continue
        out[sid] = {
            "audio_text": str(row.get("audio_text") or row.get("transcript") or "").strip(),
            "history_texts": hist_int,
            "asr_phonetic_diff": str(row.get("asr_phonetic_diff") or row.get("phonetic_diff") or "").strip(),
        }
    return out


def _generate_irrelevance_oos(
    train_samples: list[Sample],
    catalog: dict[str, dict[str, Any]],
    *,
    count: int = 250,
    seed: int = 42,
    exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build OOS irrelevance examples by hiding used apps behind installed distractors."""
    rng = random.Random(seed)
    results: list[dict[str, Any]] = []
    act_samples = [
        s for s in train_samples
        if s.label and getattr(s.label, "tool_calls", None) and str(s.id) not in (exclude_ids or set())
    ]
    rng.shuffle(act_samples)

    for s in act_samples:
        if len(results) >= count:
            break
        assert_not_excluded(str(s.id), exclude_ids)
        calls = getattr(s.label, "tool_calls", None) or []
        used_apps = {c.name.split(".", 1)[0] for c in calls if "." in c.name}
        ctx = copy.deepcopy(s.context or {})
        inst = ctx.get("installed_apps")
        if not isinstance(inst, dict) or not used_apps:
            continue
        reduced = {k: v for k, v in inst.items() if k not in used_apps}
        if not reduced:
            reduced = {"weather": "weather", "calculator": "calculator"}
        ctx["installed_apps"] = reduced
        mod = Sample(id=str(s.id), audio_text=s.audio_text, history=list(s.history), context=ctx)
        prompt_msgs = build_sample_prefix_messages(mod, catalog)
        st = {
            "id": f"irr_oos_{s.id}",
            "sample_id": str(s.id),
            "step_index": 0,
            "step_category": "oos",
            "prompt_messages": prompt_msgs,
            "step_calls": [],
            "tools": list(reduced.keys()),
            "sample_context": ctx,
        }
        if verify_step_validity(st, catalog):
            results.append(st)
    return results


def _attach_asr_diff_to_latest_user(
    prompt_messages: list[dict[str, Any]],
    asr_diff: str,
) -> list[dict[str, Any]]:
    if not asr_diff:
        return prompt_messages
    out = [dict(m) for m in prompt_messages]
    for idx in range(len(out) - 1, -1, -1):
        if out[idx].get("role") == "user":
            content = str(out[idx].get("content") or "")
            if "[Gợi ý đồng âm ASR:" not in content:
                out[idx]["content"] = f"{content}\n{asr_diff}".strip()
            break
    return out


def _collect_recovery_steps(
    base_steps: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pool = [st for st in base_steps if st.get("step_category") == "act" and st.get("step_calls")]
    rng.shuffle(pool)
    out: list[dict[str, Any]] = []
    for st in pool:
        if len(out) >= count:
            break
        rec_empty = create_empty_search_recovery_step(st)
        if rec_empty is not None and verify_step_validity(rec_empty, catalog):
            out.append(rec_empty)
            if len(out) >= count:
                break
        rec_xref = create_xref_repair_recovery_step(
            st, require_grounded_context=True, catalog=catalog
        )
        if rec_xref is not None and verify_step_validity(rec_xref, catalog):
            out.append(rec_xref)
    return out[:count]


def _extract_constituent_sample_ids(step: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    sid = str(step.get("sample_id") or "").strip()
    if sid:
        ids.append(sid)
    meta = step.get("meta")
    if isinstance(meta, dict):
        for k in ("sample_id", "s1_id", "s2_id", "pair_id"):
            val = str(meta.get(k) or "").strip()
            if val:
                ids.append(val)
    return ids


def build_qwen35_sft_v3_dataset(
    split_name: str = "train",
    root: str | Path = "data",
    folds_file: str | Path | None = "configs/splits/train_only_5folds.json",
    include_folds: list[str] | None = None,
    exclude_ids_path: str | Path = "configs/splits/dev_v1.json",
    out_dir: str | Path = "data/sft",
    out_stem: str = "qwen35_train_sft_v3_clean",
    *,
    asr_transcripts_path: str | Path | None = None,
    asr_ratio: float = 0.40,
    simulated_asr_ratio: float = 0.12,
    enable_thinking: bool = True,
    max_base_samples: int | None = None,
    edde_time_math_count: int = 350,
    edde_long_history_count: int = 0,
    edde_fuzzy_count: int = 300,
    edde_sc_oos_count: int = 50,
    when2call_step0_count: int = 250,
    when2call_post_count: int = 350,
    anchor_count: int = 400,
    apigen_count: int = 450,
    recovery_count: int = 250,
    irrelevance_oos_count: int = 140,
    hard_replay_count: int = 250,
    keep_single_stop_ratio: float = 0.25,
    hammer_mask_prob: float = 0.10,
    seed: int = 42,
) -> tuple[int, Path]:
    """Assemble and serialize leak-free SFT V3 dataset to compressed JSONL."""
    rng = random.Random(seed)
    dev_ids = _load_dev_ids(exclude_ids_path)
    inc_ids, exc_fold_ids = _load_fold_splits(folds_file, include_folds)
    all_exclude_ids = dev_ids | exc_fold_ids

    catalog = load_catalog(split_name, root)
    all_samples = load_split(split_name, root)

    clean_samples: list[Sample] = []
    for s in all_samples:
        sid = str(s.id)
        if sid in all_exclude_ids:
            continue
        if inc_ids is not None and sid not in inc_ids:
            continue
        assert_not_excluded(sid, all_exclude_ids)
        clean_samples.append(s)

    if max_base_samples is not None and max_base_samples > 0:
        clean_samples = clean_samples[:max_base_samples]

    sample_ctx_map: dict[str, dict[str, Any]] = {
        str(s.id): (s.context or {}) for s in clean_samples
    }
    asr_map = _load_asr_map(asr_transcripts_path)
    # Remove any accidental excluded IDs from asr_map before use
    asr_map = {k: v for k, v in asr_map.items() if k not in all_exclude_ids}

    # 1. Base trajectories (mixing real ASR + clean transcripts)
    base_steps: list[dict[str, Any]] = []
    stop_steps_seen = 0
    for s in clean_samples:
        sid = str(s.id)
        use_asr = sid in asr_map and rng.random() < asr_ratio
        asr_entry = asr_map.get(sid)
        cur_txt = (asr_entry["audio_text"] or None) if use_asr and asr_entry else None
        hist_txt = (asr_entry["history_texts"] or None) if use_asr and asr_entry else None
        traj = build_sft_trajectory(s, catalog, current_text=cur_txt, history_texts=hist_txt)
        steps = unroll_trajectory_steps(traj, split=None, root=None)
        for st in steps:
            st["sample_id"] = sid
            st["sample_context"] = s.context or {}
            st["has_history"] = bool(s.history)
            if not verify_step_validity(st, catalog):
                continue
            if st.get("step_category") == "act" and not st.get("step_calls"):
                stop_steps_seen += 1
                if rng.random() > keep_single_stop_ratio:
                    continue
            if use_asr:
                if asr_entry and asr_entry.get("asr_phonetic_diff"):
                    st["prompt_messages"] = _attach_asr_diff_to_latest_user(
                        st["prompt_messages"], asr_entry["asr_phonetic_diff"]
                    )
                st["asr_applied"] = "real"
            elif (
                not use_asr
                and st.get("step_index", 0) == 0
                and simulated_asr_ratio > 0
                and rng.random() < simulated_asr_ratio
                and st.get("prompt_messages")
                and st["prompt_messages"][-1].get("role") == "user"
            ):
                orig_u = str(st["prompt_messages"][-1].get("content") or "")
                st["prompt_messages"][-1]["content"] = simulate_ctc_transcript(orig_u)
                st["asr_applied"] = "sim"
            base_steps.append(st)

    # 2. EDDE & contrastive slices (strictly over clean_samples + all_exclude_ids guard)
    edde_tm = generate_time_and_math_edde(
        clean_samples, catalog, count=edde_time_math_count, seed=seed + 1, exclude_ids=all_exclude_ids
    )
    edde_lh = generate_long_history_coref_edde(
        clean_samples, catalog, count=edde_long_history_count, seed=seed + 2, exclude_ids=all_exclude_ids
    )
    edde_fz = generate_fuzzy_fidelity_edde(
        clean_samples, catalog, count=edde_fuzzy_count, seed=seed + 3, exclude_ids=all_exclude_ids
    )
    edde_sc = generate_self_correction_and_oos_edde(
        clean_samples, catalog, count=edde_sc_oos_count, seed=seed + 4, exclude_ids=all_exclude_ids
    )
    w2c_step0 = generate_when2call_contrastive_samples(
        clean_samples, catalog, count=when2call_step0_count, seed=seed + 5
    )
    w2c_post = generate_post_lookup_ask_samples(
        clean_samples, catalog, count=when2call_post_count, seed=seed + 6
    )
    anchor_steps = generate_anchor_binding_samples(
        clean_samples, catalog, count=anchor_count, seed=seed + 7
    )
    apigen_steps = generate_apigen_multi_intent_samples(
        clean_samples, catalog, count=apigen_count, seed=seed + 8, include_step_1=True
    )
    recovery_steps = _collect_recovery_steps(
        base_steps, catalog, count=recovery_count, seed=seed + 9
    )
    irr_oos_steps = _generate_irrelevance_oos(
        clean_samples, catalog, count=irrelevance_oos_count, seed=seed + 10, exclude_ids=all_exclude_ids
    )

    hard_candidates = [
        copy.deepcopy(st)
        for st in base_steps
        if st.get("step_index", 0) >= 1 or st.get("has_history") or st.get("step_category") in ("ask", "oos")
    ]
    rng.shuffle(hard_candidates)
    hard_replay = hard_candidates[:hard_replay_count]
    for idx, hr in enumerate(hard_replay):
        hr["id"] = f"{hr.get('id', 'hr')}_replay_{idx}"

    combined_sources: list[tuple[str, list[dict[str, Any]]]] = [
        ("base", base_steps),
        ("edde_time_math", edde_tm),
        ("edde_long_history", edde_lh),
        ("edde_fuzzy", edde_fz),
        ("edde_sc_oos", edde_sc),
        ("when2call_step0", w2c_step0),
        ("when2call_post", w2c_post),
        ("anchor_binding", anchor_steps),
        ("apigen_mt", apigen_steps),
        ("recovery", recovery_steps),
        ("irrelevance_oos", irr_oos_steps),
        ("hard_replay", hard_replay),
    ]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    gz_path = out_path / f"{out_stem}.jsonl.gz"
    stats_path = out_path / f"{out_stem}_stats.json"

    all_rows: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    cat_counts: dict[str, int] = {}
    asr_real_hits = 0
    hammer_masked_count = 0

    for src_name, step_list in combined_sources:
        accepted = 0
        for st in step_list:
            for sid in _extract_constituent_sample_ids(st):
                assert_not_excluded(sid, all_exclude_ids)

            # Skip upstream items lacking valid unmasked structure
            if not verify_step_validity(st, catalog):
                continue

            step_copy = copy.deepcopy(st)
            # Clear upstream pre-baked think so V3 4-slot <think> synthesizes AFTER ASR/masking
            step_copy.pop("think", None)

            ctx = step_copy.get("sample_context") or sample_ctx_map.get(str(step_copy.get("sample_id") or "")) or {}
            tools = resolve_step_tools(step_copy, ctx, catalog)

            ex = step_to_qwen35_example(
                step_copy,
                ctx,
                tools,
                catalog,
                enable_thinking=enable_thinking,
                rng=rng,
            )
            sid_out = str(step_copy.get("sample_id") or "")
            assert_not_excluded(sid_out, all_exclude_ids)

            if step_copy.get("asr_applied") == "real":
                asr_real_hits += 1

            row = {
                "id": str(step_copy.get("id") or f"{src_name}_{len(all_rows)}"),
                "sample_id": sid_out,
                "source": src_name,
                "step_category": str(step_copy.get("step_category") or "act"),
                "prompt_messages": ex["prompt_messages"],
                "tools": ex["tools"],
                "completion": ex["completion"],
                "enable_thinking": ex["enable_thinking"],
            }
            all_rows.append(row)
            accepted += 1
            cat_counts[row["step_category"]] = cat_counts.get(row["step_category"], 0) + 1

            # Additive Hammer schema masking ONLY on real base act steps (never overwrite unmasked rows)
            if (
                hammer_mask_prob > 0
                and src_name == "base"
                and step_copy.get("step_category") == "act"
                and step_copy.get("step_calls")
                and rng.random() < hammer_mask_prob
            ):
                step_hm = copy.deepcopy(st)
                step_hm.pop("think", None)
                step_hm["tools"] = tools
                step_hm = mask_tools_and_params_hammer(step_hm, rng=rng)
                tools_hm = step_hm.get("tools") or tools
                local_catalog_hm = {
                    t["function"]["name"]: t["function"]
                    for t in tools_hm
                    if isinstance(t, dict) and isinstance(t.get("function"), dict)
                }
                ex_hm = step_to_qwen35_example(
                    step_hm,
                    ctx,
                    tools_hm,
                    local_catalog_hm,
                    enable_thinking=enable_thinking,
                    rng=rng,
                )
                row_hm = {
                    "id": f"{row['id']}_hm",
                    "sample_id": sid_out,
                    "source": "hammer_masked",
                    "step_category": str(step_hm.get("step_category") or row["step_category"]),
                    "prompt_messages": ex_hm["prompt_messages"],
                    "tools": ex_hm["tools"],
                    "completion": ex_hm["completion"],
                    "enable_thinking": ex_hm["enable_thinking"],
                }
                all_rows.append(row_hm)
                hammer_masked_count += 1
                cat_counts[row_hm["step_category"]] = cat_counts.get(row_hm["step_category"], 0) + 1
        source_counts[src_name] = accepted
    if hammer_masked_count:
        source_counts["hammer_masked"] = hammer_masked_count

    rng.shuffle(all_rows)
    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        for row in all_rows:
            assert_not_excluded(row["sample_id"], all_exclude_ids)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "total_rows": len(all_rows),
        "base_samples": len(clean_samples),
        "excluded_ids_count": len(all_exclude_ids),
        "leaked_ids": 0,
        "enable_thinking": enable_thinking,
        "asr_real_hits": asr_real_hits,
        "hammer_masked_count": hammer_masked_count,
        "source_counts": source_counts,
        "category_counts": cat_counts,
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(all_rows), gz_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leak-free SFT V3/V3.1 dataset (Folds 1..4)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--root", default="data")
    parser.add_argument("--folds-file", default="configs/splits/train_only_5folds.json")
    parser.add_argument("--include-folds", nargs="+", default=["fold_1", "fold_2", "fold_3", "fold_4"])
    parser.add_argument("--exclude-ids", default="configs/splits/dev_v1.json")
    parser.add_argument("--out-dir", default="data/sft")
    parser.add_argument("--out-stem", default="qwen35_train_sft_v3_1_clean")
    parser.add_argument("--asr-transcripts", default=None)
    parser.add_argument("--asr-ratio", type=float, default=0.40)
    parser.add_argument("--enable-thinking", action="store_true", default=True)
    parser.add_argument("--no-enable-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--hammer-mask-prob", type=float, default=0.10)
    parser.add_argument("--edde-long-history-count", type=int, default=0)
    parser.add_argument("--edde-sc-oos-count", type=int, default=50)
    parser.add_argument("--when2call-post-count", type=int, default=350)
    parser.add_argument("--irrelevance-oos-count", type=int, default=140)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total, gz_path = build_qwen35_sft_v3_dataset(
        split_name=args.split,
        root=args.root,
        folds_file=args.folds_file,
        include_folds=args.include_folds,
        exclude_ids_path=args.exclude_ids,
        out_dir=args.out_dir,
        out_stem=args.out_stem,
        asr_transcripts_path=args.asr_transcripts,
        asr_ratio=args.asr_ratio,
        enable_thinking=args.enable_thinking,
        edde_long_history_count=args.edde_long_history_count,
        edde_sc_oos_count=args.edde_sc_oos_count,
        when2call_post_count=args.when2call_post_count,
        irrelevance_oos_count=args.irrelevance_oos_count,
        hammer_mask_prob=args.hammer_mask_prob,
        seed=args.seed,
    )
    print(f"Built SFT dataset: {total} rows -> {gz_path}")


if __name__ == "__main__":
    main()
