"""Regression tests for SFT V3.1 data-quality and latency fixes."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from src.core.dataset import load_catalog, load_split
from src.data_gen.distill_think import (
    _extract_observation_facts,
    synthesize_deterministic_think,
)
from src.data_gen.edde_augmenter import generate_time_and_math_edde
from src.data_gen.when2call_gen import (
    drop_required_param_from_utterance,
    generate_post_lookup_ask_samples,
    generate_when2call_contrastive_samples,
)


def test_extract_observation_facts_covers_gold_bills_and_devices():
    messages = [
        {
            "role": "tool",
            "name": "gold_price.get_price",
            "content": json.dumps({"prices": [{"type": "sjc", "buy": 83000000, "sell": 85500000}]}),
        },
        {
            "role": "tool",
            "name": "bill_payment.list_bills",
            "content": json.dumps(
                {
                    "bills": [
                        {"id": "bill_01", "type": "electric", "amount": 920000, "due_date": "2026-11-20"},
                        {"id": "bill_02", "type": "water", "amount": 180000, "due_date": "2026-11-25"},
                    ]
                }
            ),
        },
        {
            "role": "tool",
            "name": "smart_home.get_status",
            "content": json.dumps(
                {
                    "devices": [
                        {"id": "light_bedroom", "floor": 2, "state": {"power": "off"}},
                        {"id": "ac_bedroom", "floor": 2, "state": {"mode": "cool", "temperature": 25}},
                    ]
                }
            ),
        },
    ]
    facts = _extract_observation_facts(messages)
    assert "83000000" in facts
    assert "85500000" in facts
    assert "bill_01" in facts and "920000" in facts
    assert "bill_02" in facts and "180000" in facts
    assert "light_bedroom" in facts


def test_think_ignores_false_cancel_markers_on_act_and_stays_concise():
    step = {
        "prompt_messages": [
            {"role": "user", "content": "Mai lùi lại thêm 1 tiếng nhé, đừng đặt sớm quá, khỏi gọi anh Minh, à thôi nhắc mang nước."}
        ],
        "step_calls": [{"name": "calendar.list_events", "arguments": {}}],
        "step_category": "act",
        "step_index": 0,
    }
    think = synthesize_deterministic_think(
        step,
        sample_context={"current_time": "2026-11-15T09:00:00+07:00"},
        catalog={"calendar.list_events": {"name": "calendar.list_events", "parameters": {"properties": {}}}},
    )
    assert "tự sửa ý" not in think
    assert "hủy lệnh" not in think
    assert len(think) < 350


def test_edde_time_math_includes_positive_late_shifts():
    catalog = load_catalog("train", "data")
    samples = load_split("train", "data")[:30]
    steps = generate_time_and_math_edde(samples, catalog, count=80, seed=42)
    assert len(steps) > 0

    pos_steps = [
        st
        for st in steps
        if any(kw in str(st["prompt_messages"][-1]["content"]) for kw in ("lùi lại", "muộn thêm", "muộn hơn"))
    ]
    assert len(pos_steps) >= 15

    tools_covered = {
        call["name"]
        for st in pos_steps
        for call in st.get("step_calls", [])
    }
    assert "calendar.update_event" in tools_covered or "alarm.create" in tools_covered or "reminders.create" in tools_covered


def test_when2call_preserves_phone_digits_and_skips_content_and_conditionals():
    masked = drop_required_param_from_utterance(
        "Chuyển 500k cho số 0912345678 nhé",
        "amount",
        500000,
    )
    assert "0912345678" in masked
    assert "chút tiền chút tiền" not in masked

    catalog = load_catalog("train", "data")
    f5 = json.loads(Path("configs/splits/train_only_5folds.json").read_text(encoding="utf-8"))
    fold0 = set(f5["folds"]["fold_0"])
    clean_samples = [s for s in load_split("train", "data") if s.id not in fold0][:120]

    step0 = generate_when2call_contrastive_samples(clean_samples, catalog, count=120, seed=42)
    for st in step0:
        for c in st.get("step_calls", []):
            if c.get("name") == "sys.ask_missing_info":
                params = c.get("arguments", {}).get("params", [])
                assert "content" not in params
                assert "chút tiền chút tiền" not in str(st["prompt_messages"][-1]["content"])

    post = generate_post_lookup_ask_samples(clean_samples, catalog, count=120, seed=42)
    for st in post:
        latest_u = next((m["content"] for m in reversed(st["prompt_messages"]) if m.get("role") == "user"), "")
        if st["id"].startswith(("when2call_ambig_", "when2call_mi_")):
            assert not latest_u.lower().startswith("nếu ")
        for c in st.get("step_calls", []):
            if c.get("name") == "sys.ask_missing_info":
                assert "content" not in c.get("arguments", {}).get("params", [])


def test_build_v3_dataset_hammer_is_strictly_additive(tmp_path):
    from scripts.build_qwen35_sft_v3_data import build_qwen35_sft_v3_dataset

    total, gz_path = build_qwen35_sft_v3_dataset(
        split_name="train",
        root="data",
        folds_file="configs/splits/train_only_5folds.json",
        include_folds=["fold_1", "fold_2", "fold_3", "fold_4"],
        exclude_ids_path="configs/splits/dev_v1.json",
        out_dir=tmp_path,
        out_stem="additive_hammer",
        asr_transcripts_path=None,
        asr_ratio=0.0,
        simulated_asr_ratio=0.0,
        enable_thinking=True,
        max_base_samples=25,
        edde_time_math_count=0,
        edde_long_history_count=0,
        edde_fuzzy_count=0,
        edde_sc_oos_count=0,
        when2call_step0_count=6,
        when2call_post_count=0,
        anchor_count=0,
        apigen_count=0,
        recovery_count=0,
        irrelevance_oos_count=0,
        hard_replay_count=0,
        hammer_mask_prob=0.5,
        seed=42,
    )
    rows = [json.loads(line) for line in gzip.open(gz_path, "rt", encoding="utf-8") if line.strip()]
    assert len(rows) == total

    base_ids = set()
    hm_ids = []
    for r in rows:
        if r["source"] in ("base", "when2call_step0"):
            assert '"fn_' not in r["completion"]
            assert '"arg_' not in r["completion"]
        if r["source"] == "base":
            base_ids.add(r["id"])
        elif r["source"] == "hammer_masked":
            assert r["id"].endswith("_hm")
            hm_ids.append(r["id"].removesuffix("_hm"))

    assert len(hm_ids) >= 1
    assert set(hm_ids).issubset(base_ids)
