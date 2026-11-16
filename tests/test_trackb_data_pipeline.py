from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Any

import pytest

from src.agent.qwen35_sft import format_qwen35_tool_calls_xml, step_to_qwen35_example
from src.agent.tool_caller import parse_tool_calls
from src.core.dataset import load_catalog, load_split
from src.core.schemas import Sample, ToolCall
from src.data_gen.distill_think import synthesize_deterministic_think
from src.data_gen.execution_verifier import verify_step_validity


def test_edde_augmenters_enforce_exclude_ids_and_validity():
    from src.data_gen.edde_augmenter import (
        assert_not_excluded,
        generate_fuzzy_fidelity_edde,
        generate_long_history_coref_edde,
        generate_self_correction_and_oos_edde,
        generate_time_and_math_edde,
    )

    with pytest.raises(ValueError, match="leak"):
        assert_not_excluded("train_0001", {"train_0001"})

    catalog = load_catalog("train", "data")
    samples = load_split("train", "data")[:180]
    excluded = {s.id for s in samples[:30]}

    generators = [
        ("time_math", generate_time_and_math_edde(samples, catalog, count=8, seed=42, exclude_ids=excluded)),
        ("long_hist", generate_long_history_coref_edde(samples, catalog, count=8, seed=42, exclude_ids=excluded)),
        ("fuzzy", generate_fuzzy_fidelity_edde(samples, catalog, count=8, seed=42, exclude_ids=excluded)),
        ("sc_oos", generate_self_correction_and_oos_edde(samples, catalog, count=8, seed=42, exclude_ids=excluded)),
    ]
    for tag, rows in generators:
        assert len(rows) >= 2, f"{tag} generated too few rows ({len(rows)})"
        for st in rows:
            sid = str(st.get("sample_id") or "")
            assert sid not in excluded, f"Excluded sample {sid} leaked in {tag}"
            assert verify_step_validity(st, catalog), f"Invalid step in {tag}: {st.get('id')}"
            if st.get("step_calls"):
                xml = format_qwen35_tool_calls_xml(st["step_calls"])
                parsed = parse_tool_calls(xml, catalog)
                assert [c.name for c in parsed] == [c["name"] for c in st["step_calls"]]


def test_distill_think_v3_4slots_math_history_fuzzy_and_asr():
    step = {
        "id": "mock_v3_think",
        "step_category": "act",
        "math_note": "20:00 - 25 phút = 19:35; Thứ Bảy + 2 ngày = Thứ Hai (2026-11-16)",
        "history_cue": "[Lịch sử lượt 1: sự kiện ev_02 ở 18 Quốc Hương, Thảo Điền]",
        "mandatory_keywords": ["tx_01", "18 Quốc Hương, Thảo Điền"],
        "prompt_messages": [
            {
                "role": "user",
                "content": 'Nhắc uống thuốc qua tử lúc 19:35 thứ Hai\n[Gợi ý đồng âm ASR: "qua tử" <-> "bao tử"]',
            }
        ],
        "step_calls": [
            {
                "name": "reminders.create",
                "arguments": {"title": "Uống thuốc bao tử (tx_01) ở 18 Quốc Hương, Thảo Điền", "remind_at": "2026-11-16T19:35:00+07:00"},
            }
        ],
    }
    think = synthesize_deterministic_think(
        step,
        sample_context={"current_time": "2026-11-14T10:00:00+07:00"},
    )
    assert think.startswith("<think>") and think.endswith("</think>")
    for slot in ("1. Ý định:", "2. Dữ liệu:", "3. Kế hoạch:", "4. Ranh giới:"):
        assert slot in think
    assert "Gợi ý đồng âm ASR" in think
    assert "19:35" in think
    assert "[Lịch sử lượt 1:" in think
    assert "Từ khóa bắt buộc:" in think
    assert "tx_01" in think
    assert "Đủ điều kiện thực thi" in think


def test_mask_tools_and_params_hammer_masks_names_params_and_defaults():
    from src.data_gen.edde_augmenter import mask_tools_and_params_hammer

    step = {
        "id": "mock_hammer_v3",
        "step_category": "act",
        "prompt_messages": [
            {"role": "system", "content": "Bạn là trợ lý...\nsmart_home.ac_set"},
            {"role": "user", "content": "Đặt điều hòa chế độ eco giúp tôi"},
        ],
        "step_calls": [{"name": "smart_home.ac_set", "arguments": {"device_id": "ac_1", "mode": "eco"}}],
        "prefix_calls": [],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "smart_home.ac_set",
                    "description": "Điều khiển điều hòa",
                    "parameters": {
                        "type": "object",
                        "required": ["device_id"],
                        "properties": {
                            "device_id": {"type": "string", "x-ref": "smart_home.devices"},
                            "mode": {"type": "string", "default": "eco", "description": "Chế độ mặc định eco"},
                        },
                    },
                },
            }
        ],
    }
    masked = mask_tools_and_params_hammer(step, rng=random.Random(42))
    m_tool = masked["tools"][0]["function"]
    assert m_tool["name"] != "smart_home.ac_set"
    m_props = m_tool["parameters"]["properties"]
    assert "mode" not in m_props and "device_id" not in m_props
    # Default value must also have been masked consistently across tool schema and step call args
    masked_call = masked["step_calls"][0]
    assert masked_call["name"] == m_tool["name"]
    mode_specs = [v for v in m_props.values() if "default" in v]
    assert mode_specs and mode_specs[0]["default"] != "eco"
    new_default = mode_specs[0]["default"]
    assert new_default in masked_call["arguments"].values()
    assert new_default in masked["prompt_messages"][-1]["content"]


def test_build_qwen35_sft_v3_dataset_excludes_fold0_and_dev_and_applies_real_asr(tmp_path):
    from scripts.build_qwen35_sft_v3_data import build_qwen35_sft_v3_dataset

    f5 = json.loads(Path("configs/splits/train_only_5folds.json").read_text(encoding="utf-8"))
    fold0_ids = set(f5["folds"]["fold_0"])
    dev_ids = set(json.loads(Path("configs/splits/dev_v1.json").read_text(encoding="utf-8"))["dev_ids"])

    # Create mock real Loop #38 PhoWhisper transcripts ONLY for folds 1..4 samples
    samples = [s for s in load_split("train", "data") if s.id not in fold0_ids and s.id not in dev_ids][:40]
    asr_path = tmp_path / "mock_asr_nbest.jsonl"
    with open(asr_path, "w", encoding="utf-8") as f:
        for s in samples:
            row = {
                "id": s.id,
                "audio_text": (s.audio_text or "bật đèn") + " nhé",
                "history_texts": {str(i): (t.audio_text or "") + " à" for i, t in enumerate(s.history) if t.role == "user"},
                "asr_phonetic_diff": '[Gợi ý đồng âm ASR: "bảo" <-> "báo"]',
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    out_stem = "test_v3_clean"
    total, gz_path = build_qwen35_sft_v3_dataset(
        split_name="train",
        root="data",
        folds_file="configs/splits/train_only_5folds.json",
        include_folds=["fold_1", "fold_2", "fold_3", "fold_4"],
        exclude_ids_path="configs/splits/dev_v1.json",
        out_dir=tmp_path,
        out_stem=out_stem,
        asr_transcripts_path=asr_path,
        asr_ratio=0.40,
        enable_thinking=True,
        max_base_samples=40,
        edde_time_math_count=6,
        edde_long_history_count=6,
        edde_fuzzy_count=6,
        edde_sc_oos_count=6,
        when2call_step0_count=4,
        when2call_post_count=4,
        anchor_count=4,
        apigen_count=4,
        recovery_count=4,
        irrelevance_oos_count=4,
        hard_replay_count=4,
        hammer_mask_prob=0.25,
        seed=42,
    )
    assert total >= 30
    assert gz_path.exists()

    rows = [json.loads(line) for line in gzip.open(gz_path, "rt", encoding="utf-8") if line.strip()]
    assert len(rows) == total
    catalog = load_catalog("train", "data")
    asr_diff_hits = 0
    for r in rows:
        assert r["sample_id"] not in fold0_ids
        assert r["sample_id"] not in dev_ids
        assert r["enable_thinking"] is True
        comp = r["completion"]
        assert comp.endswith("<|im_end|>")
        assert "</think>\n\n" in comp
        for slot in ("1. Ý định:", "2. Dữ liệu:", "3. Kế hoạch:", "4. Ranh giới:"):
            assert slot in comp
        body = comp.split("</think>\n\n", 1)[1].removesuffix("<|im_end|>")
        if "<tool_call>" in body:
            assert parse_tool_calls(body, catalog)
        if "[Gợi ý đồng âm ASR:" in str(r["prompt_messages"][-1]["content"]):
            asr_diff_hits += 1
    assert asr_diff_hits >= 1
    stats = json.loads((tmp_path / f"{out_stem}_stats.json").read_text(encoding="utf-8"))
    assert stats["leaked_ids"] == 0
    assert stats["asr_real_hits"] >= 1


def test_build_qwen35_sft_v3_tracks_real_asr_without_phonetic_diff(tmp_path):
    from scripts.build_qwen35_sft_v3_data import build_qwen35_sft_v3_dataset

    f5 = json.loads(Path("configs/splits/train_only_5folds.json").read_text(encoding="utf-8"))
    fold0_ids = set(f5["folds"]["fold_0"])
    dev_ids = set(json.loads(Path("configs/splits/dev_v1.json").read_text(encoding="utf-8"))["dev_ids"])
    samples = [s for s in load_split("train", "data") if s.id not in fold0_ids and s.id not in dev_ids][:15]
    asr_path = tmp_path / "greedy_asr.jsonl"
    with open(asr_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({"id": s.id, "audio_text": s.audio_text, "history_texts": {}}, ensure_ascii=False) + "\n")

    build_qwen35_sft_v3_dataset(
        split_name="train",
        root="data",
        folds_file="configs/splits/train_only_5folds.json",
        include_folds=["fold_1", "fold_2", "fold_3", "fold_4"],
        exclude_ids_path="configs/splits/dev_v1.json",
        out_dir=tmp_path,
        out_stem="greedy_check",
        asr_transcripts_path=asr_path,
        asr_ratio=1.0,
        enable_thinking=True,
        max_base_samples=15,
        edde_time_math_count=0,
        edde_long_history_count=0,
        edde_fuzzy_count=0,
        edde_sc_oos_count=0,
        when2call_step0_count=0,
        when2call_post_count=0,
        anchor_count=0,
        apigen_count=0,
        recovery_count=0,
        irrelevance_oos_count=0,
        hard_replay_count=0,
        seed=42,
    )
    stats = json.loads((tmp_path / "greedy_check_stats.json").read_text(encoding="utf-8"))
    assert stats["asr_real_hits"] >= 10
    assert stats["leaked_ids"] == 0
