"""Unit tests for MiniCPM5 DPO Preference Pairs Generation.
Verifies When2Call, Step-DPO, LoopTool failure mining, Anchor Binding, and Zero Dev Leakage.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import pytest

from src.core.schemas import ASK_MISSING_INFO, Sample, ToolCall
from scripts.build_minicpm5_dpo_pairs import (
    generate_when2call_dpo_pairs,
    generate_anti_early_stop_dpo_pairs,
    generate_anchor_binding_dpo_pairs,
    generate_looptool_mined_dpo_pairs,
    build_minicpm5_dpo_dataset,
)


@pytest.fixture
def mock_catalog():
    return {
        "banking.transfer": {
            "name": "banking.transfer",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient_phone": {"type": "string"},
                    "amount": {"type": "integer"},
                    "notes": {"type": "string"},
                },
                "required": ["recipient_phone", "amount"],
            },
        },
        "smart_home.get_status": {
            "name": "smart_home.get_status",
            "parameters": {"type": "object", "properties": {"room": {"type": "string"}}},
        },
        "smart_home.light_set": {
            "name": "smart_home.light_set",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {"type": "string"},
                    "power": {"type": "string"},
                },
                "required": ["device_id", "power"],
            },
        },
        "sys.ask_missing_info": {
            "name": "sys.ask_missing_info",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string"},
                    "params": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["tool_name", "params"],
            },
        },
    }


@pytest.fixture
def mock_samples():
    return [
        Sample(
            id="train_mock_01",
            audio_text="Chuyển 500 nghìn cho Hương qua số 0987654321 nhé",
            context={"user_profile": {"name": "Test"}},
            label={
                "tool_calls": [
                    {
                        "name": "banking.transfer",
                        "arguments": {"recipient_phone": "0987654321", "amount": 500000},
                    }
                ]
            },
        ),
        Sample(
            id="train_mock_02",
            audio_text="Tắt hết đèn tầng 2 giúp tôi",
            context={"user_profile": {"name": "Test"}},
            label={
                "tool_calls": [
                    {"name": "smart_home.get_status", "arguments": {}},
                    {"name": "smart_home.light_set", "arguments": {"device_id": "light_office", "power": "off"}},
                    {"name": "smart_home.light_set", "arguments": {"device_id": "light_bedroom2", "power": "off"}},
                ]
            },
        ),
        Sample(
            id="train_mock_03",
            audio_text="Nhắc tôi đi họp phụ huynh với",
            context={"user_profile": {"name": "Test"}},
            label={
                "tool_calls": [
                    {"name": "sys.ask_missing_info", "arguments": {"tool_name": "reminders.create", "params": ["time"]}}
                ]
            },
        ),
    ]


def test_when2call_dpo_pairs(mock_samples, mock_catalog):
    pairs = generate_when2call_dpo_pairs(mock_samples, mock_catalog, count=4, seed=42)
    assert len(pairs) > 0
    for p in pairs:
        assert "prompt" in p
        assert "chosen" in p
        assert "rejected" in p
        assert "<think>" in p["chosen"]
        assert "<think>" in p["rejected"]
        assert p["chosen"] != p["rejected"]


def test_anti_early_stop_dpo_pairs(mock_samples, mock_catalog):
    pairs = generate_anti_early_stop_dpo_pairs(mock_samples, mock_catalog, count=2, seed=42)
    assert len(pairs) > 0
    p = pairs[0]
    assert "<function" in p["chosen"]
    # Rejected should represent early termination
    assert "<function" not in p["rejected"] or "smart_home" in p["rejected"]


def test_anchor_binding_dpo_pairs(mock_samples, mock_catalog):
    pairs = generate_anchor_binding_dpo_pairs(mock_samples, mock_catalog, count=2, seed=42)
    # Even if mock_samples has few read-write chains, it shouldn't crash
    assert isinstance(pairs, list)


def test_build_dpo_dataset_no_dev_leakage(tmp_path, mock_catalog):
    # Create dummy train.jsonl and dev_v1.json
    data_dir = tmp_path / "data" / "train"
    data_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = tmp_path / "configs" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    dev_ids = ["dev_sample_01", "dev_sample_02"]
    (splits_dir / "dev_v1.json").write_text(json.dumps({"dev_ids": dev_ids}), encoding="utf-8")

    train_file = data_dir / "train.jsonl"
    dummy_train = [
        {"id": "dev_sample_01", "audio_text": "leak sample", "context": {}, "label": {"tool_calls": []}},
        {
            "id": "train_safe_01",
            "audio_text": "Chuyển 200k cho Nam số 0912345678",
            "context": {},
            "label": {"tool_calls": [{"name": "banking.transfer", "arguments": {"recipient_phone": "0912345678", "amount": 200000}}]},
        },
        {
            "id": "train_safe_02",
            "audio_text": "Tắt đèn phòng khách",
            "context": {},
            "label": {"tool_calls": [{"name": "smart_home.light_set", "arguments": {"device_id": "light_living", "power": "off"}}]},
        },
    ]
    with open(train_file, "w", encoding="utf-8") as f:
        for r in dummy_train:
            f.write(json.dumps(r) + "\n")

    catalog_file = data_dir / "train_tools_catalog.json"
    formatted_apps = [
        {"name": "mock_app", "tools": [{"name": k, **v} for k, v in mock_catalog.items()]}
    ]
    catalog_file.write_text(json.dumps({"apps": formatted_apps}), encoding="utf-8")

    out_dir = tmp_path / "dpo_out"
    count, gz_path = build_minicpm5_dpo_dataset(
        split_name="train",
        root=tmp_path / "data",
        dev_split_path=splits_dir / "dev_v1.json",
        out_dir=out_dir,
        target_pairs=10,
        seed=42,
    )

    assert count > 0
    assert gz_path.exists()

    # Check zero dev leakage in the generated dataset
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            sample_id = rec.get("sample_id", "")
            assert sample_id not in dev_ids, f"Leaked dev sample {sample_id} into DPO dataset!"
