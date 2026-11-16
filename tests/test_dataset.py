import json
from pathlib import Path

import pytest

from src.core.dataset import audio_path, load_catalog, load_split
from src.core.schemas import Sample

ROW = {
    "id": "train_9999",
    "audio": "audio/train_9999/train_9999.wav",
    "audio_text": "Tắt đèn bếp",
    "context": {
        "current_time": "2026-11-16T17:00:00+07:00",
        "timezone": "Asia/Ho_Chi_Minh",
        "installed_apps": {"smart_home": {"tools": ["smart_home.get_status", "smart_home.light_set"]}},
    },
    "history": [
        {"role": "user", "audio": "audio/train_9999/train_9999_u1.wav"},
        {"role": "assistant", "text": "OK", "tool_calls": [{"name": "smart_home.get_status", "arguments": {}}]},
    ],
    "label": {
        "tool_calls": [{"name": "smart_home.light_set", "arguments": {"device_id": "light_kitchen", "power": "off"}}],
        "tool_results": [{"ok": True}],
        "final_state": {},
    },
}


def test_sample_parses_and_exposes_allowed_tools():
    s = Sample.model_validate(ROW)
    assert s.history[1].tool_calls[0].name == "smart_home.get_status"
    assert s.label.tool_calls[0].arguments["power"] == "off"
    assert s.allowed_tools == {"smart_home.get_status", "smart_home.light_set", "sys.ask_missing_info"}


def test_load_split_from_tmp(tmp_path: Path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "train.jsonl").write_text(json.dumps(ROW, ensure_ascii=False) + "\n", encoding="utf-8")
    samples = load_split("train", tmp_path)
    assert [s.id for s in samples] == ["train_9999"]


def test_audio_path_rejects_environment_dir(tmp_path: Path):
    with pytest.raises(PermissionError):
        audio_path("train", tmp_path, "train_environment/train_0001/state.json")


def test_real_data_counts(data_root: Path):
    assert len(load_split("train", data_root)) == 1545
    assert len(load_split("public_test", data_root)) == 634
    catalog = load_catalog("train", data_root)
    assert len(catalog) == 147 and "sys.ask_missing_info" in catalog
