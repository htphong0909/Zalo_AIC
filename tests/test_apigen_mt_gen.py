"""Unit tests for APIGen-MT (NeurIPS 2025) Multi-Intent Chaining & Horizon-Aware Micro-Think."""
from __future__ import annotations

import copy
import pytest

from src.core.schemas import ASK_MISSING_INFO, Label, Sample, ToolCall
from src.data_gen.distill_think import synthesize_deterministic_think
from src.data_gen.apigen_mt_gen import (
    _CONNECTORS,
    generate_apigen_multi_intent_samples,
)


def test_horizon_aware_think():
    """Verify horizon awareness: non-empty remaining_subgoals renders into Slot 4."""
    step = {
        "step_category": "act",
        "step_calls": [{"name": "alarm.create", "arguments": {"time": "06:00"}}],
        "remaining_subgoals": ["sms.send"],
        "prompt_messages": [{"role": "user", "content": "Đặt báo thức 6h và gửi sms cho Lan"}],
    }
    think = synthesize_deterministic_think(step)
    assert "Còn: [sms.send]." in think


def test_horizon_aware_think_empty_and_multiple_subgoals():
    """Verify Slot 4 formatting for empty and multiple remaining subgoals."""
    # Empty subgoals
    step_empty = {
        "step_category": "act",
        "step_calls": [{"name": "alarm.create", "arguments": {"time": "06:00"}}],
        "remaining_subgoals": [],
        "prompt_messages": [{"role": "user", "content": "Đặt báo thức 6h"}],
    }
    think_empty = synthesize_deterministic_think(step_empty)
    assert "Còn:" not in think_empty

    # Multiple subgoals
    step_multi = {
        "step_category": "act",
        "step_calls": [{"name": "alarm.create", "arguments": {"time": "06:00"}}],
        "remaining_subgoals": ["sms.send", "email.send"],
        "prompt_messages": [{"role": "user", "content": "Đặt báo thức 6h, gửi sms và email cho Lan"}],
    }
    think_multi = synthesize_deterministic_think(step_multi)
    assert "Còn: [sms.send, email.send]." in think_multi


def test_generate_apigen_multi_intent_samples_basic():
    """Verify basic multi-intent pair synthesis with natural Vietnamese connectors."""
    sample1 = Sample(
        id="train_01",
        audio_text="Đặt báo thức 6 giờ",
        context={"installed_apps": {"alarm": {}}},
        label={"tool_calls": [{"name": "alarm.create", "arguments": {"time": "06:00"}}]},
    )
    sample2 = Sample(
        id="train_02",
        audio_text="Nhắn tin cho Lan",
        context={"installed_apps": {"sms": {}}},
        label={"tool_calls": [{"name": "sms.send", "arguments": {"contact_id": "c1", "content": "hello"}}]},
    )
    catalog = {
        "alarm.create": {"kind": "write", "parameters": {}},
        "sms.send": {"kind": "write", "parameters": {}},
    }
    results = generate_apigen_multi_intent_samples([sample1, sample2], catalog, count=1, seed=42)
    assert len(results) >= 1
    step = results[0]

    user_content = step["prompt_messages"][1]["content"]
    assert "và" in user_content or "đồng thời" in user_content or "với lại" in user_content or "rồi" in user_content
    assert step["step_index"] == 0
    assert step["step_category"] == "act"
    assert step["remaining_subgoals"] == ["sms.send"]
    assert step["step_calls"] == [{"name": "alarm.create", "arguments": {"time": "06:00"}}]
    assert "think" in step
    assert "Còn: [sms.send]." in step["think"]


def test_generate_apigen_multi_intent_samples_count_and_seed_reproducibility():
    """Verify sample count capping and deterministic seed reproducibility."""
    s1 = Sample(
        id="s1",
        audio_text="Bật đèn phòng khách",
        context={"installed_apps": {"smart_home": {}}},
        label={"tool_calls": [{"name": "smart_home.set", "arguments": {"device": "light", "state": "on"}}]},
    )
    s2 = Sample(
        id="s2",
        audio_text="Khóa cửa chính",
        context={"installed_apps": {"smart_home": {}}},
        label={"tool_calls": [{"name": "smart_home.set", "arguments": {"device": "door", "state": "locked"}}]},
    )
    s3 = Sample(
        id="s3",
        audio_text="Mở bài hát Nắng Ấm Xa Dần",
        context={"installed_apps": {"media_player": {}}},
        label={"tool_calls": [{"name": "media_player.play", "arguments": {"title": "Nắng Ấm Xa Dần"}}]},
    )
    catalog = {
        "smart_home.set": {"kind": "write", "parameters": {}},
        "media_player.play": {"kind": "write", "parameters": {}},
    }

    res_run1 = generate_apigen_multi_intent_samples([s1, s2, s3], catalog, count=4, seed=123)
    res_run2 = generate_apigen_multi_intent_samples([s1, s2, s3], catalog, count=4, seed=123)

    assert len(res_run1) == 4
    assert len(res_run2) == 4
    for r1, r2 in zip(res_run1, res_run2):
        assert r1["id"] == r2["id"]
        assert r1["prompt_messages"] == r2["prompt_messages"]
        assert r1["step_calls"] == r2["step_calls"]
        assert r1["remaining_subgoals"] == r2["remaining_subgoals"]


def test_generate_apigen_multi_intent_filters_and_tools_merging():
    """Verify filtering of OOS / ask_missing_info samples and correct tools merging."""
    # Valid sample with Pydantic Label
    s_valid = Sample(
        id="s_valid",
        audio_text="Đặt báo thức 7h",
        context={"installed_apps": {"alarm": {"tools": ["alarm.create"]}}},
        label=Label(tool_calls=[ToolCall(name="alarm.create", arguments={"time": "07:00"})]),
    )
    # Another valid sample with dict Label
    s_valid2 = Sample(
        id="s_valid2",
        audio_text="Kiểm tra thời tiết",
        context={"installed_apps": {"weather": {"tools": ["weather.get"]}}},
        label={"tool_calls": [{"name": "weather.get", "arguments": {"location": "Hà Nội"}}]},
    )
    # Ask missing info sample (should be skipped)
    s_ask = Sample(
        id="s_ask",
        audio_text="Chuyển tiền nhé",
        context={"installed_apps": {"banking": {}}},
        label={"tool_calls": [{"name": ASK_MISSING_INFO, "arguments": {"tool_name": "banking.transfer", "params": ["amount"]}}]},
    )
    # Empty tool calls sample (should be skipped)
    s_empty = Sample(
        id="s_empty",
        audio_text="Hôm nay trời đẹp quá",
        context={"installed_apps": {}},
        label={"tool_calls": []},
    )

    catalog = {
        "alarm.create": {"kind": "write", "parameters": {}},
        "weather.get": {"kind": "read", "parameters": {}},
    }

    # Only 1 valid sample -> returns empty list
    res_single = generate_apigen_multi_intent_samples([s_valid, s_ask, s_empty], catalog, count=2)
    assert res_single == []

    # 2 valid samples -> returns generated step with merged tools
    res_pair = generate_apigen_multi_intent_samples([s_valid, s_valid2, s_ask, s_empty], catalog, count=1, seed=42)
    assert len(res_pair) == 1
    item = res_pair[0]
    assert "alarm" in item["tools"]
    assert "weather" in item["tools"]


def test_generate_apigen_multi_intent_with_step1_option():
    """Verify include_step_1 generates both Step 0 and Step 1 with clear subgoal resolution."""
    sample1 = Sample(
        id="train_01",
        audio_text="Bật điều hòa 24 độ",
        context={"installed_apps": {"smart_home": {}}},
        label={"tool_calls": [{"name": "smart_home.set_temp", "arguments": {"temp": 24}}]},
    )
    sample2 = Sample(
        id="train_02",
        audio_text="Bật quạt gió",
        context={"installed_apps": {"smart_home": {}}},
        label={"tool_calls": [{"name": "smart_home.fan_on", "arguments": {}}]},
    )
    catalog = {
        "smart_home.set_temp": {"kind": "write", "parameters": {}},
        "smart_home.fan_on": {"kind": "write", "parameters": {}},
    }
    results = generate_apigen_multi_intent_samples(
        [sample1, sample2],
        catalog,
        count=2,
        seed=42,
        include_step_1=True,
    )
    assert len(results) == 2
    step0, step1 = results[0], results[1]

    # Step 0 checks
    assert step0["step_index"] == 0
    assert step0["remaining_subgoals"] == ["smart_home.fan_on"]
    assert "Còn: [smart_home.fan_on]." in step0["think"]

    # Step 1 checks
    assert step1["step_index"] == 1
    assert step1["remaining_subgoals"] == []
    assert any(m["role"] == "tool" for m in step1["prompt_messages"])
    assert step1["step_calls"] == [{"name": "smart_home.fan_on", "arguments": {}}]
    assert "Còn:" not in step1["think"]
