"""Unit tests for When2Call (NAACL 2025) Tri-Branch Contrastive Data Generator."""
from __future__ import annotations

import pytest
from src.core.schemas import ASK_MISSING_INFO, Sample, ToolCall
from src.data_gen.when2call_gen import (
    drop_required_param_from_utterance,
    generate_when2call_contrastive_samples,
)


def test_drop_required_param_amount():
    text = "Chuyển 500 nghìn cho Hương qua ví điện tử"
    param_name = "amount"
    modified = drop_required_param_from_utterance(text, param_name)
    assert "500 nghìn" not in modified
    assert "Hương" in modified

    text_k = "Chuyển 50k cho Nam nhé"
    modified_k = drop_required_param_from_utterance(text_k, "amount")
    assert "50k" not in modified_k
    assert "Nam" in modified_k

    text_word = "Chuyển hai triệu tám cho chủ nhà"
    modified_word = drop_required_param_from_utterance(text_word, "amount")
    assert "hai triệu tám" not in modified_word
    assert "chủ nhà" in modified_word


def test_drop_required_param_time():
    text = "Đặt báo thức lúc 6 giờ sáng mai nhé"
    param_name = "time"
    modified = drop_required_param_from_utterance(text, param_name)
    assert "6" not in modified
    assert "báo thức" in modified

    text_h = "Nhắc tôi uống thuốc lúc 8h tối"
    modified_h = drop_required_param_from_utterance(text_h, "time")
    assert "8h" not in modified_h
    assert "uống thuốc" in modified_h


def test_drop_required_param_contact():
    text = "Gửi tin nhắn cho Hương nội dung là về sớm nhé"
    param_name = "recipient"
    modified = drop_required_param_from_utterance(text, param_name)
    assert "Hương" not in modified
    assert "tin nhắn" in modified


def test_drop_required_param_phone():
    text = "Chuyển tiền đến số 0987654321 qua ngân hàng"
    param_name = "recipient_phone"
    modified = drop_required_param_from_utterance(text, param_name)
    assert "0987654321" not in modified
    assert "Chuyển tiền" in modified


def test_drop_required_param_content():
    text = "Nhắn cho Lan là tối nay con về muộn nhé"
    param_name = "content"
    modified = drop_required_param_from_utterance(text, param_name)
    assert "tối nay con về muộn" not in modified
    assert "Lan" in modified


def test_drop_required_param_with_explicit_val():
    text = "Tắt đèn phòng khách ngay"
    modified = drop_required_param_from_utterance(text, "device_id", param_val="phòng khách")
    assert "phòng khách" not in modified
    assert "Tắt đèn" in modified


def test_generate_when2call_samples():
    sample = Sample(
        id="train_mock_01",
        audio="mock.wav",
        audio_text="Chuyển 500 nghìn cho Hương qua ví điện tử",
        context={
            "current_time": "2026-11-10T10:00:00+07:00",
            "user_profile": {"name": "Test User"},
            "installed_apps": {"ewallet": {"tools": ["ewallet.transfer"]}},
        },
        label={
            "tool_calls": [
                {"name": "ewallet.transfer", "arguments": {"recipient_phone": "0987654321", "amount": 500000}}
            ]
        },
    )
    catalog = {
        "ewallet.transfer": {
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient_phone": {"type": "string"},
                    "amount": {"type": "integer"},
                },
                "required": ["recipient_phone", "amount"],
            }
        }
    }
    results = generate_when2call_contrastive_samples([sample], catalog, count=1, seed=42)
    assert len(results) == 1
    res = results[0]
    assert res["step_category"] == "ask"
    assert any(c["name"] == ASK_MISSING_INFO for c in res["step_calls"])
    ask_call = next(c for c in res["step_calls"] if c["name"] == ASK_MISSING_INFO)
    assert ask_call["arguments"]["tool_name"] == "ewallet.transfer"
    assert "amount" in ask_call["arguments"]["params"] or "recipient_phone" in ask_call["arguments"]["params"]
    # Check 4-slot <think> trace boundary
    assert "think" in res
    assert "Quyết định: Cần hỏi lại (Ask Missing Info) trước khi thực hiện." in res["think"]
    assert "<think>" in res["think"] and "</think>" in res["think"]


def test_generate_when2call_empty_when_no_required_params():
    sample = Sample(
        id="train_mock_02",
        audio="mock.wav",
        audio_text="Xem dự báo thời tiết hôm nay",
        context={"installed_apps": {"weather": {"tools": ["weather.get_current"]}}},
        label={"tool_calls": [{"name": "weather.get_current", "arguments": {}}]},
    )
    catalog = {
        "weather.get_current": {
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            }
        }
    }
    results = generate_when2call_contrastive_samples([sample], catalog, count=5)
    assert len(results) == 0


def test_generate_when2call_seed_reproducibility():
    sample = Sample(
        id="train_mock_03",
        audio="mock.wav",
        audio_text="Chuyển 500 nghìn cho Hương qua ví điện tử",
        context={"installed_apps": {"ewallet": {"tools": ["ewallet.transfer"]}}},
        label={
            "tool_calls": [
                {"name": "ewallet.transfer", "arguments": {"recipient_phone": "0987654321", "amount": 500000}}
            ]
        },
    )
    catalog = {
        "ewallet.transfer": {
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient_phone": {"type": "string"},
                    "amount": {"type": "integer"},
                },
                "required": ["recipient_phone", "amount"],
            }
        }
    }
    r1 = generate_when2call_contrastive_samples([sample], catalog, count=2, seed=123)
    r2 = generate_when2call_contrastive_samples([sample], catalog, count=2, seed=123)
    assert len(r1) == len(r2)
    assert r1[0]["step_calls"] == r2[0]["step_calls"]


def test_when2call_skips_xref_id_conditional_and_preserves_lowercase_clauses():
    # Lowercase Vietnamese 'tới thì chuyển' must NOT be deleted by capitalized proper-noun pattern
    out_txt = drop_required_param_from_utterance(
        "Vàng trên 124 triệu, chưa tới thì chuyển 2 triệu cho Lê Minh nhé.", "recipient"
    )
    assert "chưa tới thì chuyển" in out_txt
    assert "Lê Minh" not in out_txt

    cat = {
        "banking.transfer": {
            "parameters": {
                "type": "object",
                "properties": {
                    "beneficiary_id": {"type": "string", "x-ref": "banking.beneficiaries"},
                    "amount": {"type": "integer"},
                },
                "required": ["beneficiary_id", "amount"],
            }
        }
    }
    s_cond = Sample(
        id="cond_01",
        audio_text="Nếu đủ tiền thì chuyển 500 nghìn cho Lê Minh nhé.",
        context={"installed_apps": {"banking": {"tools": ["banking.transfer"]}}},
        label={"tool_calls": [{"name": "banking.transfer", "arguments": {"beneficiary_id": "b1", "amount": 500000}}]},
    )
    assert generate_when2call_contrastive_samples([s_cond], cat, count=5) == []

    s_id_only = Sample(
        id="id_only_01",
        audio_text="Chuyển tiền cho Lê Minh giúp tôi.",
        context={"installed_apps": {"banking": {"tools": ["banking.transfer"]}}},
        label={"tool_calls": [{"name": "banking.transfer", "arguments": {"beneficiary_id": "b1"}}]},
    )
    assert generate_when2call_contrastive_samples([s_id_only], cat, count=5) == []
