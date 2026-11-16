"""Unit tests for ToolSandbox (NAACL 2025) Milestone Anchor Binding Generator."""
from __future__ import annotations

import json
import pytest

from src.core.schemas import Label, Sample, ToolCall
from src.data_gen.anchor_binding_gen import (
    generate_anchor_binding_samples,
    render_minicpm5_function_call_xml,
)


def test_render_minicpm5_function_call_xml():
    call_xml = render_minicpm5_function_call_xml(
        "contacts.search",
        {"query": "Sếp Dũng"},
    )
    assert call_xml == '<function name="contacts.search"><param name="query">Sếp Dũng</param></function>'

    empty_xml = render_minicpm5_function_call_xml("smart_home.get_status", {})
    assert empty_xml == '<function name="smart_home.get_status"></function>'


def test_generate_anchor_binding_samples_basic():
    sample = Sample(
        id="train_mock_read_write",
        audio_text="Gọi cho Sếp Dũng nhé",
        context={"user_profile": {"name": "Test"}},
        label={
            "tool_calls": [
                {"name": "contacts.search", "arguments": {"query": "Sếp Dũng"}},
                {"name": "phone.call", "arguments": {"contact_id": "c_08"}},
            ],
            "tool_results": [
                {"ok": True, "contacts": [{"id": "c_08", "name": "Sếp Dũng", "phone": "09925652941"}]}
            ],
        },
    )
    catalog = {
        "contacts.search": {
            "kind": "read",
            "parameters": {"properties": {"query": {"type": "string"}}},
        },
        "phone.call": {
            "kind": "write",
            "parameters": {"properties": {"contact_id": {"type": "string"}}},
        },
    }
    results = generate_anchor_binding_samples([sample], catalog, count=1)
    assert len(results) >= 1
    step = results[0]

    # Verify ChatML structure
    assert step["id"] == "anchor_train_mock_read_write_s1"
    assert step["sample_id"] == "train_mock_read_write"
    assert step["step_index"] == 1
    assert step["step_category"] == "act"

    # Verify prompt_messages contains observation from Step 0
    prompt_msgs = step["prompt_messages"]
    assert any(m.get("role") == "user" and "Sếp Dũng" in m.get("content", "") for m in prompt_msgs)
    assert any(m.get("role") == "assistant" and '<function name="contacts.search">' in m.get("content", "") for m in prompt_msgs)
    assert any(m.get("role") == "tool" and "c_08" in m.get("content", "") for m in prompt_msgs)

    # Check that Step 1 calls phone.call with contact_id="c_08"
    assert any(
        c.get("name") == "phone.call" and (c.get("arguments") or {}).get("contact_id") == "c_08"
        for c in step["step_calls"]
    )

    # Verify 4-slot <think> trace with Slot 2 grounding
    assert "think" in step
    think = step["think"]
    assert "<think>" in think and "</think>" in think
    assert "1. Ý định:" in think
    assert "2. Dữ liệu:" in think
    assert "contacts.search" in think
    assert "contact_id=\"c_08\"" in think
    assert "3. Kế hoạch:" in think
    assert "4. Ranh giới: Quyết định: Đủ điều kiện thực thi (Act)." in think


def test_generate_anchor_binding_samples_pydantic_models():
    sample = Sample(
        id="train_mock_bill",
        audio_text="Thanh toán hóa đơn tiền điện",
        context={
            "installed_apps": {
                "bill_payment": {"tools": ["bill_payment.list_bills", "bill_payment.pay"]}
            }
        },
        label=Label(
            tool_calls=[
                ToolCall(name="bill_payment.list_bills", arguments={}),
                ToolCall(name="bill_payment.pay", arguments={"bill_id": "bill_elec_01", "source": "bank"}),
            ],
            tool_results=[
                {"ok": True, "bills": [{"id": "bill_elec_01", "type": "electric", "amount": 350000}]}
            ],
        ),
    )
    catalog = {
        "bill_payment.list_bills": {
            "kind": "read",
            "parameters": {"properties": {}},
        },
        "bill_payment.pay": {
            "kind": "write",
            "parameters": {"properties": {"bill_id": {"type": "string"}}},
        },
    }
    results = generate_anchor_binding_samples([sample], catalog, count=10)
    assert len(results) == 1
    step = results[0]
    assert step["sample_id"] == "train_mock_bill"
    assert step["step_calls"][0]["arguments"]["bill_id"] == "bill_elec_01"
    assert "bill_payment.list_bills" in step["think"]
    assert "bill_id=\"bill_elec_01\"" in step["think"]


def test_generate_anchor_binding_skips_unbound_or_single_step():
    single_call_sample = Sample(
        id="mock_single",
        audio_text="Bật đèn phòng khách",
        context={},
        label={
            "tool_calls": [{"name": "smart_home.light_set", "arguments": {"device_id": "light_1"}}],
            "tool_results": [{"ok": True}],
        },
    )
    unbound_sample = Sample(
        id="mock_unbound",
        audio_text="Xem dự báo thời tiết rồi đặt báo thức",
        context={},
        label={
            "tool_calls": [
                {"name": "weather.get_current", "arguments": {}},
                {"name": "alarm.create_alarm", "arguments": {"time": "07:00"}},
            ],
            "tool_results": [
                {"ok": True, "weather": "rainy", "temp": 25}
            ],
        },
    )
    catalog = {
        "smart_home.light_set": {"kind": "write"},
        "weather.get_current": {"kind": "read"},
        "alarm.create_alarm": {"kind": "write"},
    }
    results = generate_anchor_binding_samples([single_call_sample, unbound_sample], catalog, count=10)
    assert len(results) == 0


def test_generate_anchor_binding_count_and_seed():
    samples = [
        Sample(
            id=f"sample_{i}",
            audio_text=f"Yêu cầu {i}",
            context={},
            label={
                "tool_calls": [
                    {"name": "calendar.list_events", "arguments": {}},
                    {"name": "calendar.delete_event", "arguments": {"event_id": f"evt_{i}"}},
                ],
                "tool_results": [{"events": [{"id": f"evt_{i}"}]}],
            },
        )
        for i in range(10)
    ]
    catalog = {
        "calendar.list_events": {"kind": "read"},
        "calendar.delete_event": {"kind": "write"},
    }
    r1 = generate_anchor_binding_samples(samples, catalog, count=3, seed=42)
    r2 = generate_anchor_binding_samples(samples, catalog, count=3, seed=42)
    assert len(r1) == 3
    assert len(r2) == 3
    assert [s["id"] for s in r1] == [s["id"] for s in r2]


def test_generate_anchor_binding_handles_complex_args_and_none_label():
    sample_none = Sample(id="none_label", audio_text="test", context={}, label=None)
    sample_complex = Sample(
        id="complex_args",
        audio_text="Thêm sự kiện mới",
        context={"installed_apps": {"calendar": {}}},
        label={
            "tool_calls": [
                {"name": "calendar.list_events", "arguments": {"filter": {"active": True}}},
                {
                    "name": "calendar.update_event",
                    "arguments": {"event_id": "e_99", "attendees": ["a@test.com", "b@test.com"], "notify": True},
                },
            ],
            "tool_results": [{"events": [{"id": "e_99"}]}],
        },
    )
    catalog = {
        "calendar.list_events": {"kind": "read"},
        "calendar.update_event": {"kind": "write"},
    }
    results = generate_anchor_binding_samples([sample_none, sample_complex], catalog, count=10)
    assert len(results) == 1
    step = results[0]
    assert step["sample_id"] == "complex_args"
    assert step["tools"] == ["calendar"]
    assert '<param name="filter">{"active": true}</param>' in step["prompt_messages"][2]["content"]
    assert step["step_calls"][0]["arguments"]["event_id"] == "e_99"
    assert "e_99" in step["think"]

