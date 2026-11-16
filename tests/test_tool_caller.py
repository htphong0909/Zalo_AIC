from __future__ import annotations

from src.agent.tool_caller import (
    extract_final_text,
    has_malformed_tool_call,
    parse_tool_calls,
    sanitize_tool_call,
)
from src.core.dataset import load_catalog
from src.core.schemas import Sample, ToolCall


def test_parse_qwen_xml_tool_calls_with_type_coercion():
    catalog = load_catalog("train", "data")
    raw = """<think>
Tôi cần chỉnh đèn bếp và đặt món ăn.
</think>
<tool_call>
<function=smart_home.light_set>
<parameter=device_id>
light_kitchen
</parameter>
<parameter=power>
on
</parameter>
<parameter=brightness>
80
</parameter>
</function>
</tool_call>
<tool_call>
<function=sys.ask_missing_info>
<parameter=tool_name>
sms.send
</parameter>
<parameter=params>
["contact_id"]
</parameter>
</function>
</tool_call>"""

    calls = parse_tool_calls(raw, catalog)
    assert len(calls) == 2
    assert calls[0].name == "smart_home.light_set"
    assert calls[0].arguments == {
        "device_id": "light_kitchen",
        "power": "on",
        "brightness": 80,
    }
    assert isinstance(calls[0].arguments["brightness"], int)
    assert calls[1].name == "sys.ask_missing_info"
    assert calls[1].arguments == {
        "tool_name": "sms.send",
        "params": ["contact_id"],
    }


def test_parse_preserves_leading_zero_phone_strings():
    catalog = load_catalog("train", "data")
    raw = """<tool_call>
<function=phone.call>
<parameter=phone_number>
0912345678
</parameter>
</function>
</tool_call>"""

    calls = parse_tool_calls(raw, catalog)
    assert len(calls) == 1
    assert calls[0].name == "phone.call"
    assert calls[0].arguments == {"phone_number": "0912345678"}
    assert isinstance(calls[0].arguments["phone_number"], str)


def test_parse_json_fallback_and_extract_final_text():
    catalog = load_catalog("train", "data")
    raw = """<think>Thinking...</think>
<tool_call>
{"name": "alarm.create", "arguments": {"time": "2026-11-14T06:30:00+07:00", "label": ""}}
</tool_call>
Đã đặt báo thức cho bạn."""

    calls = parse_tool_calls(raw, catalog)
    assert len(calls) == 1
    assert calls[0].name == "alarm.create"
    assert calls[0].arguments == {"time": "2026-11-14T06:30:00+07:00"}
    assert extract_final_text(raw) == "Đã đặt báo thức cho bạn."


def test_detects_malformed_tool_call():
    catalog = load_catalog("train", "data")
    broken = "<tool_call>\n<function=smart_home.light_set>\n<parameter=device_id>light_1"
    assert parse_tool_calls(broken, catalog) == []
    assert has_malformed_tool_call(broken, catalog) is True
    assert has_malformed_tool_call("Xin chào, tôi không thể giúp việc này.", catalog) is False


def test_sanitize_tool_call_cleans_empty_and_normalizes_contacts_and_addresses():
    sample = Sample(
        id="s1",
        audio="a.wav",
        audio_text="Tìm danh bạ của Hương dùm em với, sau đó đặt xe về nhà.",
        context={
            "user_profile": {"home_address": "123 Hoàng Hoa Thám, Hà Nội", "city": "Hà Nội"},
            "installed_apps": {},
        },
    )
    # 1. Cleans empty values and strips polite particles
    c1 = ToolCall(
        name="contacts.search",
        arguments={"query": "Hương dùm em với", "limit": None, "offset": ""},
    )
    san1 = sanitize_tool_call(c1, sample)
    assert san1.arguments == {"query": "Hương"}

    # 2. Resolves relative addresses ("nhà" -> home_address)
    c2 = ToolCall(
        name="ride_hailing.book",
        arguments={"pickup": "Cơ quan", "dropoff": "nhà", "notes": []},
    )
    san2 = sanitize_tool_call(c2, sample)
    assert san2.arguments == {"pickup": "Cơ quan", "dropoff": "123 Hoàng Hoa Thám, Hà Nội"}

