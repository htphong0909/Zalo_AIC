"""Unit tests for MiniCPM5 XML Tool Parser and its integration with tool_caller."""
from __future__ import annotations

from src.agent.minicpm5_parser import (
    extract_minicpm5_xml_calls,
    normalize_minicpm5_output,
)
from src.agent.tool_caller import (
    extract_final_text,
    has_malformed_tool_call,
    parse_tool_calls,
)
from src.core.schemas import ToolCall

MOCK_CATALOG = {
    "alarm.create_alarm": {
        "parameters": {
            "type": "object",
            "properties": {
                "time": {"type": "string"},
                "label": {"type": "string"},
                "repeat": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["time"],
        }
    },
    "smart_home.light_set": {
        "parameters": {
            "type": "object",
            "properties": {
                "device_id": {"type": "string"},
                "power": {"type": "string"},
                "brightness": {"type": "integer"},
                "is_active": {"type": "boolean"},
            },
            "required": ["device_id"],
        }
    },
    "notes.create_note": {
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "content": {"type": "string"},
                "tags": {"type": "array"},
                "metadata": {"type": "object"},
            },
            "required": ["title", "content"],
        }
    },
}


def test_normalize_output_handles_bpe_space_and_collapsed_tags():
    raw = "\u0120<functionname=\"contacts.search\">\u0120<paramname=\"query\">Mẹ</param>\u0120</function>"
    norm = normalize_minicpm5_output(raw)
    assert "\u0120" not in norm
    assert '<function name="contacts.search">' in norm
    assert '<param name="query">Mẹ</param>' in norm
    assert "</function>" in norm


def test_extract_single_call_with_think():
    text = """<think>
1. Intent: Đặt báo thức lúc 6:30.
2. State: Đủ tham số bắt buộc.
3. Decision: ACT -> alarm.create_alarm.
</think>
<function name="alarm.create_alarm">
<param name="time">06:30</param>
<param name="label">Tập thể dục buổi sáng</param>
</function>"""
    calls = extract_minicpm5_xml_calls(text, MOCK_CATALOG)
    assert len(calls) == 1
    assert calls[0].name == "alarm.create_alarm"
    assert calls[0].arguments == {
        "time": "06:30",
        "label": "Tập thể dục buổi sáng",
    }


def test_extract_parallel_calls():
    text = """<function name="smart_home.light_set">
<param name="device_id">dev_light_01</param>
<param name="power">on</param>
</function>
<function name="smart_home.light_set">
<param name="device_id">dev_light_02</param>
<param name="power">off</param>
</function>"""
    calls = extract_minicpm5_xml_calls(text, MOCK_CATALOG)
    assert len(calls) == 2
    assert calls[0].arguments == {"device_id": "dev_light_01", "power": "on"}
    assert calls[1].arguments == {"device_id": "dev_light_02", "power": "off"}


def test_type_conversions_via_catalog():
    text = """<function name="smart_home.light_set">
<param name="device_id">dev_01</param>
<param name="brightness">80</param>
<param name="is_active">true</param>
</function>"""
    calls = extract_minicpm5_xml_calls(text, MOCK_CATALOG)
    assert len(calls) == 1
    assert calls[0].arguments["device_id"] == "dev_01"
    assert calls[0].arguments["brightness"] == 80
    assert isinstance(calls[0].arguments["brightness"], int)
    assert calls[0].arguments["is_active"] is True


def test_cdata_and_structured_types():
    text = """<function name="notes.create_note">
<param name="title">Họp dự án</param>
<param name="content"><![CDATA[Nội dung cuộc họp: <thông báo> & lưu ý quan trọng]]></param>
<param name="tags">["công việc", "quan trọng"]</param>
<param name="metadata">{"priority": "high", "room": 302}</param>
</function>"""
    calls = extract_minicpm5_xml_calls(text, MOCK_CATALOG)
    assert len(calls) == 1
    assert calls[0].arguments["title"] == "Họp dự án"
    assert calls[0].arguments["content"] == "Nội dung cuộc họp: <thông báo> & lưu ý quan trọng"
    assert calls[0].arguments["tags"] == ["công việc", "quan trọng"]
    assert calls[0].arguments["metadata"] == {"priority": "high", "room": 302}


def test_tool_caller_integration_parse_and_extract_final_text():
    text = """<think>Đã hiểu yêu cầu</think>
<function name="alarm.create_alarm">
<param name="time">07:00</param>
</function>
Tôi đã tạo báo thức lúc 7 giờ sáng cho bạn."""
    calls = parse_tool_calls(text, MOCK_CATALOG)
    assert len(calls) == 1
    assert calls[0].name == "alarm.create_alarm"
    assert calls[0].arguments == {"time": "07:00"}

    final_text = extract_final_text(text)
    assert final_text == "Tôi đã tạo báo thức lúc 7 giờ sáng cho bạn."
    assert not has_malformed_tool_call(text, MOCK_CATALOG)


def test_has_malformed_tool_call_detection():
    # Broken function tag with missing closing
    broken = '<function name="broken_tool">'
    assert has_malformed_tool_call(broken, MOCK_CATALOG)


def test_strip_think_when_opening_think_is_in_prompt():
    # Simulates enable_thinking=True where '<think>\n' is part of the prompt
    raw_output = (
        '1. Ý định: Đặt báo thức.\n'
        '2. Dữ liệu: Không gọi <function name="smart_home.light_set"><param name="device_id">d1</param></function>.\n'
        '</think>\n\n'
        '<function name="alarm.create_alarm"><param name="time">07:30</param></function>'
    )
    calls = parse_tool_calls(raw_output, MOCK_CATALOG)
    assert len(calls) == 1
    assert calls[0].name == "alarm.create_alarm"
    assert calls[0].arguments == {"time": "07:30"}
    assert extract_final_text(raw_output) == ""


def test_minicpm5_dataset_formatting_and_template_masking():
    from jinja2 import Template
    from scripts.build_minicpm5_dataset import (
        convert_step_to_minicpm5_messages,
        format_minicpm5_function_xml,
        format_minicpm5_tools_block,
    )
    from scripts.train_minicpm5_sft import TRAIN_CHAT_TEMPLATE

    tools = [
        {
            "type": "function",
            "function": {
                "name": "contacts.search",
                "description": "Search contacts",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
    ]
    step = {
        "id": "s1",
        "step_index": 1,
        "step_category": "act",
        "prompt_messages": [
            {"role": "system", "content": "Sys prompt"},
            {"role": "user", "content": "Gọi cho Dũng"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "contacts.search", "arguments": {"query": "Dũng"}}},
                    {"type": "function", "function": {"name": "contacts.search", "arguments": {"query": "A & B"}}},
                ],
            },
            {"role": "tool", "name": "contacts.search", "content": '[{"contact_id": "c1"}]'},
            {"role": "tool", "name": "contacts.search", "content": "[]"},
        ],
        "step_calls": [{"name": "phone.call", "arguments": {"contact_id": "c1"}}],
        "tools": tools,
    }
    msgs = convert_step_to_minicpm5_messages(step, {"current_time": "2026-11-10T08:00:00+07:00"}, MOCK_CATALOG)

    # 1. System prompt uses official <tools>...</tools> block, not <|tool_list_start|>
    assert "<tools>" in msgs[0]["content"]
    assert "<|tool_list_start|>" not in msgs[0]["content"]

    # 2. Prior assistant turn uses <think>\n\n</think>\n\n and CDATA for '&'
    prior_asst = next(m for m in msgs[:-1] if m["role"] == "assistant")
    assert prior_asst["content"].startswith("<think>\n\n</think>\n\n")
    assert "<![CDATA[A & B]]>" in prior_asst["content"]

    # 3. Target assistant turn (last) has full 4-slot <think> + \n\n + <function>
    target_asst = msgs[-1]
    assert target_asst["role"] == "assistant"
    assert target_asst["content"].startswith("<think>\n1. Ý định:")
    assert "\n</think>\n\n<function name=\"phone.call\">" in target_asst["content"]

    # 4. Tool messages do not have [tool_name]: prefix
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["content"] == '[{"contact_id": "c1"}]'

    # 5. Verify TRAIN_CHAT_TEMPLATE groups consecutive tool responses and only unmasks loop.last
    # Replace {% generation %} and {% endgeneration %} with markers to test Jinja logic
    TestableTemplate = (
        TRAIN_CHAT_TEMPLATE.replace("{%- generation %}", "<<GEN_START>>")
        .replace("{%- endgeneration %}", "<<GEN_END>>")
    )
    rendered = Template(TestableTemplate).render(
        bos_token="<s>",
        messages=msgs,
        add_generation_prompt=False,
    )
    assert rendered.count("<<GEN_START>>") == 1
    assert rendered.count("<<GEN_END>>") == 1
    gen_segment = rendered.split("<<GEN_START>>")[1].split("<<GEN_END>>")[0]
    assert gen_segment.startswith("<think>\n1. Ý định:")
    assert '<function name="phone.call">' in gen_segment
    # Consecutive tool messages are grouped inside a single user block
    assert "<tool_response>\n[{\"contact_id\": \"c1\"}]\n</tool_response>\n<tool_response>\n[]\n</tool_response><|im_end|>" in rendered


def test_prior_assistant_turn_uses_python_repr_like_official_template():
    from scripts.build_minicpm5_dataset import format_minicpm5_function_xml

    call = [{"type": "function", "function": {"name": "t.f", "arguments": {
        "items": [{"item_id": "a", "quantity": 3}], "flag": True, "n": 5, "s": "x & y"}}}]
    prior = format_minicpm5_function_xml(call, python_repr=True)
    assert "{'item_id': 'a', 'quantity': 3}" in prior  # Python repr, as Jinja renders it at inference
    assert '<param name="flag">True</param>' in prior
    assert "<![CDATA[x & y]]>" in prior
    target = format_minicpm5_function_xml(call)  # loss-bearing target keeps canonical JSON
    assert '[{"item_id": "a", "quantity": 3}]' in target
    assert '<param name="flag">true</param>' in target


def test_collect_stop_token_ids_includes_im_end():
    from src.agent.llm_client import _collect_stop_token_ids

    class Tok:
        eos_token_id = 1
        unk_token_id = 2

        def convert_tokens_to_ids(self, t):
            return 130073 if t == "<|im_end|>" else 2

    assert _collect_stop_token_ids(Tok()) == [1, 130073]
