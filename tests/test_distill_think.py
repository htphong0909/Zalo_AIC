"""Tests for 4-slot micro-<think> reasoning distillation."""
from src.data_gen.distill_think import (
    format_4slot_think,
    synthesize_deterministic_think,
    attach_think_to_completion,
)


def test_format_4slot_think_structure():
    t = format_4slot_think(
        intent="Bật đèn phòng khách",
        grounding="Thời gian 20:00",
        action_plan="smart_home.light_set",
        boundary="Đủ điều kiện",
    )
    assert t.startswith("<think>")
    assert t.endswith("</think>")
    assert "1. Ý định:" in t
    assert "2. Dữ liệu:" in t
    assert "3. Kế hoạch:" in t
    assert "4. Ranh giới:" in t


def test_synthesize_deterministic_think_for_act_and_ask():
    # Act step
    act_step = {
        "step_category": "act",
        "step_calls": [{"name": "alarm.create", "arguments": {"time": "06:30"}}],
        "prompt_messages": [{"role": "user", "content": "Đặt báo thức 6 rưỡi sáng mai nhé."}],
    }
    think_act = synthesize_deterministic_think(act_step, sample_context={"current_time": "2026-11-10T20:00:00+07:00"})
    assert "alarm.create" in think_act
    assert "Đủ điều kiện thực thi" in think_act

    # Ask step
    ask_step = {
        "step_category": "ask",
        "step_calls": [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "phone.call", "params": ["contact_id"]}}],
        "prompt_messages": [{"role": "user", "content": "Gọi điện thoại giúp anh."}],
    }
    think_ask = synthesize_deterministic_think(ask_step)
    assert "sys.ask_missing_info" in think_ask
    assert "Cần hỏi lại" in think_ask


def test_attach_think_to_completion():
    think_block = "<think>\n1. Ý định: test\n</think>"
    raw_comp = "<tool_call>\n<function=foo></function>\n</tool_call>"
    attached = attach_think_to_completion(raw_comp, think_block)
    assert attached.startswith("<think>")
    assert attached.endswith("</tool_call>")
    assert "<function=foo>" in attached
