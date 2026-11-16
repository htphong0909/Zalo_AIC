"""Tests for in-loop error recovery augmentation."""
from src.data_gen.recovery_augmenter import (
    create_empty_search_recovery_step,
    create_xref_repair_recovery_step,
)


def test_create_empty_search_recovery_step():
    step = {
        "id": "step_1",
        "step_category": "act",
        "step_calls": [{"name": "contacts.search", "arguments": {"query": "Hương"}}],
        "prompt_messages": [
            {"role": "user", "content": "Tìm số của Hương giúp em."},
        ],
        "completion_message": {
            "role": "assistant",
            "tool_calls": [{"type": "function", "function": {"name": "contacts.search", "arguments": {"query": "Hương"}}}],
        },
    }

    aug = create_empty_search_recovery_step(step)
    assert aug is not None
    assert aug["id"] == "step_1_rec_empty"
    # prompt_messages now has the failed search turn followed by the clean search
    assert len(aug["prompt_messages"]) == 3
    assert aug["prompt_messages"][1]["role"] == "assistant"
    assert aug["prompt_messages"][2]["content"] == "[]"


def test_create_xref_repair_recovery_step():
    step = {
        "id": "step_2",
        "step_category": "act",
        "step_calls": [{"name": "smart_home.light_set", "arguments": {"device_id": "light_1", "power": "on"}}],
        "prompt_messages": [
            {"role": "user", "content": "Bật đèn phòng khách giúp em."},
        ],
        "completion_message": {
            "role": "assistant",
            "tool_calls": [{"type": "function", "function": {"name": "smart_home.light_set", "arguments": {"device_id": "light_1", "power": "on"}}}],
        },
    }

    aug = create_xref_repair_recovery_step(step)
    assert aug is not None
    assert aug["id"] == "step_2_rec_xref"
    assert len(aug["prompt_messages"]) == 3
    assert "fake_dev_999" in str(aug["prompt_messages"][1])
    assert '"hint":' in str(aug["prompt_messages"][2]["content"])


def test_read_to_premature_write_matches_official_catalog_and_isolated_search():
    from src.core.dataset import load_catalog
    from src.data_gen.recovery_augmenter import _READ_TO_PREMATURE_WRITE

    cat = load_catalog("train", "data")
    for read_tool, (write_tool, args) in _READ_TO_PREMATURE_WRITE.items():
        assert read_tool in cat
        assert write_tool in cat
        params_spec = cat[write_tool]["parameters"]
        props = params_spec["properties"]
        for k in args:
            assert k in props, f"{k} invalid for {write_tool}"
        for req in params_spec.get("required") or []:
            assert req in args, f"{req} missing for {write_tool}"

    multi_step = {
        "id": "multi_s0",
        "step_category": "act",
        "step_calls": [
            {"name": "alarm.create", "arguments": {"time": "06:00"}},
            {"name": "contacts.search", "arguments": {"query": "Lan"}},
        ],
        "prompt_messages": [{"role": "user", "content": "Đặt báo thức và tìm số Lan"}],
    }
    aug = create_empty_search_recovery_step(multi_step)
    assert aug is not None
    assert [c["name"] for c in aug["step_calls"]] == ["contacts.search"]
    assert any(m["role"] == "tool" and m["name"] == "alarm.create" for m in aug["prompt_messages"])
