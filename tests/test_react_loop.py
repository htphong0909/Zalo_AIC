from __future__ import annotations

from src.agent.llm_client import SequenceFakeBackend
from src.agent.react_loop import run_react_loop
from src.core.dataset import load_catalog, load_split
from src.env.safe_env import SafeEnv


def test_react_loop_multi_step_read_then_write():
    catalog = load_catalog("train", "data")
    sample = next(
        s
        for s in load_split("train", "data")
        if [c.name for c in s.label.tool_calls] == ["smart_home.get_status", "smart_home.light_set"]
    )
    gold_write = sample.label.tool_calls[1]

    fake_llm = SequenceFakeBackend(
        [
            "<tool_call>\n<function=smart_home.get_status>\n</function>\n</tool_call>",
            (
                "<tool_call>\n<function=smart_home.light_set>\n"
                f"<parameter=device_id>\n{gold_write.arguments['device_id']}\n</parameter>\n"
                f"<parameter=power>\n{gold_write.arguments.get('power', '')}\n</parameter>\n"
                f"<parameter=brightness>\n{gold_write.arguments.get('brightness', '')}\n</parameter>\n"
                "</function>\n</tool_call>"
            ),
            "Đã thực hiện xong.",
        ]
    )
    env = SafeEnv("train", sample.id, "data")
    res = run_react_loop(sample, env, catalog, fake_llm)

    assert [c["name"] for c in res.tool_calls] == ["smart_home.get_status", "smart_home.light_set"]
    assert res.repairs == 0
    assert res.final_text == "Đã thực hiện xong."


def test_react_loop_preflight_validation_repairs_without_logging_rejected_call():
    catalog = load_catalog("train", "data")
    sample = next(
        s
        for s in load_split("train", "data")
        if any(c.name == "alarm.create" for c in s.label.tool_calls)
    )
    gold_alarm = next(c for c in sample.label.tool_calls if c.name == "alarm.create")

    # Step 1: invalid call (missing required 'time'); Step 2: repaired call; Step 3: done
    fake_llm = SequenceFakeBackend(
        [
            "<tool_call>\n<function=alarm.create>\n</function>\n</tool_call>",
            (
                "<tool_call>\n<function=alarm.create>\n"
                f"<parameter=time>\n{gold_alarm.arguments['time']}\n</parameter>\n"
                "</function>\n</tool_call>"
            ),
            "Đã đặt báo thức.",
        ]
    )
    env = SafeEnv("train", sample.id, "data")
    res = run_react_loop(sample, env, catalog, fake_llm, max_repairs=2)

    assert res.repairs == 1
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0]["name"] == "alarm.create"
    assert res.tool_calls[0]["arguments"] == {"time": gold_alarm.arguments["time"]}


def test_react_loop_skips_unrecoverable_schema_error_after_max_repairs():
    catalog = load_catalog("train", "data")
    sample = next(
        s
        for s in load_split("train", "data")
        if any(c.name == "alarm.create" for c in s.label.tool_calls)
    )
    fake_llm = SequenceFakeBackend(
        [
            "<tool_call>\n<function=alarm.create>\n</function>\n</tool_call>",
            "<tool_call>\n<function=alarm.create>\n</function>\n</tool_call>",
            "<tool_call>\n<function=alarm.create>\n</function>\n</tool_call>",
        ]
    )
    env = SafeEnv("train", sample.id, "data")
    res = run_react_loop(sample, env, catalog, fake_llm, max_repairs=2)

    assert res.repairs == 2
    assert res.tool_calls == []


def test_react_loop_stops_immediately_on_ask_missing_info():
    catalog = load_catalog("train", "data")
    sample = load_split("train", "data")[0]
    fake_llm = SequenceFakeBackend(
        [
            (
                "<tool_call>\n<function=sys.ask_missing_info>\n"
                "<parameter=tool_name>\nphone.call\n</parameter>\n"
                '<parameter=params>\n["contact_id"]\n</parameter>\n'
                "</function>\n</tool_call>"
            ),
            "<tool_call>\n<function=smart_home.get_status>\n</function>\n</tool_call>",
        ]
    )
    env = SafeEnv("train", sample.id, "data")
    res = run_react_loop(sample, env, catalog, fake_llm)

    assert len(res.tool_calls) == 1
    assert res.tool_calls[0]["name"] == "sys.ask_missing_info"


def test_react_loop_blocks_hallucinated_xref_id_and_executes_valid_read():
    catalog = load_catalog("train", "data")
    sample = next(
        s
        for s in load_split("train", "data")
        if [c.name for c in s.label.tool_calls] == ["smart_home.get_status", "smart_home.light_set"]
    )
    gold_write = sample.label.tool_calls[1]

    fake_llm = SequenceFakeBackend(
        [
            # Step 1: tries to call light_set with a fake ID without calling get_status first
            "<tool_call>\n<function=smart_home.light_set>\n<parameter=device_id>\nfake_id_999\n</parameter>\n<parameter=power>\noff\n</parameter>\n</function>\n</tool_call>",
            # Step 2: calls get_status
            "<tool_call>\n<function=smart_home.get_status>\n</function>\n</tool_call>",
            # Step 3: calls light_set with real ID from get_status
            (
                "<tool_call>\n<function=smart_home.light_set>\n"
                f"<parameter=device_id>\n{gold_write.arguments['device_id']}\n</parameter>\n"
                f"<parameter=power>\n{gold_write.arguments.get('power', '')}\n</parameter>\n"
                f"<parameter=brightness>\n{gold_write.arguments.get('brightness', '')}\n</parameter>\n"
                "</function>\n</tool_call>"
            ),
            "Xong.",
        ]
    )
    env = SafeEnv("train", sample.id, "data")
    res = run_react_loop(sample, env, catalog, fake_llm)

    assert [c["name"] for c in res.tool_calls] == ["smart_home.get_status", "smart_home.light_set"]
    assert res.tool_calls[1]["arguments"]["device_id"] == gold_write.arguments["device_id"]


def test_react_loop_ask_guard_intercepts_ambiguous_duplicate_contact():
    from src.core.config import GuardrailConfig

    catalog = load_catalog("train", "data")
    sample = next(
        s
        for s in load_split("train", "data")
        if [c.name for c in s.label.tool_calls] == ["contacts.search", "sys.ask_missing_info"]
        and s.label.tool_calls[1].arguments.get("tool_name") == "phone.call"
    )
    search_arg = sample.label.tool_calls[0].arguments.get("query", "Minh")
    first_cid = sample.label.tool_results[0]["contacts"][0]["id"]

    fake_llm = SequenceFakeBackend(
        [
            f"<tool_call>\n<function=contacts.search>\n<parameter=query>\n{search_arg}\n</parameter>\n</function>\n</tool_call>",
            f"<tool_call>\n<function=phone.call>\n<parameter=contact_id>\n{first_cid}\n</parameter>\n</function>\n</tool_call>",
        ]
    )
    env = SafeEnv("train", sample.id, "data")
    res = run_react_loop(
        sample,
        env,
        catalog,
        fake_llm,
        guardrails=GuardrailConfig(enable_ask_guard=True),
    )

    assert [c["name"] for c in res.tool_calls] == ["contacts.search", "sys.ask_missing_info"]
    assert res.tool_calls[1]["arguments"] == {"tool_name": "phone.call", "params": ["contact_id"]}


def test_react_loop_ask_guard_intercepts_vague_transfer_amount():
    from src.core.config import GuardrailConfig

    catalog = load_catalog("train", "data")
    sample = next(
        s
        for s in load_split("train", "data")
        if any(
            c.name == "sys.ask_missing_info"
            and c.arguments.get("tool_name") == "ewallet.transfer"
            and c.arguments.get("params") == ["amount"]
            for c in s.label.tool_calls
        )
    )
    fake_llm = SequenceFakeBackend(
        [
            "<tool_call>\n<function=ewallet.transfer>\n<parameter=contact_id>\nc_01\n</parameter>\n<parameter=amount>\n50000\n</parameter>\n</function>\n</tool_call>"
        ]
    )
    env = SafeEnv("train", sample.id, "data")
    res = run_react_loop(
        sample,
        env,
        catalog,
        fake_llm,
        guardrails=GuardrailConfig(enable_ask_guard=True),
    )

    assert len(res.tool_calls) == 1
    assert res.tool_calls[0] == {
        "name": "sys.ask_missing_info",
        "arguments": {"tool_name": "ewallet.transfer", "params": ["amount"]},
    }


def test_react_loop_qwen35_prompt_style_and_malformed_repair():
    catalog = load_catalog("train", "data")
    sample = next(
        s for s in load_split("train", "data") if any(c.name == "alarm.create" for c in s.label.tool_calls)
    )
    gold_alarm = next(c for c in sample.label.tool_calls if c.name == "alarm.create")
    fake_llm = SequenceFakeBackend(
        [
            "<tool_call>\n<function=alarm.create>\n<parameter=time>broken",
            (
                "<tool_call>\n<function=alarm.create>\n"
                f"<parameter=time>\n{gold_alarm.arguments['time']}\n</parameter>\n"
                "</function>\n</tool_call>"
            ),
            "Đã xong.",
        ]
    )
    env = SafeEnv("train", sample.id, "data")
    res = run_react_loop(sample, env, catalog, fake_llm, prompt_style="qwen35", max_repairs=2)

    assert res.repairs == 1
    assert len(res.tool_calls) == 1
    first_sys = fake_llm.calls_log[0]["messages"][0]["content"]
    assert "<tool_call><function=tên_tool>" in first_sys
    repair_msg = fake_llm.calls_log[1]["messages"][-1]["content"]
    assert "<tool_call><function=tên_tool>" in repair_msg


def test_react_loop_allows_spoken_and_freetext_and_corrected_phone_numbers():
    from src.core.schemas import Sample

    class _DummyPhoneEnv:
        def __init__(self) -> None:
            self._calls: list[dict[str, object]] = []

        def validate(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
            if name in ("phone.call", "sys.ask_missing_info"):
                return {"ok": True}
            return {"ok": False, "error": f"unknown tool {name}"}

        def call(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
            self._calls.append({"name": name, "arguments": dict(arguments)})
            return {"ok": True}

        def get_tool_calls(self) -> list[dict[str, object]]:
            return list(self._calls)

        def get_state_hash(self) -> str:
            return "dummy_hash"

    catalog = load_catalog("train", "data")
    sample_spoken = Sample(
        id="train_0541",
        audio="dummy.wav",
        audio_text="Gọi số không chín không một, hai ba bốn, chín chín chín sáu giúp em",
        context={"user_profile": {}, "installed_apps": {"phone": {}, "contacts": {}}},
        history=[],
    )
    llm_spoken = SequenceFakeBackend(
        [
            "<tool_call>\n<function=phone.call>\n<parameter=phone_number>\n09012349996\n</parameter>\n</function>\n</tool_call>",
            "Đang gọi.",
        ]
    )
    res_spoken = run_react_loop(sample_spoken, _DummyPhoneEnv(), catalog, llm_spoken)
    assert res_spoken.repairs == 0
    assert res_spoken.tool_calls == [{"name": "phone.call", "arguments": {"phone_number": "09012349996"}}]

    # Corrected trailing digits from history phone
    sample_corr = Sample(
        id="train_corr",
        audio="dummy.wav",
        audio_text="Sáng nay tôi đọc nhầm số, đuôi là sáu tám chứ không phải tám sáu, gọi lại giúp tôi.",
        context={"user_profile": {}, "installed_apps": {"phone": {}}},
        history=[{"role": "user", "text": "Gọi giùm số 09832415086"}],
    )
    llm_corr = SequenceFakeBackend(
        [
            "<tool_call>\n<function=phone.call>\n<parameter=phone_number>\n09832415068\n</parameter>\n</function>\n</tool_call>",
            "Đang gọi.",
        ]
    )
    res_corr = run_react_loop(sample_corr, _DummyPhoneEnv(), catalog, llm_corr)
    assert res_corr.repairs == 0
    assert res_corr.tool_calls == [{"name": "phone.call", "arguments": {"phone_number": "09832415068"}}]

    # Phone number extracted from chat_ott.read_chat / unstructured history tool result text
    sample_msg = Sample(
        id="train_msg",
        audio="dummy.wav",
        audio_text="Gọi lại số trong tin nhắn đó cho tôi",
        context={"user_profile": {}, "installed_apps": {"phone": {}, "chat_ott": {}}},
        history=[
            {
                "role": "assistant",
                "text": "Đã đọc tin nhắn",
                "tool_calls": [{"name": "chat_ott.read_chat", "arguments": {"chat_id": "chat_06"}}],
                "tool_results": [{"messages": [{"text": "Số sự cố mới bên mình là 09098877660 nhé"}]}],
            }
        ],
    )
    llm_msg = SequenceFakeBackend(
        [
            "<tool_call>\n<function=phone.call>\n<parameter=phone_number>\n09098877660\n</parameter>\n</function>\n</tool_call>",
            "Đang gọi.",
        ]
    )
    res_msg = run_react_loop(sample_msg, _DummyPhoneEnv(), catalog, llm_msg)
    assert res_msg.repairs == 0
    assert res_msg.tool_calls == [{"name": "phone.call", "arguments": {"phone_number": "09098877660"}}]

    # Truly hallucinated phone still gets blocked by xref_guard (not app check)
    sample_fake = Sample(
        id="train_fake",
        audio="dummy.wav",
        audio_text="Gọi cho mẹ đi",
        context={"user_profile": {}, "installed_apps": {"phone": {}}},
        history=[],
    )
    llm_fake = SequenceFakeBackend(
        [
            "<tool_call>\n<function=phone.call>\n<parameter=phone_number>\n0999888777\n</parameter>\n</function>\n</tool_call>",
            "Không tìm thấy số.",
        ]
    )
    res_fake = run_react_loop(sample_fake, _DummyPhoneEnv(), catalog, llm_fake, max_repairs=1)
    assert res_fake.repairs == 1
    assert res_fake.tool_calls == []
    assert "không có trong câu nói" in res_fake.trace[0]["validations"][0]["error"]


def test_react_loop_read_before_ask_injects_prereq_lookup():
    from src.core.schemas import Sample

    catalog = load_catalog("train", "data")
    sample = next(
        s for s in load_split("train", "data") if any(c.name == "notes.list" for c in s.label.tool_calls)
    )
    fake_llm = SequenceFakeBackend(
        [
            (
                "<tool_call>\n<function=sys.ask_missing_info>\n"
                "<parameter=tool_name>\nnotes.append\n</parameter>\n"
                '<parameter=params>\n["note_id"]\n</parameter>\n'
                "</function>\n</tool_call>"
            ),
            (
                "<tool_call>\n<function=sys.ask_missing_info>\n"
                "<parameter=tool_name>\nnotes.append\n</parameter>\n"
                '<parameter=params>\n["note_id"]\n</parameter>\n'
                "</function>\n</tool_call>"
            ),
        ]
    )
    env = SafeEnv("train", sample.id, "data")
    res = run_react_loop(sample, env, catalog, fake_llm)
    assert [c["name"] for c in res.tool_calls] == ["notes.list", "sys.ask_missing_info"]


def test_react_loop_boundary_verifier_ambiguity_and_guided_repair():
    from src.core.schemas import Sample

    catalog = load_catalog("train", "data")
    sample = next(
        s for s in load_split("train", "data") if any(c.name == "notes.append" for c in s.label.tool_calls)
    )
    dup_sample = Sample(
        id=sample.id,
        audio=sample.audio,
        audio_text="Ghi thêm vô ghi chú Chi tiêu là đã trả",
        context={"user_profile": {}, "installed_apps": {"notes": {}}},
        history=[
            {
                "role": "assistant",
                "text": "Danh sách ghi chú",
                "tool_calls": [{"name": "notes.list", "arguments": {}}],
                "tool_results": [
                    {
                        "notes": [
                            {"id": "note_01", "title": "Chi tiêu"},
                            {"id": "note_02", "title": "Chi tiêu"},
                        ]
                    }
                ],
            }
        ],
    )

    class _RecordingBackend:
        def __init__(self, responses):
            self.responses = list(responses)
            self.force_flags = []
            self.calls = []

        def generate(self, messages, tools, *, force_tool_schema=False):
            self.force_flags.append(force_tool_schema)
            self.calls.append(messages)
            return self.responses.pop(0)

    gold_alarm_sample = next(
        s for s in load_split("train", "data") if any(c.name == "alarm.create" for c in s.label.tool_calls)
    )
    gold_alarm = next(c for c in gold_alarm_sample.label.tool_calls if c.name == "alarm.create")
    rec_llm = _RecordingBackend(
        [
            "<tool_call>\n<function=alarm.create>\n</function>\n</tool_call>",
            (
                "<tool_call>\n<function=alarm.create>\n"
                f"<parameter=time>\n{gold_alarm.arguments['time']}\n</parameter>\n"
                "</function>\n</tool_call>"
            ),
            "Xong.",
        ]
    )
    res_rec = run_react_loop(
        gold_alarm_sample, SafeEnv("train", gold_alarm_sample.id, "data"), catalog, rec_llm, max_repairs=3
    )
    assert rec_llm.force_flags[0] is False
    assert rec_llm.force_flags[1] is True
    assert "time" in rec_llm.calls[1][-1]["content"]
    assert res_rec.repairs == 1

    # BoundaryVerifier duplicate non-contact entity guard -> asks clarification
    llm_dup = SequenceFakeBackend(
        [
            (
                "<tool_call>\n<function=notes.append>\n"
                "<parameter=note_id>\nnote_01\n</parameter>\n"
                "<parameter=content>\nđã trả\n</parameter>\n"
                "</function>\n</tool_call>"
            )
        ]
    )
    res_dup = run_react_loop(dup_sample, SafeEnv("train", sample.id, "data"), catalog, llm_dup)
    assert res_dup.tool_calls == [
        {"name": "sys.ask_missing_info", "arguments": {"tool_name": "notes.append", "params": ["note_id"]}}
    ]



