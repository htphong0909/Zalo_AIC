"""Test collection-exact x-ref guards and history read caching."""
from src.core.dataset import load_catalog, load_split
from src.core.schemas import Sample, ToolCall
from src.env.safe_env import SafeEnv
from src.agent.react_loop import _xref_error, _collect_ids_from_result, run_react_loop
from tests.test_react_loop import SequenceFakeBackend


def test_collection_exact_xref_blocks_playlist_as_track():
    catalog = load_catalog("train", "data")
    by_app = {
        "music": {"pl_01", "tr_02"},
        "music.playlists": {"pl_01"},
        "music.tracks": {"tr_02"},
    }

    # Attempting to pass playlist ID into track_id
    call_bad = ToolCall(name="music.play", arguments={"track_id": "pl_01"})
    err = _xref_error(call_bad, catalog, by_app)
    assert err is not None
    assert "playlist_id" in err

    # Valid track ID passes
    call_good = ToolCall(name="music.play", arguments={"track_id": "tr_02"})
    assert _xref_error(call_good, catalog, by_app) is None


def test_collection_exact_xref_blocks_scene_as_device():
    catalog = load_catalog("train", "data")
    by_app = {
        "smart_home": {"sc_01", "dev_02"},
        "smart_home.scenes": {"sc_01"},
        "smart_home.devices": {"dev_02"},
    }

    call_bad = ToolCall(name="smart_home.light_set", arguments={"device_id": "sc_01", "power": "on"})
    err = _xref_error(call_bad, catalog, by_app)
    assert err is not None
    assert "scene_id" in err

    call_good = ToolCall(name="smart_home.light_set", arguments={"device_id": "dev_02", "power": "on"})
    assert _xref_error(call_good, catalog, by_app) is None


def test_history_read_cache_prevents_duplicate_safe_env_call():
    catalog = load_catalog("train", "data")
    # Sample with history containing a read call
    sample = Sample(
        id="train_0001",
        audio="dummy.wav",
        audio_text="Gọi cho Lan đi em.",
        context={"user_profile": {}, "installed_apps": {"contacts": {}, "phone": {}}},
        history=[
            {
                "role": "assistant",
                "text": "Tìm số của Lan giúp anh.",
                "tool_calls": [{"name": "contacts.search", "arguments": {"query": "Lan"}}],
                "tool_results": [[{"id": "c_lan_01", "name": "Lan", "phone": "0912345678"}]],
            }
        ],
    )
    env = SafeEnv("train", "train_0001", "data")

    # The LLM calls contacts.search again, then phone.call with the retrieved contact_id
    fake_llm = SequenceFakeBackend(
        [
            "<tool_call>\n<function=contacts.search>\n<parameter=query>\nLan\n</parameter>\n</function>\n</tool_call>",
            '<tool_call>\n<function=sms.send>\n<parameter=contact_id>\nc_lan_01\n</parameter>\n<parameter=content>\nAlo\n</parameter>\n</function>\n</tool_call>',
            "Đã nhắn tin cho Lan.",
        ]
    )

    res = run_react_loop(sample, env, catalog, fake_llm)

    # In SafeEnv, contacts.search was cached from history, so env.get_tool_calls() contains ONLY sms.send!
    env_calls = env.get_tool_calls()
    assert len(env_calls) == 1
    assert env_calls[0]["name"] == "sms.send"
    assert env_calls[0]["arguments"] == {"contact_id": "c_lan_01", "content": "Alo"}


def test_email_xref_blocks_hallucinated_email_and_allows_seen_email():
    from src.core.dataset import Sample, load_catalog

    class _MemEmailEnv:
        def __init__(self):
            self.calls = []

        def validate(self, name, arguments):
            return {"ok": True}

        def call(self, name, arguments):
            self.calls.append({"name": name, "arguments": arguments})
            if name == "contacts.search":
                return {
                    "ok": True,
                    "contacts": [
                        {"id": "c_lan", "name": "Lan", "email": "lan.real@example.test"}
                    ],
                }
            return {"ok": True}

        def get_tool_calls(self):
            return list(self.calls)

        def get_state_hash(self):
            return "mem_hash"

    catalog = load_catalog("train", "data")
    ctx = {
        "current_time": "2026-10-01T09:00:00+07:00",
        "timezone": "Asia/Ho_Chi_Minh",
        "location": {"lat": 21.0, "lng": 105.8},
        "user_profile": {"name": "Minh", "home_address": "HN", "work_address": "HN", "city": "HN"},
        "installed_apps": {
            "contacts": {"tools": ["contacts.search", "contacts.get", "contacts.update"]},
            "email": {"tools": ["email.list", "email.read", "email.send", "email.create_draft"]},
        },
    }

    # 1) Hallucinated email must be blocked until looked up via contacts.search
    s1 = Sample(
        id="mem_email_1",
        audio_path="a.wav",
        audio_text="Gửi email cho chị Lan báo họp lúc 3h chiều.",
        context=ctx,
        history=[],
        label={"tool_calls": [], "state_hash": "x"},
    )
    llm1 = SequenceFakeBackend(
        [
            '<tool_call>\n<function=email.send>\n<parameter=to>\n["lan.guessed@example.test"]\n</parameter>\n<parameter=subject>\nHọp\n</parameter>\n<parameter=body>\nHọp 3h.\n</parameter>\n</function>\n</tool_call>',
            '<tool_call>\n<function=contacts.search>\n<parameter=query>\nLan\n</parameter>\n</function>\n</tool_call>',
            '<tool_call>\n<function=email.send>\n<parameter=to>\n["lan.real@example.test"]\n</parameter>\n<parameter=subject>\nHọp\n</parameter>\n<parameter=body>\nHọp 3h.\n</parameter>\n</function>\n</tool_call>',
            "Đã gửi email cho chị Lan.",
        ]
    )
    env1 = _MemEmailEnv()
    res1 = run_react_loop(s1, env1, catalog, llm1, max_repairs=2)
    assert [c["name"] for c in res1.tool_calls] == ["contacts.search", "email.send"]
    assert res1.tool_calls[1]["arguments"]["to"] == ["lan.real@example.test"]

    # 2) Spoken Vietnamese email dictation + CC from history turn.tool_calls (list arg) allowed; hallucinated CC blocked
    s2 = Sample(
        id="mem_email_2",
        audio_path="a.wav",
        audio_text="Soạn nháp gửi team chấm qa a còng example chấm test và cc cho anh cũ ở mail trước.",
        context=ctx,
        history=[
            {
                "role": "assistant",
                "text": "Gửi mail trước cho sếp",
                "tool_calls": [
                    {
                        "name": "email.send",
                        "arguments": {"to": ["sep.old@example.test"], "subject": "Hi", "body": "Hi"},
                    }
                ],
                "tool_results": [{"ok": True}],
            }
        ],
        label={"tool_calls": [], "state_hash": "x"},
    )
    llm2 = SequenceFakeBackend(
        [
            '<tool_call>\n<function=email.create_draft>\n<parameter=to>\n["team.qa@example.test"]\n</parameter>\n<parameter=cc>\n["invented@example.test"]\n</parameter>\n<parameter=subject>\nQA\n</parameter>\n<parameter=body>\nCheck.\n</parameter>\n</function>\n</tool_call>',
            '<tool_call>\n<function=email.create_draft>\n<parameter=to>\n["team.qa@example.test"]\n</parameter>\n<parameter=cc>\n["sep.old@example.test"]\n</parameter>\n<parameter=subject>\nQA\n</parameter>\n<parameter=body>\nCheck.\n</parameter>\n</function>\n</tool_call>',
            "Đã soạn nháp.",
        ]
    )
    env2 = _MemEmailEnv()
    res2 = run_react_loop(s2, env2, catalog, llm2, max_repairs=2)
    assert len(res2.tool_calls) == 1
    assert res2.tool_calls[0]["name"] == "email.create_draft"
    assert res2.tool_calls[0]["arguments"]["to"] == ["team.qa@example.test"]
    assert res2.tool_calls[0]["arguments"]["cc"] == ["sep.old@example.test"]

