from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

from src.agent.llm_client import SequenceFakeBackend, VLLMBackend, build_qwen35_tool_grammar
from src.agent.prompts import build_system_prompt
from src.agent.react_loop import run_react_loop
from src.asr.whisper_engine import WhisperASREngine
from src.core.dataset import load_catalog, load_split
from src.core.schemas import Sample


def test_qwen35_tool_grammar_supports_thinking_block(tmp_path, monkeypatch):
    import transformers

    tools = [
        {
            "type": "function",
            "function": {
                "name": "alarm.create",
                "parameters": {
                    "type": "object",
                    "required": ["time"],
                    "properties": {"time": {"type": "string"}},
                },
            },
        }
    ]
    strict_grammar = build_qwen35_tool_grammar(tools, allow_thinking=False)
    assert "root ::= call" in strict_grammar
    assert "think-block" not in strict_grammar

    think_grammar = build_qwen35_tool_grammar(tools, allow_thinking=True)
    assert "think-block" in think_grammar
    assert "</think>" in think_grammar
    assert "root ::= ( think-block )? call" in think_grammar

    tok_dir = tmp_path / "tok"
    tok_dir.mkdir()
    (tok_dir / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForCausalLM"]}), encoding="utf-8")

    class _DummyTok:
        pad_token_id = 0
        eos_token_id = 1

        def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=True, enable_thinking=False):
            return "<|im_start|>assistant\n<think>\n" if enable_thinking else "<|im_start|>assistant\n"

    class _FakeLLM:
        def __init__(self, **kwargs):
            pass

    class _FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeStructuredOutputsParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_so_mod = types.SimpleNamespace(StructuredOutputsParams=_FakeStructuredOutputsParams)
    fake_vllm = types.SimpleNamespace(LLM=_FakeLLM, SamplingParams=_FakeSamplingParams, sampling_params=fake_so_mod)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", fake_so_mod)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **kw: _DummyTok()))

    backend = VLLMBackend(model_id=str(tok_dir), max_model_len=4096, enable_thinking=True, guided_decoding=True)
    sp = backend._build_sampling_params(tools, force_tool_schema=True)
    grammar_used = sp.kwargs["structured_outputs"].kwargs["grammar"]
    assert "think-block" in grammar_used
    assert "</think>" in grammar_used


def test_whisper_diverse_nbest_preserves_greedy_1best_and_falls_back(tmp_path):
    wav_path = tmp_path / "sample.wav"
    import soundfile as sf

    sf.write(str(wav_path), np.zeros(1600, dtype=np.float32), 16000)

    recorded_calls: list[dict[str, Any]] = []

    class _MockProcessor:
        tokenizer = types.SimpleNamespace(unk_token_id=-1, convert_tokens_to_ids=lambda _tok: -1)

        def __call__(self, waveform, sampling_rate=16000, return_tensors="pt"):
            import torch

            return types.SimpleNamespace(input_features=torch.zeros((1, 80, 3000), dtype=torch.float32))

        def batch_decode(self, seqs, skip_special_tokens=True):
            return [f"hyp_{i}" for i in range(len(seqs))]

    class _MockModel:
        def generate(self, features, **kwargs):
            import torch

            recorded_calls.append(dict(kwargs))
            if "diversity_penalty" in kwargs or "num_beam_groups" in kwargs:
                # Raise on first attempt when diversity kwargs passed to verify fallback
                raise TypeError("unexpected keyword argument 'diversity_penalty'")
            n_ret = int(kwargs.get("num_return_sequences", 1))
            seqs = torch.ones((n_ret, 4), dtype=torch.long)
            if kwargs.get("return_dict_in_generate"):
                return types.SimpleNamespace(sequences=seqs, sequences_scores=torch.zeros((n_ret,), dtype=torch.float32))
            return seqs

    engine = WhisperASREngine.__new__(WhisperASREngine)
    engine.device = "cpu"
    import torch

    engine.torch_dtype = torch.float32
    engine.language = "vi"
    engine.task = "transcribe"
    engine.initial_prompt = ""
    engine.enable_itn = False
    engine.repetition_penalty = 1.0
    engine.trim_silence = False
    engine.peak_normalize = False
    engine.pre_roll_pad_ms = 0.0
    engine.post_roll_pad_ms = 0.0
    engine.processor = _MockProcessor()
    engine.model = _MockModel()

    res = engine.transcribe(wav_path, num_beams_nbest=3, diverse_nbest=True)
    assert res.text == "Hyp_0"
    assert len(res.nbest) == 3
    # Call 0: strict greedy 1-best unchanged
    assert "num_beams" not in recorded_calls[0]
    assert "num_beam_groups" not in recorded_calls[0]
    assert "diversity_penalty" not in recorded_calls[0]
    # Call 1: diverse n-best attempted first
    assert recorded_calls[1]["num_beam_groups"] == 3
    assert recorded_calls[1]["diversity_penalty"] == 0.5
    # Call 2: fell back cleanly to standard beam search
    assert recorded_calls[2]["num_beams"] == 3
    assert "diversity_penalty" not in recorded_calls[2]


def test_build_system_prompt_includes_trackb_hardening_rules():
    sample = load_split("train", "data")[0]
    prompt = build_system_prompt(sample, prompt_style="qwen35")
    assert "loyalty.get_points({})" in prompt
    assert "Gợi ý đồng âm ASR" in prompt
    assert "sao chép chính xác địa chỉ đầy đủ" in prompt.lower()


def test_react_loop_loyalty_empty_program_appends_recovery_hint():
    catalog = load_catalog("train", "data")
    sample = Sample(
        id="mock_loyalty_01",
        audio="dummy.wav",
        audio_text="Kiểm tra điểm thành viên Sen Air giúp tôi",
        context={
            "current_time": "2026-11-14T12:00:00+07:00",
            "timezone": "Asia/Ho_Chi_Minh",
            "location": {},
            "user_profile": {},
            "installed_apps": {"loyalty": {"tools": ["loyalty.get_points", "loyalty.redeem"]}},
        },
        history=[],
    )

    class _MockLoyaltyEnv:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def validate(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"ok": True}

        def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            self.calls.append({"name": tool_name, "arguments": dict(arguments)})
            if tool_name == "loyalty.get_points" and arguments.get("program"):
                return {"points": []}
            return {"points": [{"program": "SenAir Rewards", "balance": 1200}]}

        def get_tool_calls(self) -> list[dict[str, Any]]:
            return self.calls

        def get_state_hash(self) -> str:
            return "hash_ok"

    fake_llm = SequenceFakeBackend(
        [
            "<tool_call>\n<function=loyalty.get_points>\n<parameter=program>\nSen Air\n</parameter>\n</function>\n</tool_call>",
            "<tool_call>\n<function=loyalty.get_points>\n</function>\n</tool_call>",
            "Bạn đang có 1200 điểm SenAir Rewards.",
        ]
    )
    env = _MockLoyaltyEnv()
    res = run_react_loop(sample, env, catalog, fake_llm)

    assert len(res.tool_calls) == 2
    assert res.tool_calls[1] == {"name": "loyalty.get_points", "arguments": {}}
    # Verify the second LLM call saw the explicit empty-program recovery hint in the tool message
    second_turn_msgs = fake_llm.calls_log[1]["messages"]
    tool_msgs = [m for m in second_turn_msgs if m.get("role") == "tool" and m.get("name") == "loyalty.get_points"]
    assert tool_msgs and "loyalty.get_points({})" in tool_msgs[-1]["content"]


def test_qwen35_grammar_allows_angle_brackets_inside_think():
    from src.agent.llm_client import build_qwen35_tool_grammar

    tools = [
        {
            "type": "function",
            "function": {
                "name": "reminders.create",
                "parameters": {"type": "object", "properties": {"content": {"type": "string"}}},
            },
        }
    ]
    gram = build_qwen35_tool_grammar(tools, allow_thinking=True)
    assert "think-body" in gram
    assert '[^<]* "</think>"' not in gram


def test_run_react_batch_matches_sequential_run_react_loop():
    from src.agent.react_loop import BatchReactItem, run_react_batch

    catalog = load_catalog("train", "data")
    samples = load_split("train", "data")[:2]

    class _MockSimpleEnv:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def validate(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"ok": True}

        def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            self.calls.append({"name": tool_name, "arguments": dict(arguments)})
            return {"ok": True}

        def get_tool_calls(self) -> list[dict[str, Any]]:
            return self.calls

        def get_state_hash(self) -> str:
            return f"hash_{len(self.calls)}"

    class _BatchFakeBackend:
        def __init__(self, script_by_user_substr: dict[str, list[str]]) -> None:
            self._scripts = {k: list(v) for k, v in script_by_user_substr.items()}
            self.batch_calls: list[int] = []

        def _next_for(self, messages: list[dict[str, Any]]) -> str:
            text = json.dumps(messages, ensure_ascii=False)
            for key, seq in self._scripts.items():
                if key in text:
                    return seq.pop(0) if seq else "Hoàn tất."
            return "Hoàn tất."

        def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, *, force_tool_schema: bool = False) -> str:
            return self._next_for(messages)

        def generate_batch(self, requests: list[tuple[list[dict[str, Any]], list[dict[str, Any]] | None, bool]]) -> list[str]:
            self.batch_calls.append(len(requests))
            return [self._next_for(msgs) for msgs, _, _ in requests]

    s0 = Sample(
        id="batch_s0",
        audio="a0.wav",
        audio_text="OOS_MARKER Hãy đặt vé xem phim",
        context=samples[0].context,
        history=[],
    )
    s1 = Sample(
        id="batch_s1",
        audio="a1.wav",
        audio_text="ACTION_MARKER Kiểm tra điểm Sen Air",
        context={
            "current_time": "2026-11-14T12:00:00+07:00",
            "timezone": "Asia/Ho_Chi_Minh",
            "location": {},
            "user_profile": {},
            "installed_apps": {"loyalty": {"tools": ["loyalty.get_points"]}},
        },
        history=[],
    )

    scripts = {
        "OOS_MARKER": ["Xin lỗi, tôi không hỗ trợ đặt vé xem phim."],
        "ACTION_MARKER": [
            "<tool_call>\n<function=loyalty.get_points>\n</function>\n</tool_call>",
            "Bạn có 500 điểm.",
        ],
    }

    seq_envs = [_MockSimpleEnv(), _MockSimpleEnv()]
    seq_res = [
        run_react_loop(s0, seq_envs[0], catalog, _BatchFakeBackend(scripts)),
        run_react_loop(s1, seq_envs[1], catalog, _BatchFakeBackend(scripts)),
    ]

    bat_envs = [_MockSimpleEnv(), _MockSimpleEnv()]
    bat_llm = _BatchFakeBackend(scripts)
    bat_res = run_react_batch(
        [
            BatchReactItem(sample=s0, env=bat_envs[0], catalog=catalog),
            BatchReactItem(sample=s1, env=bat_envs[1], catalog=catalog),
        ],
        bat_llm,
    )

    assert bat_llm.batch_calls == [2, 1]
    for r_seq, r_bat in zip(seq_res, bat_res):
        assert r_bat.id == r_seq.id
        assert r_bat.tool_calls == r_seq.tool_calls
        assert r_bat.state_hash == r_seq.state_hash
        assert r_bat.steps == r_seq.steps
        assert r_bat.repairs == r_seq.repairs


def test_vllm_backend_max_num_seqs_and_generate_batch(tmp_path, monkeypatch):
    import transformers

    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForCausalLM"]}), encoding="utf-8")

    init_captured: dict[str, Any] = {}
    batch_inputs: list[int] = []

    class _FakeLLM:
        def __init__(self, model: str, **kwargs: Any):
            init_captured["model"] = model
            init_captured.update(kwargs)

        def generate(self, prompts, sampling_params, lora_request=None, use_tqdm=False):
            batch_inputs.append(len(prompts))
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=f"ok_{i}")]) for i in range(len(prompts))]

    class _FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _DummyTok:
        pad_token_id = 0
        eos_token_id = 1

        def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=True, enable_thinking=False):
            return "PROMPT " + " ".join(str(m.get("content", "")) for m in messages)

        def __call__(self, text: str, return_tensors: str | None = None, add_special_tokens: bool = False):
            return {"input_ids": list(range(1, len(text.split()) + 1))}

    fake_vllm = types.SimpleNamespace(LLM=_FakeLLM, SamplingParams=_FakeSamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **kw: _DummyTok()))

    backend = VLLMBackend(model_id=str(model_dir), max_model_len=4096, max_num_seqs=16, gpu_memory_utilization=0.65)
    assert init_captured["max_num_seqs"] == 16
    assert init_captured["gpu_memory_utilization"] == 0.65

    outs = backend.generate_batch([
        ([{"role": "user", "content": "one"}], None, False),
        ([{"role": "user", "content": "two"}], None, False),
    ])
    assert batch_inputs == [2]
    assert outs == ["ok_0", "ok_1"]


def test_itn_normalizes_spoken_emails():
    from src.asr.itn import inverse_text_normalize

    assert (
        inverse_text_normalize("gửi final a còng example chấm test nha")
        == "gửi final@example.test nha"
    )
    assert (
        inverse_text_normalize("liên hệ lan chấm nguyen a còng congty chấm com chấm vn")
        == "liên hệ lan.nguyen@congty.com.vn"
    )


def test_init_known_ids_reads_transcribed_history_texts():
    from src.agent.react_loop import _init_known_ids
    from src.core.schemas import Turn

    catalog = load_catalog("train", "data")
    sample = Sample(
        id="hist_phone",
        audio="curr.wav",
        context={"installed_apps": {"contacts": {"tools": ["contacts.search", "contacts.update"]}}},
        history=[Turn(role="user", audio="u0.wav", text="")],
    )
    known = _init_known_ids(sample, catalog, history_texts={0: "Đổi sang số 0989165158 nhé"})
    assert "0989165158" in known["phones"]


def test_boundary_verifier_unseen_prereq_and_unfiltered_search_fallback():
    from src.agent.boundary_verifier import BoundaryVerifier
    from src.core.schemas import ASK_MISSING_INFO, ToolCall

    pub_cat = load_catalog("public_test", "data")
    bv = BoundaryVerifier(catalog=pub_cat)
    available = set(pub_cat.keys())

    # 1. Missing list_sessions prereq before asking parking.end_session session_id
    req1 = bv.find_missing_prereq_read(
        ToolCall(name=ASK_MISSING_INFO, arguments={"tool_name": "parking.end_session", "params": ["session_id"]}),
        seen_reads=set(),
        available_tools=available,
    )
    assert req1 is not None and req1.name == "parking.list_sessions"

    # 2. After filtered event_tickets.search returned empty (known_ids['event_tickets'] empty), fall back to {}
    req2 = bv.find_missing_prereq_read(
        ToolCall(name=ASK_MISSING_INFO, arguments={"tool_name": "event_tickets.book", "params": ["event_id"]}),
        seen_reads={("event_tickets.search", '{"query":"Đèn Khoa học"}')},
        available_tools=available,
        known_ids_by_app={"event_tickets": set()},
    )
    assert req2 is not None and req2.name == "event_tickets.search" and req2.arguments == {}

    # 3. Brand-new synthetic unseen app (private_test simulation) resolved dynamically via x-ref + kind=read + required=[]
    synth_cat = {
        "drone_delivery.list_orders": {
            "kind": "read",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        "drone_delivery.list_drones": {
            "kind": "read",
            "parameters": {"type": "object", "properties": {"zone": {"type": "string"}}, "required": []},
        },
        "drone_delivery.dispatch": {
            "kind": "write",
            "parameters": {
                "type": "object",
                "properties": {
                    "drone_id": {"type": "string", "x-ref": "drone_delivery.drones"},
                    "ticket_id": {"type": "string", "x-ref": "drone_delivery.tickets"},
                },
                "required": ["drone_id"],
            },
        },
    }
    synth_bv = BoundaryVerifier(catalog=synth_cat)
    req_synth = synth_bv.find_missing_prereq_read(
        ToolCall(name=ASK_MISSING_INFO, arguments={"tool_name": "drone_delivery.dispatch", "params": ["drone_id"]}),
        seen_reads=set(),
        available_tools=set(synth_cat.keys()),
    )
    assert req_synth is not None and req_synth.name == "drone_delivery.list_drones" and req_synth.arguments == {}

    # And after filtered list_drones returned empty, fall back to unfiltered {}
    req_synth_fb = synth_bv.find_missing_prereq_read(
        ToolCall(name=ASK_MISSING_INFO, arguments={"tool_name": "drone_delivery.dispatch", "params": ["drone_id"]}),
        seen_reads={("drone_delivery.list_drones", '{"zone":"Q1"}')},
        available_tools=set(synth_cat.keys()),
        known_ids_by_app={"drone_delivery": set()},
    )
    assert req_synth_fb is not None and req_synth_fb.name == "drone_delivery.list_drones" and req_synth_fb.arguments == {}

    from src.agent.react_loop import _is_verifiable_entity_id_param
    assert _is_verifiable_entity_id_param("drone_delivery.dispatch", "ticket_id", synth_cat) is True
    assert _is_verifiable_entity_id_param("drone_delivery.dispatch", "drone_id", synth_cat) is True
    assert _is_verifiable_entity_id_param("drone_delivery.dispatch", "contact_id", synth_cat) is False


def test_react_loop_anti_false_ask_verification():
    catalog = load_catalog("train", "data")

    class _MockEnv:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def validate(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"ok": True}

        def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            self.calls.append({"name": tool_name, "arguments": dict(arguments)})
            return {"ok": True}

        def get_tool_calls(self) -> list[dict[str, Any]]:
            return self.calls

        def get_state_hash(self) -> str:
            return f"h_{len(self.calls)}"

    class _VerifyBackend:
        def __init__(self, responses: list[str]) -> None:
            self.responses = list(responses)
            self.prompts: list[list[dict[str, Any]]] = []

        def generate(self, messages: list[dict[str, Any]], tools: Any = None, *, force_tool_schema: bool = False) -> str:
            self.prompts.append(messages)
            return self.responses.pop(0)

    sample = Sample(
        id="false_ask_0",
        audio="a.wav",
        audio_text="Nhắc cho mình đổ xăng lúc 5 giờ chiều mai nhé.",
        context={"installed_apps": {"reminders": {"tools": ["reminders.list", "reminders.create"]}}},
        history=[],
    )
    ask_xml = '<tool_call>\n<function=sys.ask_missing_info>\n<parameter=tool_name>\nreminders.create\n</parameter>\n<parameter=params>\n["content"]\n</parameter>\n</function>\n</tool_call>'
    act_xml = '<tool_call>\n<function=reminders.create>\n<parameter=content>\nĐổ xăng\n</parameter>\n<parameter=time>\n2026-11-17T17:00:00+07:00\n</parameter>\n</function>\n</tool_call>'
    res = run_react_loop(sample, _MockEnv(), catalog, _VerifyBackend([ask_xml, act_xml, "Xong."]))
    assert [c["name"] for c in res.tool_calls] == ["reminders.create"]

    # If model confirms sys.ask_missing_info on second check, it terminates cleanly on ask
    res2 = run_react_loop(sample, _MockEnv(), catalog, _VerifyBackend([ask_xml, ask_xml]))
    assert [c["name"] for c in res2.tool_calls] == ["sys.ask_missing_info"]


def test_build_audio_capped_chunks_preserves_order_and_splits_heavy_history():
    import importlib.util
    from src.core.schemas import Turn

    spec = importlib.util.spec_from_file_location("sub_agent", "submission/qwen35_4b_sft_v2_e2e/agent.py")
    sub_agent = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(sub_agent)

    def _mk(sid: str, n_hist_user: int) -> Sample:
        hist = [Turn(role="user", audio=f"{sid}_h{i}.wav") for i in range(n_hist_user)]
        return Sample(id=sid, audio=f"{sid}.wav", context={}, history=hist)

    samples = [_mk("s0", 0), _mk("s1", 0), _mk("s2", 0), _mk("s3", 0), _mk("s4", 6), _mk("s5", 6), _mk("s6", 0)]
    chunks = sub_agent.build_audio_capped_chunks(samples, max_batch_size=4, max_chunk_audios=8)
    assert [s.id for ch in chunks for s in ch] == [s.id for s in samples]
    assert [[s.id for s in ch] for ch in chunks] == [["s0", "s1", "s2", "s3"], ["s4"], ["s5", "s6"]]

