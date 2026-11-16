from __future__ import annotations

import json
from typing import Any

from src.agent.llm_client import TransformersBackend, _fit_messages_to_token_budget


class _DummyWordTokenizer:
    """Lightweight whitespace tokenizer for deterministic zero-compute unit tests."""

    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        parts: list[str] = []
        if tools:
            parts.append("TOOLS_SCHEMA_HEADER " + json.dumps(tools, ensure_ascii=False))
        for m in messages:
            parts.append(f"<|im_start|>{m['role']} {m.get('content', '')} <|im_end|>")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant")
        text = "\n".join(parts)
        if tokenize:
            return self(text)["input_ids"]
        return text

    def __call__(self, text: str, return_tensors: str | None = None, add_special_tokens: bool = False) -> dict[str, Any]:
        import torch

        tokens = text.split()
        ids = [idx + 2 for idx, _ in enumerate(tokens)]
        self._last_tokens = tokens
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            }
        return {"input_ids": ids}

    def decode(self, token_ids: Any, skip_special_tokens: bool = False) -> str:
        return "Đã thực hiện xong.<|im_end|>"


def test_fit_messages_preserves_system_prompt_tools_and_latest_user_turn():
    tok = _DummyWordTokenizer()
    tools = [{"type": "function", "function": {"name": "alarm.create", "parameters": {"type": "object"}}}]
    messages = [
        {"role": "system", "content": "SYSTEM_ANCHOR_LIFEMATE quy tắc bắt buộc đọc trước khi ghi"},
        {"role": "user", "content": "OLD_TURN_1 " + ("lịch_sử_cũ " * 80)},
        {"role": "assistant", "content": "OLD_REPLY_1 " + ("trả_lời_cũ " * 80)},
        {"role": "user", "content": "OLD_TURN_2 " + ("lịch_sử_hai " * 80)},
        {"role": "assistant", "content": "OLD_REPLY_2 " + ("trả_lời_hai " * 80)},
        {"role": "user", "content": "CURRENT_USER_REQUEST Đặt báo thức 6 giờ sáng mai"},
    ]

    fitted_msgs, prompt = _fit_messages_to_token_budget(
        tok,
        messages,
        tools,
        max_input_len=90,
        enable_thinking=False,
    )
    encoded = tok(prompt, return_tensors="pt")
    assert encoded["input_ids"].shape[1] <= 90
    assert fitted_msgs[0]["role"] == "system"
    assert "SYSTEM_ANCHOR_LIFEMATE" in prompt
    assert "TOOLS_SCHEMA_HEADER" in prompt
    assert "alarm.create" in prompt
    assert "CURRENT_USER_REQUEST" in prompt
    assert "OLD_TURN_1" not in prompt


def test_transformers_backend_generate_never_left_slices_system_prompt():
    import torch

    tok = _DummyWordTokenizer()
    captured_first_token_text: list[str] = []

    class _DummyModel:
        device = torch.device("cpu")

        def generate(self, **kwargs: Any) -> Any:
            input_ids = kwargs["input_ids"]
            assert input_ids.shape[1] <= 80
            # First token of prompt must correspond to index 2 (the start of TOOLS_SCHEMA_HEADER), never left-sliced!
            captured_first_token_text.append(tok._last_tokens[0])
            assert int(input_ids[0, 0].item()) == 2
            gen = torch.tensor([[999]], dtype=torch.long)
            return torch.cat([input_ids, gen], dim=1)

    backend = TransformersBackend.__new__(TransformersBackend)
    backend.tokenizer = tok
    backend.model = _DummyModel()
    backend.max_model_len = 100
    backend.max_new_tokens = 20
    backend.enable_thinking = False
    backend.repetition_penalty = 1.1

    tools = [{"type": "function", "function": {"name": "weather.get_forecast", "parameters": {"type": "object"}}}]
    messages = [
        {"role": "system", "content": "SYSTEM_ANCHOR"},
        {"role": "user", "content": "OLD_HISTORY " + ("x " * 120)},
        {"role": "assistant", "content": "OLD_ANS " + ("y " * 120)},
        {"role": "user", "content": "LATEST_QUESTION Thời tiết ngày mai thế nào"},
    ]

    out = backend.generate(messages, tools)
    assert out == "Đã thực hiện xong."
    assert captured_first_token_text == ["TOOLS_SCHEMA_HEADER"]


def test_resolve_tokenizer_path_and_vl_detection(tmp_path):
    from src.agent.llm_client import _is_vision_language_model, _resolve_tokenizer_path

    adapter_no_tok = tmp_path / "adapter_only"
    adapter_no_tok.mkdir()
    assert _resolve_tokenizer_path("empero-ai/Qwen3.8-4B-Distill", str(adapter_no_tok)) == "empero-ai/Qwen3.8-4B-Distill"

    adapter_with_tok = tmp_path / "adapter_tok"
    adapter_with_tok.mkdir()
    (adapter_with_tok / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert _resolve_tokenizer_path("empero-ai/Qwen3.8-4B-Distill", str(adapter_with_tok)) == str(adapter_with_tok)

    vl_dir = tmp_path / "vl_model"
    vl_dir.mkdir()
    (vl_dir / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"], "vision_config": {"depth": 24}}),
        encoding="utf-8",
    )
    assert _is_vision_language_model(str(vl_dir)) is True

    text_dir = tmp_path / "text_model"
    text_dir.mkdir()
    (text_dir / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text"}),
        encoding="utf-8",
    )
    assert _is_vision_language_model(str(text_dir)) is False


def test_vllm_backend_passes_language_model_only_for_vl_models(tmp_path, monkeypatch):
    import sys
    import types
    import transformers
    from src.agent.llm_client import VLLMBackend

    vl_dir = tmp_path / "qwen38_vl"
    vl_dir.mkdir()
    (vl_dir / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"], "vision_config": {}}),
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    class _FakeLLM:
        def __init__(self, model, dtype="bfloat16", max_model_len=16384, gpu_memory_utilization=0.85,
                     enable_lora=False, trust_remote_code=True, language_model_only=False):
            captured["model"] = model
            captured["language_model_only"] = language_model_only
            captured["trust_remote_code"] = trust_remote_code

    class _FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_vllm = types.SimpleNamespace(LLM=_FakeLLM, SamplingParams=_FakeSamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **kw: _DummyWordTokenizer()))

    VLLMBackend(model_id=str(vl_dir), max_model_len=4096)
    assert captured["model"] == str(vl_dir)
    assert captured["language_model_only"] is True
    assert captured["trust_remote_code"] is True


def test_vllm_guided_decoding_on_repair_only(tmp_path, monkeypatch):
    import sys
    import types
    import transformers
    from src.agent.llm_client import VLLMBackend, build_qwen35_tool_grammar, build_tool_call_json_schema

    tools = [
        {
            "type": "function",
            "function": {
                "name": "alarm.create",
                "parameters": {
                    "type": "object",
                    "required": ["time"],
                    "properties": {"time": {"type": "string"}, "label": {"type": "string"}},
                },
            },
        }
    ]
    schema = build_tool_call_json_schema(tools)
    grammar = build_qwen35_tool_grammar(tools)
    assert "alarm.create" in json.dumps(schema)
    assert "alarm.create" in grammar and "time" in grammar

    tok_dir = tmp_path / "tok"
    tok_dir.mkdir()
    (tok_dir / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForCausalLM"]}), encoding="utf-8")

    gen_sampling_params: list[Any] = []

    class _FakeLLM:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, sampling_params, lora_request=None, use_tqdm=False):
            gen_sampling_params.append(sampling_params)
            out_obj = types.SimpleNamespace(text="<tool_call>\n<function=alarm.create>\n<parameter=time>\n07:00\n</parameter>\n</function>\n</tool_call>")
            return [types.SimpleNamespace(outputs=[out_obj])]

    class _FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeStructuredOutputsParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_so_mod = types.SimpleNamespace(StructuredOutputsParams=_FakeStructuredOutputsParams)
    fake_vllm = types.SimpleNamespace(
        LLM=_FakeLLM,
        SamplingParams=_FakeSamplingParams,
        sampling_params=fake_so_mod,
    )
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", fake_so_mod)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **kw: _DummyWordTokenizer()))

    backend = VLLMBackend(model_id=str(tok_dir), max_model_len=4096, guided_decoding=True)
    msgs = [{"role": "user", "content": "Đặt báo thức"}]
    backend.generate(msgs, tools, force_tool_schema=False)
    backend.generate(msgs, tools, force_tool_schema=True)

    assert "structured_outputs" not in gen_sampling_params[0].kwargs
    assert "structured_outputs" in gen_sampling_params[1].kwargs
    assert "alarm.create" in gen_sampling_params[1].kwargs["structured_outputs"].kwargs["grammar"]


def test_vllm_passes_max_lora_rank_when_llm_accepts_kwargs(tmp_path, monkeypatch):
    import sys
    import types
    import transformers
    from src.agent.llm_client import VLLMBackend

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForCausalLM"]}), encoding="utf-8")

    adapter_dir = tmp_path / "adapter_r32"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 32}), encoding="utf-8")

    captured: dict[str, Any] = {}

    class _KwargsLLM:
        def __init__(self, model: str, **kwargs: Any):
            captured["model"] = model
            captured.update(kwargs)

    class _FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeLoRARequest:
        def __init__(self, *args):
            self.args = args

    fake_vllm = types.SimpleNamespace(LLM=_KwargsLLM, SamplingParams=_FakeSamplingParams)
    fake_lora_req_mod = types.SimpleNamespace(LoRARequest=_FakeLoRARequest)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", types.SimpleNamespace(request=fake_lora_req_mod))
    monkeypatch.setitem(sys.modules, "vllm.lora.request", fake_lora_req_mod)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **kw: _DummyWordTokenizer()))

    VLLMBackend(model_id=str(model_dir), adapter_path=str(adapter_dir), max_model_len=4096)
    assert captured["enable_lora"] is True
    assert captured["max_lora_rank"] == 32
    assert captured["seed"] == 42

