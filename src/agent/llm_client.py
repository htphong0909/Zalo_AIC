"""LLM backend protocol, deterministic fake backend, Transformers (with PEFT/LoRA), and vLLM backend."""
from __future__ import annotations

from pathlib import Path
import inspect
import json
import os
import re
from typing import TYPE_CHECKING, Any, Protocol

from src.agent.minicpm5_parser import normalize_minicpm5_output

if TYPE_CHECKING:
    from src.core.config import LLMConfig


def build_tool_call_json_schema(tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a Draft-7 JSON schema covering all allowed tool names and parameter definitions."""
    variants: list[dict[str, Any]] = []
    for t in tools or []:
        fn = t.get("function") or t
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        params = fn.get("parameters") or {"type": "object", "properties": {}}
        variants.append(
            {
                "type": "object",
                "required": ["name", "arguments"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "const": name},
                    "arguments": params,
                },
            }
        )
    if not variants:
        return {"type": "object"}
    return {
        "type": "array",
        "minItems": 1,
        "items": {"anyOf": variants},
    }


def _gb_escape(literal: str) -> str:
    return json.dumps(literal, ensure_ascii=False)


def build_qwen35_tool_grammar(
    tools: list[dict[str, Any]],
    *,
    allow_thinking: bool = False,
) -> str:
    """Build an EBNF/GBNF grammar enforcing Qwen3.5 XML <tool_call><function=...> syntax and allowed names/params."""
    fn_rules: list[str] = []
    fn_refs: list[str] = []
    for idx, t in enumerate(tools or []):
        fn = t.get("function") or t
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        props = ((fn.get("parameters") or {}).get("properties") or {})
        rule_id = f"fn_{idx}"
        fn_refs.append(rule_id)
        if props:
            param_names = " | ".join(_gb_escape(str(p)) for p in props.keys())
            p_rule = f"param_{idx}"
            fn_rules.append(
                f'{p_rule} ::= "<parameter=" ({param_names}) ">\\n" value "\\n</parameter>\\n"\n'
                f'{rule_id} ::= "<function=" {_gb_escape(name)} ">\\n" ({p_rule})* "</function>"'
            )
        else:
            fn_rules.append(f'{rule_id} ::= "<function=" {_gb_escape(name)} ">\\n</function>"')

    root_prefix = "( think-block )? " if allow_thinking else ""
    think_rule = (
        [
            'think-block ::= ( "<think>" )? think-body "</think>" ws',
            'think-body ::= ( [^<] | "<" [^/] | "</" [^t] | "</t" [^h] | "</th" [^i] | "</thi" [^n] | "</thin" [^k] | "</think" [^>] )*',
        ]
        if allow_thinking
        else []
    )

    if not fn_refs:
        return "\n".join(
            [
                f"root ::= {root_prefix}call ws",
                'call ::= "<tool_call>\\n" "<function=" [^>]+ ">\\n</function>\\n</tool_call>"',
                'ws ::= [ \\n\\t]*',
                *think_rule,
            ]
        )

    call_expr = " | ".join(fn_refs)
    lines = [
        f'root ::= {root_prefix}call ( "\\n" call )* ws',
        'call ::= "<tool_call>\\n" (' + call_expr + ') "\\n</tool_call>"',
        'value ::= [^<]+',
        'ws ::= [ \\n\\t]*',
        *think_rule,
        *fn_rules,
    ]
    return "\n".join(lines)


class ChatBackend(Protocol):
    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        force_tool_schema: bool = False,
    ) -> str: ...


class SequenceFakeBackend:
    """Deterministic fake backend that returns scripted responses in order."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses or [])
        self.calls_log: list[dict[str, Any]] = []

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        self.calls_log.append({"messages": list(messages), "tools": list(tools)})
        if not self._responses:
            return ""
        return self._responses.pop(0)


def _format_chat_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    enable_thinking: bool = False,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tools=tools or None,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    except Exception:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _strip_special_markers(text: str) -> str:
    cleaned = re.sub(r"(<\|im_end\|>|<\|endoftext\|>)+$", "", text).strip()
    cleaned = re.sub(r"^<\|im_start\|>\s*", "", cleaned)
    return cleaned


def _encode_prompt_ids(tokenizer: Any, prompt: str) -> list[int]:
    encoded = tokenizer(prompt, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else getattr(encoded, "input_ids", encoded)
    if hasattr(ids, "shape"):
        flat = ids.reshape(-1)
        return flat.tolist()
    if ids and isinstance(ids[0], list):
        return [int(x) for x in ids[0]]
    return [int(x) for x in ids]


def _count_prompt_tokens(tokenizer: Any, prompt: str) -> int:
    return len(_encode_prompt_ids(tokenizer, prompt))


def _compact_tool_msg(msg: dict[str, Any], max_chars: int = 300) -> dict[str, Any]:
    if msg.get("role") != "tool":
        return msg
    content = str(msg.get("content") or "")
    if len(content) <= max_chars:
        return msg
    compacted = dict(msg)
    compacted["content"] = content[: max_chars - 15] + '..."truncated"}'
    return compacted


def _drop_oldest_turn_unit(middle: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not middle:
        return []
    first_role = middle[0].get("role")
    idx = 1
    if first_role == "user":
        while idx < len(middle) and middle[idx].get("role") in ("assistant", "tool"):
            idx += 1
        return middle[idx:]
    if first_role == "assistant":
        while idx < len(middle) and middle[idx].get("role") == "tool":
            idx += 1
        return middle[idx:]
    return middle[1:]


def _fit_messages_with_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_input_len: int,
    *,
    enable_thinking: bool = False,
) -> tuple[list[dict[str, Any]], str, list[int]]:
    """Fit chat messages into max_input_len while returning rendered prompt and token ids."""
    prompt = _format_chat_prompt(tokenizer, messages, tools, enable_thinking=enable_thinking)
    prompt_ids = _encode_prompt_ids(tokenizer, prompt)
    if len(prompt_ids) <= max_input_len or len(messages) <= 2:
        return list(messages), prompt, prompt_ids

    has_sys = bool(messages and messages[0].get("role") == "system")
    head = [messages[0]] if has_sys else []
    body = list(messages[1:] if has_sys else messages)

    user_positions = [i for i, m in enumerate(body) if m.get("role") == "user"]
    if user_positions:
        last_u = user_positions[-1]
        middle = body[:last_u]
        tail = body[last_u:]
    else:
        middle = []
        tail = body

    # 1. Compact older tool outputs in middle first
    if middle:
        middle = [_compact_tool_msg(m, max_chars=300) for m in middle]
        cand = [*head, *middle, *tail]
        prompt = _format_chat_prompt(tokenizer, cand, tools, enable_thinking=enable_thinking)
        prompt_ids = _encode_prompt_ids(tokenizer, prompt)
        if len(prompt_ids) <= max_input_len:
            return cand, prompt, prompt_ids

    # 2. Drop oldest history turn units from middle until within token budget
    while middle:
        middle = _drop_oldest_turn_unit(middle)
        cand = [*head, *middle, *tail]
        prompt = _format_chat_prompt(tokenizer, cand, tools, enable_thinking=enable_thinking)
        prompt_ids = _encode_prompt_ids(tokenizer, prompt)
        if len(prompt_ids) <= max_input_len:
            return cand, prompt, prompt_ids

    # 3. If current ReAct turn itself is huge, compact older tool outputs inside tail
    if len(tail) > 2:
        compacted_tail = [tail[0], *[_compact_tool_msg(m, max_chars=300) for m in tail[1:-1]], tail[-1]]
        cand = [*head, *compacted_tail]
        prompt = _format_chat_prompt(tokenizer, cand, tools, enable_thinking=enable_thinking)
        prompt_ids = _encode_prompt_ids(tokenizer, prompt)
        if len(prompt_ids) <= max_input_len:
            return cand, prompt, prompt_ids

        sub_steps = compacted_tail[1:]
        while len(sub_steps) > 2:
            sub_steps = _drop_oldest_turn_unit(sub_steps)
            cand = [*head, compacted_tail[0], *sub_steps]
            prompt = _format_chat_prompt(tokenizer, cand, tools, enable_thinking=enable_thinking)
            prompt_ids = _encode_prompt_ids(tokenizer, prompt)
            if len(prompt_ids) <= max_input_len:
                return cand, prompt, prompt_ids
        return [*head, compacted_tail[0], *sub_steps], prompt, prompt_ids

    cand = [*head, *tail]
    prompt = _format_chat_prompt(tokenizer, cand, tools, enable_thinking=enable_thinking)
    prompt_ids = _encode_prompt_ids(tokenizer, prompt)
    return cand, prompt, prompt_ids


def _fit_messages_to_token_budget(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_input_len: int,
    *,
    enable_thinking: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    """Fit chat messages into max_input_len while preserving system prompt, tools, and latest turn."""
    cand, prompt, _ = _fit_messages_with_ids(
        tokenizer, messages, tools, max_input_len, enable_thinking=enable_thinking
    )
    return cand, prompt


def _collect_stop_token_ids(tokenizer: Any) -> list[int]:
    """EOS ids that end an assistant turn: tokenizer EOS plus ChatML ``<|im_end|>`` when present.

    MiniCPM5's tokenizer EOS is ``</s>`` (id 1) but its chat template closes every assistant turn with
    ``<|im_end|>`` (id 130073); the model's generation_config lists both. Stopping only on ``</s>`` makes
    the model run past ``<|im_end|>`` and hallucinate extra turns / tool calls.
    """
    ids: list[int] = []
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        ids.append(eos)
    elif isinstance(eos, (list, tuple)):
        ids.extend(int(x) for x in eos)
    try:
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    except Exception:
        im_end = None
    unk = getattr(tokenizer, "unk_token_id", None)
    if isinstance(im_end, int) and im_end >= 0 and im_end != unk and im_end not in ids:
        ids.append(im_end)
    return ids


def _resolve_tokenizer_path(model_id: str, adapter_path: str | None) -> str:
    if adapter_path:
        p_ad = Path(adapter_path)
        if (p_ad / "tokenizer.json").exists() or (p_ad / "tokenizer_config.json").exists():
            return adapter_path
    return model_id


def _requires_trust_remote_code(model_id: str) -> bool:
    low = model_id.lower()
    return not ("minicpm" in low or "llama" in low)


def _is_offline_or_local(path_or_id: str) -> bool:
    return (
        Path(path_or_id).exists()
        or os.environ.get("HF_HUB_OFFLINE") == "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    )


def _is_vision_language_model(model_id: str, *, trust_remote: bool = True) -> bool:
    """Detect VL checkpoints (e.g. Qwen3.5/Qwen3.8 ForConditionalGeneration) that ship unused vision weights."""
    local_cfg = Path(model_id) / "config.json"
    if local_cfg.exists():
        try:
            raw = json.loads(local_cfg.read_text(encoding="utf-8"))
            archs = raw.get("architectures") or []
            return bool(raw.get("vision_config")) or any("ConditionalGeneration" in str(a) for a in archs)
        except Exception:
            return False
    try:
        from transformers import AutoConfig

        load_kw: dict[str, Any] = {"trust_remote_code": trust_remote}
        if _is_offline_or_local(model_id):
            load_kw["local_files_only"] = True
        cfg = AutoConfig.from_pretrained(model_id, **load_kw)
        archs = getattr(cfg, "architectures", None) or []
        return hasattr(cfg, "vision_config") or any("ConditionalGeneration" in str(a) for a in archs)
    except Exception:
        return False


class TransformersBackend:
    """Greedy causal LM backend using Hugging Face Transformers with optional PEFT/LoRA adapter."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.5-4B",
        *,
        adapter_path: str | None = None,
        max_model_len: int = 16384,
        max_new_tokens: int = 384,
        enable_thinking: bool = False,
        dtype: str = "bfloat16",
        repetition_penalty: float = 1.0,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        if torch_dtype == torch.bfloat16 and torch.cuda.is_available():
            try:
                bf16_ok = torch.cuda.is_bf16_supported(including_emulation=False)
            except TypeError:
                bf16_ok = torch.cuda.get_device_capability()[0] >= 8
            if not bf16_ok:
                torch_dtype = torch.float16
        if enable_thinking is None:
            enable_thinking = "minicpm" in model_id.lower() or bool(
                adapter_path and "minicpm" in adapter_path.lower()
            )
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.max_model_len = max_model_len
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = bool(enable_thinking)
        self.repetition_penalty = float(repetition_penalty)
        tok_path = _resolve_tokenizer_path(model_id, adapter_path)
        trust_remote = _requires_trust_remote_code(model_id)
        local_kw = {"local_files_only": True} if _is_offline_or_local(model_id) else {}
        tok_local_kw = {"local_files_only": True} if _is_offline_or_local(tok_path) else {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_path,
            trust_remote_code=trust_remote,
            padding_side="left",
            **tok_local_kw,
        )
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=trust_remote,
                **local_kw,
            )
        except TypeError:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=trust_remote,
                **local_kw,
            )
        if adapter_path:
            from peft import PeftModel

            base_model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model = base_model
        self.model.eval()

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        force_tool_schema: bool = False,
    ) -> str:
        import torch

        max_input_len = max(256, self.max_model_len - self.max_new_tokens)
        _, prompt = _fit_messages_to_token_budget(
            self.tokenizer,
            messages,
            tools,
            max_input_len,
            enable_thinking=self.enable_thinking,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        if inputs["input_ids"].shape[1] > max_input_len:
            head_len = max_input_len // 2
            tail_len = max_input_len - head_len
            inputs = {k: torch.cat([v[:, :head_len], v[:, -tail_len:]], dim=1) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "pad_token_id": (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
            "eos_token_id": _collect_stop_token_ids(self.tokenizer) or self.tokenizer.eos_token_id,
        }
        if self.repetition_penalty > 1.0:
            gen_kwargs["repetition_penalty"] = self.repetition_penalty

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        gen_ids = out[0][prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        if prompt.rstrip().endswith("<think>") and not text.lstrip().startswith("<think>"):
            text = "<think>\n" + text
        text = normalize_minicpm5_output(text)
        return _strip_special_markers(text)


class VLLMBackend:
    """High-throughput greedy causal LM backend using vLLM with optional LoRA adapter."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.5-4B",
        *,
        adapter_path: str | None = None,
        max_model_len: int = 16384,
        max_new_tokens: int = 384,
        enable_thinking: bool = False,
        dtype: str = "bfloat16",
        repetition_penalty: float = 1.0,
        gpu_memory_utilization: float = 0.85,
        max_num_seqs: int | None = None,
        guided_decoding: bool = False,
        seed: int = 42,
    ):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.model_id = model_id
        self.adapter_path = adapter_path
        self.max_model_len = max_model_len
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = bool(enable_thinking)
        self.repetition_penalty = float(repetition_penalty)
        self.guided_decoding = bool(guided_decoding)
        self.seed = int(seed)
        self._SamplingParams = SamplingParams
        tok_path = _resolve_tokenizer_path(model_id, adapter_path)
        trust_remote = _requires_trust_remote_code(model_id)
        tok_local_kw = {"local_files_only": True} if _is_offline_or_local(tok_path) else {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_path, trust_remote_code=trust_remote, **tok_local_kw
        )
        llm_kwargs: dict[str, Any] = {
            "model": model_id,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enable_lora": bool(adapter_path),
            "trust_remote_code": trust_remote,
        }
        try:
            llm_sig = inspect.signature(LLM.__init__).parameters
        except (TypeError, ValueError):
            llm_sig = {}
        has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in llm_sig.values())
        if adapter_path and ("max_lora_rank" in llm_sig or has_var_kwargs or not llm_sig):
            cfg_file = Path(adapter_path) / "adapter_config.json"
            lora_r = 64
            if cfg_file.exists():
                try:
                    lora_r = max(32, int(json.loads(cfg_file.read_text(encoding="utf-8")).get("r", 64)))
                except Exception:
                    lora_r = 64
            llm_kwargs["max_lora_rank"] = lora_r
        if "seed" in llm_sig or has_var_kwargs or not llm_sig:
            llm_kwargs["seed"] = self.seed
        if max_num_seqs is not None and ("max_num_seqs" in llm_sig or has_var_kwargs):
            llm_kwargs["max_num_seqs"] = int(max_num_seqs)
        if "language_model_only" in llm_sig and _is_vision_language_model(model_id, trust_remote=trust_remote):
            llm_kwargs["language_model_only"] = True
        try:
            self.llm = LLM(**llm_kwargs)
        except TypeError:
            llm_kwargs.pop("language_model_only", None)
            self.llm = LLM(**llm_kwargs)
        self._base_sampling_kwargs: dict[str, Any] = {
            "temperature": 0.0,
            "max_tokens": max_new_tokens,
            "repetition_penalty": self.repetition_penalty,
            "skip_special_tokens": False,
            "stop_token_ids": _collect_stop_token_ids(self.tokenizer),
        }
        try:
            sp_sig = inspect.signature(SamplingParams).parameters
        except (TypeError, ValueError):
            sp_sig = {}
        if "seed" in sp_sig:
            self._base_sampling_kwargs["seed"] = self.seed
        self.sampling_params = SamplingParams(**self._base_sampling_kwargs)
        self._lora_request = None
        if adapter_path:
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest("adapter", 1, adapter_path)

    def _build_sampling_params(self, tools: list[dict[str, Any]] | None, force_tool_schema: bool) -> Any:
        if not (force_tool_schema and getattr(self, "guided_decoding", False) and tools):
            return self.sampling_params
        allow_thinking = bool(getattr(self, "enable_thinking", False))
        sig_parts = []
        for t in tools:
            fn = t.get("function") or t
            props = ((fn.get("parameters") or {}).get("properties") or {})
            sig_parts.append((str(fn.get("name") or ""), tuple(props.keys())))
        cache_key = (allow_thinking, tuple(sig_parts))
        sp_cache = getattr(self, "_sp_cache", None)
        if sp_cache is None:
            sp_cache = {}
            self._sp_cache = sp_cache
        if cache_key in sp_cache:
            return sp_cache[cache_key]

        grammar = build_qwen35_tool_grammar(
            tools,
            allow_thinking=allow_thinking,
        )
        kwargs = dict(self._base_sampling_kwargs)
        built = None
        try:
            from vllm.sampling_params import StructuredOutputsParams

            kwargs["structured_outputs"] = StructuredOutputsParams(grammar=grammar)
            built = self._SamplingParams(**kwargs)
        except Exception:
            built = None
        if built is None:
            try:
                from vllm.sampling_params import GuidedDecodingParams

                kwargs.pop("structured_outputs", None)
                kwargs["guided_decoding"] = GuidedDecodingParams(grammar=grammar, backend="xgrammar")
                built = self._SamplingParams(**kwargs)
            except Exception:
                built = self.sampling_params
        if len(sp_cache) < 256:
            sp_cache[cache_key] = built
        return built

    def _to_vllm_input(self, prompt: str, prompt_ids: list[int]) -> Any:
        if prompt_ids and hasattr(getattr(self, "llm", None), "llm_engine"):
            try:
                from vllm.inputs import TokensPrompt

                return TokensPrompt(prompt_token_ids=prompt_ids)
            except Exception:
                pass
        return prompt

    @staticmethod
    def _decode_output(prompt: str, output_obj: Any) -> str:
        text = output_obj.outputs[0].text if output_obj and getattr(output_obj, "outputs", None) else ""
        if prompt.rstrip().endswith("<think>") and not text.lstrip().startswith("<think>"):
            text = "<think>\n" + text
        text = normalize_minicpm5_output(text)
        return _strip_special_markers(text)

    def warmup(self) -> None:
        """Run minimal dummy generation so Triton/CUDA sampling kernels compile before timed inference."""
        try:
            warm_sp = self._SamplingParams(temperature=0.0, max_tokens=2, skip_special_tokens=False)
            self.llm.generate(
                ["<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"],
                sampling_params=warm_sp,
                lora_request=self._lora_request,
                use_tqdm=False,
            )
        except Exception:
            pass

    def generate_batch(
        self,
        requests: list[tuple[list[dict[str, Any]], list[dict[str, Any]] | None, bool]],
    ) -> list[str]:
        """Run deterministic batched generation across multiple independent prompts."""
        if not requests:
            return []
        max_input_len = max(256, self.max_model_len - self.max_new_tokens)
        vllm_inputs: list[Any] = []
        raw_prompts: list[str] = []
        params_list: list[Any] = []
        for messages, tools, force_tool_schema in requests:
            _, prompt, prompt_ids = _fit_messages_with_ids(
                self.tokenizer,
                messages,
                tools or [],
                max_input_len,
                enable_thinking=self.enable_thinking,
            )
            raw_prompts.append(prompt)
            vllm_inputs.append(self._to_vllm_input(prompt, prompt_ids))
            params_list.append(self._build_sampling_params(tools, force_tool_schema))

        first_sp = params_list[0]
        uniform_sp = all(sp is first_sp for sp in params_list)
        all_default = uniform_sp and first_sp is self.sampling_params
        sp_arg: Any = first_sp if uniform_sp else params_list
        try:
            outputs = self.llm.generate(
                vllm_inputs,
                sampling_params=sp_arg,
                lora_request=self._lora_request,
                use_tqdm=False,
            )
        except Exception:
            if all_default:
                raise
            outputs = self.llm.generate(
                vllm_inputs,
                sampling_params=self.sampling_params,
                lora_request=self._lora_request,
                use_tqdm=False,
            )
        return [self._decode_output(p, out) for p, out in zip(raw_prompts, outputs)]

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        force_tool_schema: bool = False,
    ) -> str:
        return self.generate_batch([(messages, tools, force_tool_schema)])[0]


def create_llm_backend(cfg: LLMConfig) -> ChatBackend:
    if cfg.backend == "fake":
        return SequenceFakeBackend()
    if cfg.backend == "vllm":
        return VLLMBackend(
            model_id=cfg.model_id,
            adapter_path=cfg.adapter_path,
            max_model_len=cfg.max_model_len,
            max_new_tokens=cfg.max_new_tokens,
            enable_thinking=cfg.enable_thinking,
            dtype=cfg.dtype,
            repetition_penalty=cfg.repetition_penalty,
            gpu_memory_utilization=getattr(cfg, "gpu_memory_utilization", 0.85),
            max_num_seqs=getattr(cfg, "max_num_seqs", None),
            guided_decoding=getattr(cfg, "guided_decoding", False),
        )
    return TransformersBackend(
        model_id=cfg.model_id,
        adapter_path=cfg.adapter_path,
        max_model_len=cfg.max_model_len,
        max_new_tokens=cfg.max_new_tokens,
        enable_thinking=cfg.enable_thinking,
        dtype=cfg.dtype,
        repetition_penalty=cfg.repetition_penalty,
    )
