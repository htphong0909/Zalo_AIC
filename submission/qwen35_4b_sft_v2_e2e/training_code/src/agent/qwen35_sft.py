"""Qwen3.5 / Qwen3.8-4B-Distill SFT formatting, LoRA configuration, and prompt-completion helpers."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from src.agent.prompts import build_system_prompt
from src.core.schemas import Sample, ToolCall
from src.data_gen.distill_think import synthesize_deterministic_think

# Hybrid architecture projection modules present across all 32 layers of Qwen3.5/Qwen3.8-4B:
# - full_attention layers (8/32): q_proj, k_proj, v_proj, o_proj
# - linear_attention / Gated DeltaNet layers (24/32): in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj
# - MLP layers (32/32): gate_proj, up_proj, down_proj
QWEN35_LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

_LEGACY_RULE7_RE = re.compile(
    r'<function name="\.\.\."><param name="\.\.\.">\.\.\.</param></function>'
)
_MINICPM_FN_XML_RE = re.compile(r'<function\s+name="[^"]*">.*?</function>', re.DOTALL | re.IGNORECASE)
_THINK_STRIP_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_INNER_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

_STOP_REPLIES: tuple[str, ...] = (
    "Đã thực hiện xong yêu cầu của bạn.",
    "Tôi đã hoàn thành yêu cầu của bạn.",
    "Yêu cầu của bạn đã được xử lý xong.",
    "Đã hoàn tất thao tác theo yêu cầu.",
)

_OOS_REPLIES: tuple[str, ...] = (
    "Yêu cầu của bạn hiện chưa có ứng dụng nào trên thiết bị hỗ trợ.",
    "Thiết bị hiện chưa cài ứng dụng hỗ trợ yêu cầu này.",
    "Xin lỗi, không có ứng dụng phù hợp để thực hiện yêu cầu này.",
    "Hiện tại chưa có công cụ nào khả dụng cho yêu cầu của bạn.",
)


def format_qwen35_tool_calls_xml(step_calls: list[dict[str, Any]] | list[ToolCall]) -> str:
    """Serialize tool calls into Qwen3.5 chat_template XML (<tool_call><function=...></tool_call>)."""
    blocks: list[str] = []
    for call in step_calls:
        if isinstance(call, ToolCall):
            name = call.name
            args = call.arguments or {}
        elif isinstance(call, dict):
            fn = call.get("function") or {}
            name = str(fn.get("name") or call.get("name") or "")
            args = fn.get("arguments") if "arguments" in fn else (call.get("arguments") or {})
        else:
            continue
        if not name:
            continue

        lines = ["<tool_call>", f"<function={name}>"]
        for k, v in (args or {}).items():
            if isinstance(v, (dict, list, bool)):
                val_str = json.dumps(v, ensure_ascii=False)
            elif v is None:
                continue
            else:
                val_str = str(v).strip()
            lines.append(f"<parameter={k}>\n{val_str}\n</parameter>")
        lines.append("</function>")
        lines.append("</tool_call>")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _rewrite_system_for_qwen35(content: str, sample_context: dict[str, Any] | None) -> str:
    if "<tool_call><function=tên_tool>" in content:
        return content
    if sample_context and content.strip().startswith("Bạn là trợ lý"):
        # Rebuild directly when context matches build_system_prompt; preserve synthetic prompts otherwise.
        if "Ngữ cảnh hiện tại" in content:
            return build_system_prompt(Sample(id="ctx", context=sample_context), prompt_style="qwen35")
    return _LEGACY_RULE7_RE.sub(
        "<tool_call><function=tên_tool><parameter=tham_số>giá_trị</parameter></function></tool_call>",
        content,
    )


def _normalize_prompt_messages(
    prompt_msgs: list[dict[str, Any]],
    sample_context: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in prompt_msgs:
        role = str(m.get("role") or "user")
        content = str(m.get("content") or "")
        if role == "system":
            out.append({"role": "system", "content": _rewrite_system_for_qwen35(content, sample_context)})
        elif role == "user":
            out.append({"role": "user", "content": content})
        elif role == "assistant":
            clean_c = _THINK_STRIP_RE.sub("", content).strip()
            prior_calls = m.get("tool_calls") or []
            norm_calls: list[dict[str, Any]] = []
            for tc in prior_calls:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                cname = fn.get("name")
                cargs = fn.get("arguments") or {}
                if cname:
                    norm_calls.append({"type": "function", "function": {"name": cname, "arguments": dict(cargs)}})
            if norm_calls:
                # Strip legacy MiniCPM5 XML from synthetic history so chat_template.jinja does not double-emit calls.
                clean_c = _MINICPM_FN_XML_RE.sub("", clean_c).strip()
            msg: dict[str, Any] = {"role": "assistant", "content": clean_c}
            if norm_calls:
                msg["tool_calls"] = norm_calls
            out.append(msg)
        elif role == "tool":
            tool_msg: dict[str, Any] = {"role": "tool", "content": content}
            if m.get("name"):
                tool_msg["name"] = str(m["name"])
            out.append(tool_msg)
    return out


def _build_terminal_reply(step_category: str, rng: random.Random | None = None) -> str:
    if step_category == "oos":
        return rng.choice(_OOS_REPLIES) if rng is not None else _OOS_REPLIES[0]
    if step_category == "ask":
        return "Tôi cần thêm thông tin để thực hiện yêu cầu này."
    return rng.choice(_STOP_REPLIES) if rng is not None else _STOP_REPLIES[0]


def build_qwen35_completion(
    step: dict[str, Any],
    sample_context: dict[str, Any] | None,
    catalog: dict[str, dict[str, Any]] | None,
    *,
    enable_thinking: bool = False,
    rng: random.Random | None = None,
) -> str:
    """Build assistant completion string matching Qwen3.5 add_generation_prompt=True suffix."""
    step_calls = step.get("step_calls") or []
    body = (
        format_qwen35_tool_calls_xml(step_calls)
        if step_calls
        else _build_terminal_reply(str(step.get("step_category") or "act"), rng=rng)
    )

    if not enable_thinking:
        return f"{body}<|im_end|>"

    think_raw = str(step.get("think") or synthesize_deterministic_think(step, sample_context, catalog))
    m = _THINK_INNER_RE.search(think_raw)
    think_inner = (m.group(1) if m else think_raw).strip()
    return f"{think_inner}\n</think>\n\n{body}<|im_end|>"


def step_to_qwen35_example(
    step: dict[str, Any],
    sample_context: dict[str, Any] | None,
    tools: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]] | None,
    *,
    enable_thinking: bool = False,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Convert an unrolled step into a structured Qwen3.5/Qwen3.8 prompt-completion example."""
    tools_copy = [dict(t) for t in tools]
    if rng is not None and len(tools_copy) > 1:
        rng.shuffle(tools_copy)

    prompt_messages = _normalize_prompt_messages(step.get("prompt_messages") or [], sample_context)
    if not any(m.get("role") == "system" for m in prompt_messages) and sample_context:
        prompt_messages.insert(
            0,
            {"role": "system", "content": build_system_prompt(Sample(id="ctx", context=sample_context), prompt_style="qwen35")},
        )

    completion = build_qwen35_completion(
        step,
        sample_context,
        catalog,
        enable_thinking=enable_thinking,
        rng=rng,
    )
    return {
        "prompt_messages": prompt_messages,
        "tools": tools_copy,
        "completion": completion,
        "enable_thinking": enable_thinking,
    }


def render_qwen35_prompt_completion(
    tokenizer: Any,
    example: dict[str, Any],
) -> dict[str, str]:
    """Render structured example into exact `(prompt, completion)` strings for TRL SFTTrainer."""
    prompt = tokenizer.apply_chat_template(
        example["prompt_messages"],
        tools=example.get("tools") or None,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(example.get("enable_thinking", False)),
    )
    return {"prompt": prompt, "completion": example["completion"]}


def save_qwen35_text_only_checkpoint(
    model_or_path: Any,
    tokenizer: Any,
    output_dir: str | Path,
    *,
    dtype: Any = None,
) -> Path:
    """Export a clean Qwen3_5ForCausalLM checkpoint without model.visual or mtp weights."""
    from transformers import AutoModelForCausalLM

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if isinstance(model_or_path, (str, Path)):
        try:
            load_kwargs: dict[str, Any] = {"trust_remote_code": True}
            if dtype is not None:
                load_kwargs["dtype"] = dtype
            model = AutoModelForCausalLM.from_pretrained(str(model_or_path), **load_kwargs)
        except TypeError:
            load_kwargs = {"trust_remote_code": True}
            if dtype is not None:
                load_kwargs["torch_dtype"] = dtype
            model = AutoModelForCausalLM.from_pretrained(str(model_or_path), **load_kwargs)
    else:
        model = model_or_path

    if hasattr(model, "config"):
        model.config.use_cache = True
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    return out
