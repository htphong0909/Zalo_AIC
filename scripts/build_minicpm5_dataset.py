#!/usr/bin/env python3
"""Build leak-free, multi-turn SFT dataset for openbmb/MiniCPM5-2B.

Encodes:
1. Exact openbmb/MiniCPM5-2B chat_template.jinja schema header (<tools>...</tools>)
2. Native MiniCPM5 XML: <function name="..."><param name="...">...</param></function> (with CDATA support)
3. Structured, observation-grounded 4-Slot micro-<think> reasoning traces
4. Multi-turn trajectory unrolling with prior assistant turns formatted as <think>\n\n</think>\n\n
5. Causally grounded Recovery Augmentation (empty search repair & xref repair)
6. Hammer (ICLR 2025) tool-list order randomization for positional invariance
7. Strict exclusion of 299 Sealed Blind Dev samples
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.history import build_messages, truncate_tool_result
from src.agent.sft_dataset import (
    build_sft_trajectory,
    group_calls_into_steps,
    unroll_trajectory_steps,
)
from src.agent.tool_selector import select_tools
from src.core.dataset import load_catalog, load_split
from src.core.schemas import ASK_MISSING_INFO, Sample, ToolCall
from src.data_gen.distill_think import synthesize_deterministic_think
from src.data_gen.recovery_augmenter import (
    create_empty_search_recovery_step,
    create_xref_repair_recovery_step,
)


def format_minicpm5_tools_block(tools: list[dict[str, Any]]) -> str:
    """Format tool definitions byte-for-byte identically to openbmb/MiniCPM5-2B chat_template.jinja."""
    if not tools:
        return ""
    tool_lines = [json.dumps(t, ensure_ascii=False) for t in tools]
    return (
        "# Tools\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n"
        + "\n".join(tool_lines)
        + "\n</tools>\n\n"
        "Tool usage guidelines:\n"
        "- You may call zero or more functions. If no function calls are needed, just answer normally and do not include any <function ... </function>.\n"
        "- When calling a function, return an XML object within <function ... </function> using:\n"
        '<function name="function-name"><param name="param-name">param-value</param></function>\n'
        "- param-value may be multi-line. If it contains <, & or newline characters, wrap it in a CDATA block: "
        '<param name="param-name"><![CDATA[...multi-line value...]]></param>'
    )


def format_minicpm5_function_xml(
    step_calls: list[dict[str, Any]] | list[ToolCall],
    *,
    python_repr: bool = False,
) -> str:
    """Format step tool calls into MiniCPM5 native XML matching chat_template.jinja:

    <function name="func_name"><param name="param_name">param_val</param></function>

    python_repr=True renders non-string values with Python ``str()`` exactly like the official Jinja
    template (``{{ param_value }}`` -> ``True``, ``['a']``, ``{'k': 1}``). Use it for *prior* assistant
    turns so the training context is byte-identical to inference. The loss-bearing target turn keeps
    canonical JSON (``python_repr=False``), which the parser accepts via json/ast fallbacks.
    """
    if not step_calls:
        return ""

    blocks: list[str] = []
    for call in step_calls:
        if isinstance(call, ToolCall):
            name = call.name
            args = call.arguments or {}
        elif isinstance(call, dict):
            fn = call.get("function") or {}
            name = fn.get("name") or call.get("name", "")
            args = fn.get("arguments") or call.get("arguments") or {}
        else:
            continue

        param_parts: list[str] = []
        for k, v in (args or {}).items():
            if isinstance(v, str):
                raw_s = v if python_repr else v.strip()
                if "<" in raw_s or "&" in raw_s or "\n" in raw_s:
                    val_str = f"<![CDATA[{raw_s}]]>"
                else:
                    val_str = raw_s
            elif python_repr:
                val_str = str(v)
            elif isinstance(v, (dict, list, bool)):
                val_str = json.dumps(v, ensure_ascii=False)
            else:
                val_str = str(v)
            param_parts.append(f'<param name="{k}">{val_str}</param>')

        inner = "".join(param_parts)
        blocks.append(f'<function name="{name}">{inner}</function>')

    return "\n".join(blocks)


def format_minicpm5_assistant_content(
    step: dict[str, Any],
    sample_context: dict[str, Any] | None,
    catalog: dict[str, dict[str, Any]] | None,
) -> str:
    """Combine 4-slot <think> with native XML function calls or text reply (using \\n\\n after </think>)."""
    think_block = synthesize_deterministic_think(step, sample_context, catalog)
    step_calls = step.get("step_calls") or []

    if step_calls:
        func_xml = format_minicpm5_function_xml(step_calls)
        return f"{think_block}\n\n{func_xml}"

    # If no tool calls, emit concise assistant closing text
    cat = step.get("step_category", "act")
    if cat == "oos":
        reply = "Yêu cầu của bạn hiện chưa có ứng dụng nào trên thiết bị hỗ trợ."
    elif cat == "ask":
        reply = "Tôi cần thêm thông tin để thực hiện yêu cầu này."
    else:
        reply = "Tôi đã thực hiện xong yêu cầu của bạn."

    return f"{think_block}\n\n{reply}"


def convert_step_to_minicpm5_messages(
    step: dict[str, Any],
    sample_context: dict[str, Any] | None,
    catalog: dict[str, dict[str, Any]] | None,
    *,
    rng: random.Random | None = None,
) -> list[dict[str, str]]:
    """Convert an unrolled step into a sequence of ChatML messages matching MiniCPM5 chat_template.jinja."""
    prompt_msgs = step.get("prompt_messages") or []
    tools = list(step.get("tools") or [])
    if rng is not None and len(tools) > 1:
        rng.shuffle(tools)

    messages: list[dict[str, str]] = []

    # 1. System prompt with official MiniCPM5 <tools> schema header
    tools_block = format_minicpm5_tools_block(tools)

    for m in prompt_msgs:
        role = m.get("role", "user")
        content = str(m.get("content") or "")

        if role == "system":
            sys_content = f"{content}\n\n{tools_block}" if tools_block else content
            messages.append({"role": "system", "content": sys_content})
        elif role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            # Prior assistant turns in context use <think>\n\n</think>\n\n matching chat_template.jinja line 57
            prior_calls = m.get("tool_calls") or []
            clean_c = re.sub(r"^<think>.*?</think>\s*", "", content, flags=re.DOTALL).lstrip("\n")
            if prior_calls:
                xml_calls = format_minicpm5_function_xml(prior_calls, python_repr=True)
                body = f"{clean_c}\n{xml_calls}" if clean_c else xml_calls
                messages.append({"role": "assistant", "content": f"<think>\n\n</think>\n\n{body}"})
            else:
                messages.append({"role": "assistant", "content": f"<think>\n\n</think>\n\n{clean_c}"})
        elif role == "tool":
            # Official chat_template.jinja renders raw tool content inside <tool_response> without [tool_name] prefix
            messages.append({"role": "tool", "content": content})

    # If prompt_msgs had no system message, insert one
    if not any(m["role"] == "system" for m in messages):
        sys_fallback = f"Bạn là trợ lý ảo LifeMate.\n\n{tools_block}" if tools_block else "Bạn là trợ lý ảo LifeMate."
        messages.insert(0, {"role": "system", "content": sys_fallback})

    # 2. Assistant target turn (loop.last) with full 4-slot <think> + \n\n + <function>
    target_content = format_minicpm5_assistant_content(step, sample_context, catalog)
    messages.append({"role": "assistant", "content": target_content})

    return messages


def build_minicpm5_sft_dataset(
    split_name: str = "train",
    root: str | Path = "data",
    dev_split_path: str | Path = "configs/splits/dev_v1.json",
    out_dir: str | Path = "data/sft",
    include_recovery: bool = True,
    seed: int = 42,
) -> tuple[int, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "minicpm5_train_sft.jsonl"
    out_gz_file = out_path / "minicpm5_train_sft.jsonl.gz"

    catalog = load_catalog(split_name, root)
    samples = load_split(split_name, root)

    dev_ids: set[str] = set()
    if Path(dev_split_path).exists():
        dev_data = json.loads(Path(dev_split_path).read_text(encoding="utf-8"))
        dev_ids = set(dev_data.get("dev_ids") or dev_data.get("ids") or [])

    train_samples = [s for s in samples if s.id not in dev_ids]
    print(
        f"Loaded {len(samples)} total samples. Filtered to {len(train_samples)} leak-free training samples "
        f"(excluded {len(dev_ids)} Dev samples)."
    )

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []

    for sample in train_samples:
        traj = build_sft_trajectory(sample, catalog)
        steps = unroll_trajectory_steps(traj, split=split_name, root=str(root))

        for step in steps:
            msgs = convert_step_to_minicpm5_messages(step, sample.context, catalog, rng=rng)
            records.append(
                {
                    "id": f"{step['id']}_s{step['step_index']}",
                    "sample_id": sample.id,
                    "category": step.get("step_category", "act"),
                    "messages": msgs,
                }
            )

            # Causally grounded recovery augmentations
            if include_recovery:
                empty_rec = create_empty_search_recovery_step(step)
                if empty_rec:
                    msgs_empty = convert_step_to_minicpm5_messages(empty_rec, sample.context, catalog, rng=rng)
                    records.append(
                        {
                            "id": f"{empty_rec['id']}_s{step['step_index']}",
                            "sample_id": sample.id,
                            "category": "recovery_empty_search",
                            "messages": msgs_empty,
                        }
                    )

                xref_rec = create_xref_repair_recovery_step(step, require_grounded_context=True)
                if xref_rec:
                    msgs_xref = convert_step_to_minicpm5_messages(xref_rec, sample.context, catalog, rng=rng)
                    records.append(
                        {
                            "id": f"{xref_rec['id']}_s{step['step_index']}",
                            "sample_id": sample.id,
                            "category": "recovery_xref_repair",
                            "messages": msgs_xref,
                        }
                    )

    rng.shuffle(records)

    # Write out JSONL and synchronized GZ archive
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with gzip.open(out_gz_file, "wt", encoding="utf-8") as f_gz:
        for r in records:
            f_gz.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"Generated {len(records)} multi-turn SFT records in {out_file} "
        f"({out_file.stat().st_size / (1024*1024):.2f} MB) and {out_gz_file} "
        f"({out_gz_file.stat().st_size / (1024*1024):.2f} MB)."
    )
    return len(records), out_file


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--out-dir", default="data/sft")
    p.add_argument("--no-recovery", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    build_minicpm5_sft_dataset(
        root=args.root,
        out_dir=args.out_dir,
        include_recovery=not args.no_recovery,
        seed=args.seed,
    )
