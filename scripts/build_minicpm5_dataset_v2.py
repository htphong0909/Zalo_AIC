#!/usr/bin/env python3
"""Build leak-free, multi-turn SFT dataset v2 for openbmb/MiniCPM5-2B.

Key enhancements for V2:
1. Exact openbmb/MiniCPM5-2B chat_template.jinja schema header (<tools>...</tools>).
2. Native MiniCPM5 XML: <function name="..."><param name="...">...</param></function> (with CDATA support).
3. Structured, observation-grounded 4-Slot micro-<think> reasoning traces with horizon awareness.
4. When2Call (NAACL 2025) Tri-Branch Contrastive samples for verifiable sys.ask_missing_info calls.
5. ToolSandbox (NAACL 2025) Milestone Anchor Binding for Read -> Write parameter binding.
6. APIGen-MT (NeurIPS 2025) multi-intent chaining with natural Vietnamese connectors.
7. Dual Execution Verifier (verify_step_validity) filtering 100% of candidate steps.
8. Hammer (ICLR 2025) tool catalog order randomization for positional invariance.
9. Strict zero-leakage exclusion against 299 Blind Dev samples.
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
from src.agent.prompts import build_system_prompt
from src.agent.sft_dataset import (
    build_sft_trajectory,
    group_calls_into_steps,
    unroll_trajectory_steps,
)
from src.agent.tool_selector import compact_tool_spec, select_tools
from src.core.dataset import load_catalog, load_split
from src.core.schemas import ASK_MISSING_INFO, Sample, ToolCall
from src.data_gen.distill_think import synthesize_deterministic_think
from src.data_gen.execution_verifier import verify_step_validity
from src.data_gen.when2call_gen import generate_when2call_contrastive_samples
from src.data_gen.anchor_binding_gen import generate_anchor_binding_samples
from src.data_gen.apigen_mt_gen import generate_apigen_multi_intent_samples


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


def resolve_step_tools(
    step: dict[str, Any],
    sample_context: dict[str, Any] | None,
    catalog: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Resolve and compact tool specifications for a step.

    Guarantees every element in the returned list is a valid MiniCPM5 function signature dict:
    {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    """
    raw_tools = step.get("tools") or []
    # If already a list of dicts with function spec
    if raw_tools and isinstance(raw_tools[0], dict) and ("function" in raw_tools[0] or "type" in raw_tools[0]):
        return [copy.deepcopy(t) for t in raw_tools]

    if not catalog:
        return []

    tool_names: set[str] = set()
    # Check if raw_tools contains strings
    for item in raw_tools:
        if isinstance(item, str):
            if item in catalog:
                tool_names.add(item)
            else:
                # App name, e.g. "smart_home"
                installed_apps = (sample_context or {}).get("installed_apps") or {}
                if item in installed_apps and isinstance(installed_apps[item], dict):
                    for t in installed_apps[item].get("tools") or []:
                        if t in catalog:
                            tool_names.add(t)
                else:
                    for cname in catalog:
                        if cname.startswith(f"{item}."):
                            tool_names.add(cname)

    # If no tool_names found yet, inspect sample_context
    if not tool_names and sample_context:
        installed_apps = sample_context.get("installed_apps") or {}
        for app_info in installed_apps.values():
            if isinstance(app_info, dict):
                for t in app_info.get("tools") or []:
                    if t in catalog:
                        tool_names.add(t)

    # Also make sure any tool called in step_calls is present
    for call in step.get("step_calls") or []:
        cname = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
        if cname and cname in catalog and cname != ASK_MISSING_INFO:
            tool_names.add(cname)
        elif cname == ASK_MISSING_INFO:
            args = call.get("arguments") if isinstance(call, dict) else getattr(call, "arguments", {})
            tname = (args or {}).get("tool_name")
            if tname and tname in catalog:
                tool_names.add(tname)

    # Always include ASK_MISSING_INFO if available in catalog
    if ASK_MISSING_INFO in catalog:
        tool_names.add(ASK_MISSING_INFO)

    ordered = [cname for cname in catalog if cname in tool_names]
    return [compact_tool_spec(catalog[name]) for name in ordered]


def format_minicpm5_assistant_content(
    step: dict[str, Any],
    sample_context: dict[str, Any] | None,
    catalog: dict[str, dict[str, Any]] | None,
) -> str:
    """Combine 4-slot <think> with native XML function calls or text reply (using \\n\\n after </think>)."""
    if step.get("think"):
        think_block = step["think"]
    else:
        think_block = synthesize_deterministic_think(step, sample_context, catalog)

    step_calls = step.get("step_calls") or []

    if step_calls:
        func_xml = format_minicpm5_function_xml(step_calls)
        return f"{think_block}\n\n{func_xml}"

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
    tools = resolve_step_tools(step, sample_context, catalog)

    # Hammer (ICLR 2025) tool catalog randomization
    if rng is not None and len(tools) > 1:
        rng.shuffle(tools)

    messages: list[dict[str, str]] = []

    # 1. System prompt with official MiniCPM5 <tools> schema header
    tools_block = format_minicpm5_tools_block(tools)

    for m in prompt_msgs:
        role = m.get("role", "user")
        content = str(m.get("content") or "")

        if role == "system":
            if content.strip() == "Bạn là trợ lý ảo LifeMate." and sample_context:
                content = build_system_prompt(Sample(id="ctx", context=sample_context))
            sys_content = f"{content}\n\n{tools_block}" if tools_block else content
            messages.append({"role": "system", "content": sys_content})
        elif role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            prior_calls = m.get("tool_calls") or []
            clean_c = re.sub(r"^<think>.*?</think>\s*", "", content, flags=re.DOTALL).lstrip("\n")
            if prior_calls:
                xml_calls = format_minicpm5_function_xml(prior_calls, python_repr=True)
                body = f"{clean_c}\n{xml_calls}" if clean_c else xml_calls
                messages.append({"role": "assistant", "content": f"<think>\n\n</think>\n\n{body}"})
            else:
                messages.append({"role": "assistant", "content": f"<think>\n\n</think>\n\n{clean_c}"})
        elif role == "tool":
            messages.append({"role": "tool", "content": content})

    # If prompt_msgs had no system message, insert one
    if not any(m["role"] == "system" for m in messages):
        sys_fallback = f"Bạn là trợ lý ảo LifeMate.\n\n{tools_block}" if tools_block else "Bạn là trợ lý ảo LifeMate."
        messages.insert(0, {"role": "system", "content": sys_fallback})

    # 2. Assistant target turn (loop.last) with full 4-slot <think> + \n\n + <function>
    target_content = format_minicpm5_assistant_content(step, sample_context, catalog)
    messages.append({"role": "assistant", "content": target_content})

    return messages


def build_v2_dataset(
    split_name: str = "train",
    root: str | Path = "data",
    dev_split_path: str | Path = "configs/splits/dev_v1.json",
    out_dir: str | Path = "data/sft",
    target_synthetic: int | None = None,
    when2call_count: int = 500,
    anchor_count: int = 400,
    apigen_count: int = 600,
    seed: int = 42,
) -> tuple[int, Path]:
    """Assemble leak-free SFT V2 dataset with base trajectories and dual-verified synthetic augmentations."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "minicpm5_train_sft_v2.jsonl"
    out_gz_file = out_path / "minicpm5_train_sft_v2.jsonl.gz"

    catalog = load_catalog(split_name, root)
    samples = load_split(split_name, root)

    dev_ids: set[str] = set()
    dev_file = Path(dev_split_path)
    if dev_file.exists():
        dev_data = json.loads(dev_file.read_text(encoding="utf-8"))
        dev_ids = set(dev_data.get("dev_ids") or dev_data.get("ids") or [])

    train_samples = [s for s in samples if s.id not in dev_ids]
    print(
        f"Loaded {len(samples)} total samples. Filtered to {len(train_samples)} leak-free training samples "
        f"(excluded {len(dev_ids)} Dev samples)."
    )
    assert len(dev_ids) > 0, "dev_ids must not be empty"
    assert len(train_samples) > 0, "train_samples must not be empty"

    rng = random.Random(seed)
    sample_map = {s.id: s for s in train_samples}

    # Determine counts for synthetic generation
    if target_synthetic is not None:
        w_cnt = max(1, target_synthetic // 3)
        anc_cnt = max(1, target_synthetic // 3)
        api_cnt = max(1, target_synthetic - w_cnt - anc_cnt)
    else:
        w_cnt = when2call_count
        anc_cnt = anchor_count
        api_cnt = apigen_count

    verified_steps: list[tuple[dict[str, Any], dict[str, Any] | None, str, str]] = []
    # 1. Base train trajectory steps
    base_step_count = 0
    base_verified_count = 0
    for sample in train_samples:
        traj = build_sft_trajectory(sample, catalog)
        steps = unroll_trajectory_steps(traj, split=split_name, root=str(root))
        for step in steps:
            base_step_count += 1
            if verify_step_validity(step, catalog):
                base_verified_count += 1
                verified_steps.append((step, sample.context, sample.id, step.get("step_category", "act")))

    print(f"Base Trajectories: {base_verified_count}/{base_step_count} steps verified.")

    # 2. When2Call contrastive samples
    when2call_steps = generate_when2call_contrastive_samples(train_samples, catalog, count=w_cnt, seed=seed)
    w_verified = 0
    for s_step in when2call_steps:
        if verify_step_validity(s_step, catalog):
            w_verified += 1
            sid = s_step.get("sample_id", "")
            ctx = s_step.get("sample_context") or (sample_map[sid].context if sid in sample_map else None)
            verified_steps.append((s_step, ctx, sid, "ask"))
    print(f"When2Call Samples: {w_verified}/{len(when2call_steps)} verified.")

    # 3. ToolSandbox Anchor Binding samples
    anchor_steps = generate_anchor_binding_samples(train_samples, catalog, count=anc_cnt, seed=seed)
    anc_verified = 0
    for s_step in anchor_steps:
        if verify_step_validity(s_step, catalog):
            anc_verified += 1
            sid = s_step.get("sample_id", "")
            ctx = s_step.get("sample_context") or (sample_map[sid].context if sid in sample_map else None)
            verified_steps.append((s_step, ctx, sid, "act"))
    print(f"Anchor Binding Samples: {anc_verified}/{len(anchor_steps)} verified.")

    # 4. APIGen-MT multi-intent chaining samples
    apigen_steps = generate_apigen_multi_intent_samples(
        train_samples, catalog, count=api_cnt, seed=seed, include_step_1=True
    )
    api_verified = 0
    for s_step in apigen_steps:
        if verify_step_validity(s_step, catalog):
            api_verified += 1
            sid = s_step.get("sample_id", "")
            ctx = s_step.get("sample_context")
            verified_steps.append((s_step, ctx, sid, "act"))
    print(f"APIGen-MT Samples: {api_verified}/{len(apigen_steps)} verified.")

    # 5. Format ChatML records with Hammer catalog randomization
    records: list[dict[str, Any]] = []
    for step, ctx, sid, cat in verified_steps:
        # Check zero dev leakage before processing
        if sid in dev_ids:
            continue
        # For multi-intent sample IDs like "train_0001_train_0003"
        parts = sid.split("_")
        if any(f"train_{p}" in dev_ids or p in dev_ids for p in parts if p.isdigit()):
            continue

        msgs = convert_step_to_minicpm5_messages(step, ctx, catalog, rng=rng)
        rec_id = step.get("id") or f"{sid}_s{step.get('step_index', 0)}"
        records.append(
            {
                "id": rec_id,
                "sample_id": sid,
                "category": cat,
                "messages": msgs,
            }
        )

    # Final zero-leakage safety assertion
    for r in records:
        assert r["sample_id"] not in dev_ids, f"Leakage detected: {r['sample_id']} in dev_ids"

    rng.shuffle(records)

    # 6. Save JSONL and GZ files
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with gzip.open(out_gz_file, "wt", encoding="utf-8") as f_gz:
        for r in records:
            f_gz.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"Generated {len(records)} verified multi-turn SFT records in {out_file} "
        f"({out_file.stat().st_size / (1024*1024):.2f} MB) and {out_gz_file} "
        f"({out_gz_file.stat().st_size / (1024*1024):.2f} MB)."
    )
    return len(records), out_gz_file


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build MiniCPM5 SFT V2 dataset")
    p.add_argument("--root", default="data")
    p.add_argument("--out-dir", default="data/sft")
    p.add_argument("--dev-split", default="configs/splits/dev_v1.json")
    p.add_argument("--when2call-count", type=int, default=500)
    p.add_argument("--anchor-count", type=int, default=400)
    p.add_argument("--apigen-count", type=int, default=600)
    p.add_argument("--target-synthetic", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    build_v2_dataset(
        root=args.root,
        out_dir=args.out_dir,
        dev_split_path=args.dev_split,
        target_synthetic=args.target_synthetic,
        when2call_count=args.when2call_count,
        anchor_count=args.anchor_count,
        apigen_count=args.apigen_count,
        seed=args.seed,
    )
