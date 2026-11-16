#!/usr/bin/env python3
"""Build leak-free DPO preference pairs dataset for openbmb/MiniCPM5-2B.

Synthesizes contrastive (prompt, chosen, rejected) pairs from:
1. LoopTool (ACL 2026) Real Failure Mining (isolates divergence step on train)
2. When2Call (NAACL 2025) Tri-Branch Contrastive Pairs (ASK vs Hallucinated ACT / False ASK)
3. Tool-N1 (ICLR 2026) Horizon Chaining (Continue next tool vs Early Stop)
4. ToolSandbox (NAACL 2025) Anchor Binding (Observation ID Binding vs Blind Placeholder ID)

Zero-leakage guarantee: Never consumes or leaks any sample from configs/splits/dev_v1.json.
Format: Native MiniCPM5 XML (<think> + <function name="..."><param name="...">...</param></function>).
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.dataset import load_catalog, load_split
from src.core.schemas import ASK_MISSING_INFO, Sample, ToolCall
from src.data_gen.when2call_gen import drop_required_param_from_utterance
from scripts.build_minicpm5_dataset_v2 import (
    format_minicpm5_tools_block,
    format_minicpm5_function_xml,
    synthesize_deterministic_think,
    resolve_step_tools,
)


def render_prompt_messages(
    step_messages: list[dict[str, Any]],
    tools_list: list[dict[str, Any]],
    *,
    rng: random.Random | None = None,
) -> list[dict[str, str]]:
    """Format prompt messages with system tool catalog and ChatML structure."""
    tools = list(tools_list)
    if rng is not None and len(tools) > 1:
        rng.shuffle(tools)

    tools_block = format_minicpm5_tools_block(tools)
    messages: list[dict[str, str]] = []

    for m in step_messages:
        role = m.get("role", "user")
        content = str(m.get("content") or "")

        if role == "system":
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

    if not any(m["role"] == "system" for m in messages):
        sys_fallback = f"Bạn là trợ lý ảo LifeMate.\n\n{tools_block}" if tools_block else "Bạn là trợ lý ảo LifeMate."
        messages.insert(0, {"role": "system", "content": sys_fallback})

    return messages


def render_prompt_string(messages: list[dict[str, str]]) -> str:
    """Render ChatML string ending with generation prompt <|im_start|>assistant\\n."""
    lines: list[str] = []
    for m in messages:
        lines.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    lines.append("<|im_start|>assistant\n")
    return "\n".join(lines)


def _get_tool_calls(sample: Sample | dict) -> list[Any]:
    """Safely extract tool_calls whether sample is a Pydantic Sample or dict."""
    if isinstance(sample, dict):
        label = sample.get("label")
    else:
        label = sample.label
    if not label:
        return []
    if hasattr(label, "tool_calls"):
        return label.tool_calls or []
    if isinstance(label, dict):
        return label.get("tool_calls") or []
    return []


# ---------------------------------------------------------------------------
# Generator 1: When2Call Tri-Branch Contrastive Pairs
# ---------------------------------------------------------------------------
def generate_when2call_dpo_pairs(
    train_samples: list[Sample],
    catalog: dict[str, Any],
    count: int = 300,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Synthesize ASK vs Hallucinated ACT, and True ACT vs False ASK contrastive pairs."""
    catalog = {k: {"name": k, **v} if "name" not in v else v for k, v in catalog.items()}
    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []

    act_candidates: list[tuple[Sample, ToolCall, str]] = []
    for s in train_samples:
        calls = _get_tool_calls(s)
        for c in calls:
            tname = c.get("name") if isinstance(c, dict) else c.name
            args = c.get("arguments") if isinstance(c, dict) else c.arguments
            if tname == ASK_MISSING_INFO or not tname:
                continue
            spec = catalog.get(tname, {}).get("parameters", {})
            required = spec.get("required") or []
            for req_param in required:
                if req_param in (args or {}):
                    act_candidates.append((s, ToolCall(name=tname, arguments=args or {}), req_param))

    if not act_candidates:
        return pairs

    # Sub-type A: Missing param (Chosen = ASK, Rejected = Hallucinated ACT)
    target_a = count // 2
    for _ in range(target_a):
        s, gold_call, req_param = rng.choice(act_candidates)
        orig_text = s.audio_text or ""
        mod_text = drop_required_param_from_utterance(orig_text, req_param)

        tools = resolve_step_tools({}, s.context, catalog)
        msgs = [
            {"role": "system", "content": "Bạn là trợ lý ảo LifeMate."},
            {"role": "user", "content": mod_text},
        ]
        prompt_msgs = render_prompt_messages(msgs, tools, rng=rng)
        prompt_str = render_prompt_string(prompt_msgs)

        chosen_call = {
            "name": ASK_MISSING_INFO,
            "arguments": {"tool_name": gold_call.name, "params": [req_param]},
        }
        chosen_think = (
            f"<think>\n"
            f"1. Intent: Thực hiện {gold_call.name}.\n"
            f"2. Audit: Thiếu tham số bắt buộc '{req_param}' trong câu nói.\n"
            f"3. Decision: ASK -> gọi sys.ask_missing_info để hỏi người dùng.\n"
            f"</think>"
        )
        chosen_resp = f"{chosen_think}\n\n{format_minicpm5_function_xml([chosen_call])}"

        # Rejected: hallucinate parameter value and call ACT
        hallucinated_args = dict(gold_call.arguments)
        rejected_think = (
            f"<think>\n"
            f"1. Intent: Thực hiện {gold_call.name}.\n"
            f"2. Audit: Tự điền tham số mặc định cho '{req_param}'.\n"
            f"3. Decision: ACT -> gọi tool ngay.\n"
            f"</think>"
        )
        rejected_resp = f"{rejected_think}\n\n{format_minicpm5_function_xml([gold_call.model_dump()])}"

        pairs.append({
            "id": f"dpo_when2call_a_{len(pairs)}",
            "source": "when2call",
            "category": "missing_arg_ask_vs_hallucinated_act",
            "sample_id": s.id,
            "prompt_messages": prompt_msgs,
            "prompt": prompt_str,
            "chosen": chosen_resp,
            "rejected": rejected_resp,
        })

    # Sub-type B: Complete query (Chosen = True ACT, Rejected = False ASK)
    target_b = count - len(pairs)
    for _ in range(target_b):
        s, gold_call, req_param = rng.choice(act_candidates)
        orig_text = s.audio_text or ""

        tools = resolve_step_tools({}, s.context, catalog)
        msgs = [
            {"role": "system", "content": "Bạn là trợ lý ảo LifeMate."},
            {"role": "user", "content": orig_text},
        ]
        prompt_msgs = render_prompt_messages(msgs, tools, rng=rng)
        prompt_str = render_prompt_string(prompt_msgs)

        chosen_think = (
            f"<think>\n"
            f"1. Intent: Thực hiện {gold_call.name}.\n"
            f"2. Audit: Đã có đủ tham số bắt buộc trong câu nói.\n"
            f"3. Decision: ACT -> gọi {gold_call.name} trực tiếp.\n"
            f"</think>"
        )
        chosen_resp = f"{chosen_think}\n\n{format_minicpm5_function_xml([gold_call.model_dump()])}"

        # Rejected: false ask demanding optional or already-present parameter
        false_ask_call = {
            "name": ASK_MISSING_INFO,
            "arguments": {"tool_name": gold_call.name, "params": [req_param]},
        }
        rejected_think = (
            f"<think>\n"
            f"1. Intent: Thực hiện {gold_call.name}.\n"
            f"2. Audit: Chưa chắc chắn về tham số '{req_param}'.\n"
            f"3. Decision: ASK -> hỏi lại người dùng.\n"
            f"</think>"
        )
        rejected_resp = f"{rejected_think}\n\n{format_minicpm5_function_xml([false_ask_call])}"

        pairs.append({
            "id": f"dpo_when2call_b_{len(pairs)}",
            "source": "when2call",
            "category": "complete_act_vs_false_ask",
            "sample_id": s.id,
            "prompt_messages": prompt_msgs,
            "prompt": prompt_str,
            "chosen": chosen_resp,
            "rejected": rejected_resp,
        })

    return pairs[:count]


# ---------------------------------------------------------------------------
# Generator 2: Anti-Early-Stop / Horizon Chaining Pairs
# ---------------------------------------------------------------------------
def generate_anti_early_stop_dpo_pairs(
    train_samples: list[Sample],
    catalog: dict[str, Any],
    count: int = 200,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Synthesize pairs punishing premature termination when remaining sub-goals exist."""
    catalog = {k: {"name": k, **v} if "name" not in v else v for k, v in catalog.items()}
    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []

    multi_call_samples = [
        s for s in train_samples
        if len(_get_tool_calls(s)) >= 2
    ]
    if not multi_call_samples:
        return pairs

    for _ in range(count):
        s = rng.choice(multi_call_samples)
        calls = _get_tool_calls(s)
        split_idx = rng.randint(1, len(calls) - 1)
        prior_calls = calls[:split_idx]
        next_call = calls[split_idx]

        tools = resolve_step_tools({}, s.context, catalog)
        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": "Bạn là trợ lý ảo LifeMate."},
            {"role": "user", "content": s.audio_text or ""},
            {"role": "assistant", "tool_calls": [c if isinstance(c, dict) else c.model_dump() for c in prior_calls]},
            {"role": "tool", "content": json.dumps({"status": "success", "executed": True}, ensure_ascii=False)},
        ]
        prompt_msgs = render_prompt_messages(msgs, tools, rng=rng)
        prompt_str = render_prompt_string(prompt_msgs)

        next_name = next_call.get("name") if isinstance(next_call, dict) else next_call.name
        chosen_think = (
            f"<think>\n"
            f"1. Completed: Đã thực hiện bước trước.\n"
            f"2. Remaining Sub-goals: [{next_name}].\n"
            f"3. Decision: Chưa kết thúc phiên -> Thực hiện bước tiếp theo.\n"
            f"</think>"
        )
        chosen_call_dict = next_call if isinstance(next_call, dict) else next_call.model_dump()
        chosen_resp = f"{chosen_think}\n\n{format_minicpm5_function_xml([chosen_call_dict])}"

        # Rejected: early stop text claiming completion
        rejected_think = (
            f"<think>\n"
            f"1. Completed: Đã thực hiện bước trước.\n"
            f"2. Decision: Đã hoàn tất nhiệm vụ của người dùng.\n"
            f"</think>"
        )
        rejected_resp = f"{rejected_think}\n\nTôi đã thực hiện xong yêu cầu của bạn."

        pairs.append({
            "id": f"dpo_anti_early_stop_{len(pairs)}",
            "source": "tool_n1_horizon",
            "category": "continue_subgoal_vs_early_stop",
            "sample_id": s.id,
            "prompt_messages": prompt_msgs,
            "prompt": prompt_str,
            "chosen": chosen_resp,
            "rejected": rejected_resp,
        })

    return pairs[:count]


# ---------------------------------------------------------------------------
# Generator 3: Anchor Binding Pairs (Observation ID Binding vs Blind Placeholder)
# ---------------------------------------------------------------------------
def generate_anchor_binding_dpo_pairs(
    train_samples: list[Sample],
    catalog: dict[str, Any],
    count: int = 150,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Synthesize pairs punishing blind writes with hallucinated IDs."""
    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []

    id_patterns = [
        ("contacts.search", ["phone.call", "sms.send"], "contact_id", "c_01", ["c_99", "1", "unknown_id"]),
        ("bill_payment.list_bills", ["bill_payment.pay"], "bill_id", "bill_02", ["bill_01", "1", "bill_unknown"]),
        ("calendar.list_events", ["calendar.delete_event", "calendar.update_event"], "event_id", "ev_04", ["1", "ev_01"]),
    ]

    for read_tool, write_tools, id_key, mock_good_id, bad_ids in id_patterns:
        # Find matching samples or synthesize
        for _ in range(count // len(id_patterns)):
            s = rng.choice(train_samples) if train_samples else None
            write_tool = rng.choice(write_tools)
            bad_id = rng.choice(bad_ids)

            user_text = s.audio_text if s else "Xử lý thông tin giúp tôi"
            tools = resolve_step_tools({}, s.context if s else {}, catalog)

            obs_data = {"status": "success", "results": [{"id": mock_good_id, "name": "Mục tìm thấy"}]}
            msgs: list[dict[str, Any]] = [
                {"role": "system", "content": "Bạn là trợ lý ảo LifeMate."},
                {"role": "user", "content": user_text},
                {"role": "assistant", "tool_calls": [{"name": read_tool, "arguments": {}}]},
                {"role": "tool", "content": json.dumps(obs_data, ensure_ascii=False)},
            ]
            prompt_msgs = render_prompt_messages(msgs, tools, rng=rng)
            prompt_str = render_prompt_string(prompt_msgs)

            chosen_call = {"name": write_tool, "arguments": {id_key: mock_good_id}}
            chosen_think = (
                f"<think>\n"
                f"1. Observation: {read_tool} trả về {id_key}='{mock_good_id}'.\n"
                f"2. Audit: Ràng buộc ID thực thể từ kết quả quan sát.\n"
                f"3. Decision: ACT -> gọi {write_tool} với {id_key}='{mock_good_id}'.\n"
                f"</think>"
            )
            chosen_resp = f"{chosen_think}\n\n{format_minicpm5_function_xml([chosen_call])}"

            rejected_call = {"name": write_tool, "arguments": {id_key: bad_id}}
            rejected_think = (
                f"<think>\n"
                f"1. Observation: Bỏ qua kết quả quan sát.\n"
                f"2. Decision: ACT -> gọi {write_tool} với ID tự chế '{bad_id}'.\n"
                f"</think>"
            )
            rejected_resp = f"{rejected_think}\n\n{format_minicpm5_function_xml([rejected_call])}"

            pairs.append({
                "id": f"dpo_anchor_{len(pairs)}",
                "source": "toolsandbox_anchor",
                "category": "observation_binding_vs_blind_id",
                "sample_id": s.id if s else "mock",
                "prompt_messages": prompt_msgs,
                "prompt": prompt_str,
                "chosen": chosen_resp,
                "rejected": rejected_resp,
            })

    return pairs[:count]


# ---------------------------------------------------------------------------
# Generator 4: LoopTool Real Failure Mining (From SFT V2 Evaluation on Train)
# ---------------------------------------------------------------------------
def generate_looptool_mined_dpo_pairs(
    train_samples: list[Sample],
    catalog: dict[str, Any],
    preds_file: str | Path | None = None,
    eval_file: str | Path | None = None,
    count: int = 300,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Mine pairs directly from real failure traces on non-dev train samples."""
    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []

    samples_map = {s.id: s for s in train_samples}

    # If preds file exists, load failure cases
    if preds_file and Path(preds_file).exists() and eval_file and Path(eval_file).exists():
        per_sample = [json.loads(line) for line in open(eval_file, encoding="utf-8")]
        preds = {r["id"]: r for r in [json.loads(line) for line in open(preds_file, encoding="utf-8")]}

        zero_ids = [r["id"] for r in per_sample if r.get("score") == 0.0 and r["id"] in samples_map]
        for sid in zero_ids:
            s = samples_map[sid]
            p = preds.get(sid, {})
            gold_calls = _get_tool_calls(s)
            pred_calls = p.get("tool_calls", [])
            if not gold_calls:
                continue

            tools = resolve_step_tools({}, s.context, catalog)
            msgs = [
                {"role": "system", "content": "Bạn là trợ lý ảo LifeMate."},
                {"role": "user", "content": s.audio_text or ""},
            ]
            prompt_msgs = render_prompt_messages(msgs, tools, rng=rng)
            prompt_str = render_prompt_string(prompt_msgs)

            chosen_think = (
                f"<think>\n"
                f"1. Intent: Thực hiện đúng yêu cầu.\n"
                f"2. Decision: Gọi chính xác nhãn chuẩn.\n"
                f"</think>"
            )
            chosen_resp = f"{chosen_think}\n\n{format_minicpm5_function_xml(gold_calls)}"

            if pred_calls:
                rejected_calls = pred_calls
                rejected_body = format_minicpm5_function_xml(rejected_calls)
            else:
                rejected_body = "Tôi đã thực hiện xong yêu cầu của bạn."

            rejected_think = (
                f"<think>\n"
                f"1. Output lỗi từ vòng lặp trước.\n"
                f"</think>"
            )
            rejected_resp = f"{rejected_think}\n\n{rejected_body}"

            pairs.append({
                "id": f"dpo_looptool_{len(pairs)}",
                "source": "looptool_mined",
                "category": "mined_real_failure",
                "sample_id": sid,
                "prompt_messages": prompt_msgs,
                "prompt": prompt_str,
                "chosen": chosen_resp,
                "rejected": rejected_resp,
            })

    # If count not reached, augment with random failure mutations on train samples
    needed = count - len(pairs)
    if needed > 0 and train_samples:
        for _ in range(needed):
            s = rng.choice(train_samples)
            gold_calls = _get_tool_calls(s)
            if not gold_calls:
                continue

            tools = resolve_step_tools({}, s.context, catalog)
            msgs = [
                {"role": "system", "content": "Bạn là trợ lý ảo LifeMate."},
                {"role": "user", "content": s.audio_text or ""},
            ]
            prompt_msgs = render_prompt_messages(msgs, tools, rng=rng)
            prompt_str = render_prompt_string(prompt_msgs)

            chosen_resp = f"<think>\n1. Decision: Gọi đúng tool.\n</think>\n\n{format_minicpm5_function_xml(gold_calls)}"

            # Mutate gold calls into a wrong action (e.g. wrong tool name or corrupted arguments)
            first_c = gold_calls[0]
            first_name = first_c.get("name") if isinstance(first_c, dict) else first_c.name
            first_args = dict(first_c.get("arguments", {}) if isinstance(first_c, dict) else first_c.arguments)

            if "contacts.search" in first_name:
                mutated_call = {"name": "sys.ask_missing_info", "arguments": {"tool_name": "phone.call", "params": ["contact_id"]}}
            elif "bill_payment.pay" in first_name:
                mutated_call = {"name": "ewallet.transfer", "arguments": {"contact_id": "c_01", "amount": 100000}}
            else:
                mutated_call = {"name": "reminders.create", "arguments": {"content": "việc cần làm", "time": "2026-11-12T12:00:00"}}

            rejected_resp = f"<think>\n1. Decision: Quyết định sai lệch.\n</think>\n\n{format_minicpm5_function_xml([mutated_call])}"

            pairs.append({
                "id": f"dpo_looptool_mut_{len(pairs)}",
                "source": "looptool_mutated",
                "category": "mined_real_failure",
                "sample_id": s.id,
                "prompt_messages": prompt_msgs,
                "prompt": prompt_str,
                "chosen": chosen_resp,
                "rejected": rejected_resp,
            })

    return pairs[:count]


# ---------------------------------------------------------------------------
# Master Builder: Assembles leak-free DPO dataset
# ---------------------------------------------------------------------------
def build_minicpm5_dpo_dataset(
    split_name: str = "train",
    root: str | Path = "data",
    dev_split_path: str | Path = "configs/splits/dev_v1.json",
    out_dir: str | Path = "data/dpo",
    preds_file: str | Path | None = None,
    eval_file: str | Path | None = None,
    target_pairs: int = 1000,
    seed: int = 42,
) -> tuple[int, Path]:
    """Assemble verified leak-free DPO preference pairs for openbmb/MiniCPM5-2B."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_path / "minicpm5_dpo_pairs.jsonl"
    out_gz = out_path / "minicpm5_dpo_pairs.jsonl.gz"

    catalog = load_catalog(split_name, root)
    all_samples = load_split(split_name, root)

    dev_ids: set[str] = set()
    dev_path = Path(dev_split_path)
    if dev_path.exists():
        dev_ids = set(json.loads(dev_path.read_text(encoding="utf-8")).get("dev_ids", []))

    # Guard: Strictly exclude any Dev sample from DPO training!
    train_pool = [s for s in all_samples if s.id not in dev_ids]
    print(f"[DPO Builder] Loaded {len(all_samples)} samples, filtered to {len(train_pool)} non-dev train samples.")

    # Generate from all 4 sources
    when2call_pairs = generate_when2call_dpo_pairs(train_pool, catalog, count=int(target_pairs * 0.35), seed=seed)
    early_stop_pairs = generate_anti_early_stop_dpo_pairs(train_pool, catalog, count=int(target_pairs * 0.25), seed=seed)
    anchor_pairs = generate_anchor_binding_dpo_pairs(train_pool, catalog, count=int(target_pairs * 0.15), seed=seed)
    looptool_pairs = generate_looptool_mined_dpo_pairs(
        train_pool, catalog, preds_file=preds_file, eval_file=eval_file, count=int(target_pairs * 0.25), seed=seed
    )

    all_pairs = when2call_pairs + early_stop_pairs + anchor_pairs + looptool_pairs
    rng = random.Random(seed)
    rng.shuffle(all_pairs)

    # Verification: Ensure zero dev leakage and valid formatting
    verified_pairs: list[dict[str, Any]] = []
    for p in all_pairs:
        sid = p.get("sample_id", "")
        if sid in dev_ids:
            continue
        assert "prompt" in p and "chosen" in p and "rejected" in p
        assert p["chosen"] != p["rejected"]
        verified_pairs.append(p)

    with open(out_jsonl, "w", encoding="utf-8") as f_out, gzip.open(out_gz, "wt", encoding="utf-8") as f_gz:
        for p in verified_pairs:
            line = json.dumps(p, ensure_ascii=False) + "\n"
            f_out.write(line)
            f_gz.write(line)

    print(f"[DPO Builder] Generated {len(verified_pairs)} verified DPO pairs -> {out_gz}")
    return len(verified_pairs), out_gz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dev-split", default="configs/splits/dev_v1.json")
    parser.add_argument("--out-dir", default="data/dpo")
    parser.add_argument("--preds-file", default="experiments/exp_20261111_minicpm5_sft_v2_g4/preds/dev_preds.jsonl")
    parser.add_argument("--eval-file", default="experiments/exp_20261111_minicpm5_sft_v2_g4/eval/per_sample.jsonl")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    count, gz_path = build_minicpm5_dpo_dataset(
        split_name=args.split,
        root=args.root,
        dev_split_path=args.dev_split,
        out_dir=args.out_dir,
        preds_file=args.preds_file,
        eval_file=args.eval_file,
        target_pairs=args.count,
        seed=args.seed,
    )
    print(f"DPO dataset ready: {count} pairs in {gz_path}")


if __name__ == "__main__":
    main()
