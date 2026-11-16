"""When2Call (NAACL 2025) Tri-Branch Contrastive Data Generator.

Adapted from references/tool_calling_repos/08_When2Call_NAACL2025/synthetic_data_gen/create_raw_train_data.py
Creates contrastive (ACT -> ASK) pairs by dropping required parameters from user utterances to generate
verifiable sys.ask_missing_info tool calls with 4-slot micro-<think> reasoning.
"""
from __future__ import annotations

import json
import random
import re
from typing import Any

from src.core.schemas import ASK_MISSING_INFO, Sample, ToolCall
from src.agent.sft_dataset import group_calls_into_steps
from src.data_gen.distill_think import synthesize_deterministic_think
from src.data_gen.prompt_utils import (
    build_assistant_history_message,
    build_sample_prefix_messages,
    call_to_dict,
    merge_sample_contexts,
    resolve_installed_apps,
    serialize_observation,
)

_AMOUNT_PATTERNS = [
    r"\b\d+[\s\.]*(?:nghìn|ngàn|triệu|tỷ|k|đ|vnd|đồng)\b",
    r"\b(?:\d{1,3}(?:[.,]\d{3})+|\d{4,9})\s*(?:đồng|đ|vnd)?\b",
    r"\b\d+\s*k\b",
    r"\b(?:hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|năm mươi|hai mươi|ba mươi|một trăm|hai trăm|năm trăm)\s*(?:trăm|mươi)?\s*(?:nghìn|ngàn|triệu|tỷ|k|đồng|đ|vnd)(?:\s*(?:tám|rưỡi|chẵn))?(?:\s*(?:đồng|đ|vnd))?\b",
]

_TIME_PATTERNS = [
    r"(?:lúc|vào|khoảng|tầm)?\s*\b\d{1,2}(?::\d{2}|h\d{1,2}|g\d{1,2})?\s*(?:giờ|g|h|tiếng|phút|am|pm)?\s*(?:rưỡi|kém\s*\d+)?\s*(?:sáng|trưa|chiều|tối|đêm)?(?:\s*(?:mai|ngày mai|hôm nay|tối nay|sáng mai|chiều nay))?\b",
    r"\b(?:năm|sáu|bảy|tám|chín|mười|mười một|mười hai)\s*(?:giờ|g|h|tiếng)\s*(?:rưỡi|\d{1,2}(?:\s*phút)?)?\s*(?:sáng|trưa|chiều|tối|đêm)?(?:\s*(?:mai|ngày mai|hôm nay))?\b",
    r"\b(?:hôm nay|ngày mai|sáng mai|chiều mai|tối mai|tối nay|chiều nay|trưa nay|trưa mai|sáng nay)\b",
    r"(?:vào\s+)?(?:ngày\s+)?\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",
]

_CONTACT_PATTERNS: list[tuple[str, int]] = [
    (r"(?:đến\s+số|vào\s+số|cho\s+số|tới\s+số|qua\s+số|số)?\s*(?:\+?84|0)\d{9,10}\b", re.IGNORECASE),
    (r"(?:cho\s+số|đến\s+số|vào\s+số|số)?\s*(?:không\s+(?:chín|ba|bảy|tám|năm)[a-zà-ỹ\s]+)", re.IGNORECASE),
    (
        r"(?:[Cc]ho|[Tt]ới|[Đđ]ến)\s+(?:(?:[Bb]ác|[Cc]ô|[Cc]hú|[Aa]nh|[Cc]hị|[Ee]m|[Bb]ạn|chủ nhà|sếp)\s+)?"
        r"([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*)",
        0,
    ),
]

_CONDITIONAL_RE = re.compile(r"\b(?:nếu(?:\s+như)?|trường\s+hợp|nhỡ(?:\s+mà)?)\b", re.IGNORECASE)

_CONTENT_PATTERNS = [
    r"(?:nội dung\s*(?:là|:)?|nói\s*(?:là|rằng)|bảo\s*(?:là|rằng)|nhắn\s*(?:là|rằng)|\blà\b|\brằng\b)\s+.*$",
]


def drop_required_param_from_utterance(
    text: str,
    param_name: str,
    param_val: Any = None,
) -> str:
    """Mask out or drop specific required parameter values from a Vietnamese utterance.

    - Amount/money: replaced with vague words like 'chút tiền'
    - Time/date: removed cleanly, preserving action intent
    - Contacts/recipient: removed or obscured
    - Content/message: removed cleanly
    """
    modified = text
    pname = param_name.lower()

    if pname in ("amount", "money", "price", "transfer_amount", "cash", "funds"):
        for pat in _AMOUNT_PATTERNS:
            modified = re.sub(pat, " chút tiền ", modified, flags=re.IGNORECASE)
        # Clean up any duplicated phrases like 'chút tiền tiền'
        modified = re.sub(r"\bchút tiền\s+tiền\b", "chút tiền", modified, flags=re.IGNORECASE)
        modified = re.sub(r"\s+", " ", modified).strip()

    elif pname in (
        "time",
        "start_time",
        "end_time",
        "alarm_time",
        "date",
        "datetime",
        "due_date",
        "reminder_time",
        "departure_time",
        "arrival_time",
        "pickup_time",
        "duration",
        "start",
    ):
        for pat in _TIME_PATTERNS:
            modified = re.sub(pat, " ", modified, flags=re.IGNORECASE)
        # Clean up hanging prepositions
        modified = re.sub(r"\s*(?:lúc|vào|khoảng|tầm)\s*$", "", modified, flags=re.IGNORECASE)
        modified = re.sub(r"\s+", " ", modified).strip()

    elif pname in (
        "recipient",
        "recipient_phone",
        "contact_name",
        "receiver",
        "to",
        "phone_number",
        "phone",
        "contact",
        "contact_id",
        "beneficiary_id",
    ):
        for pat, flags in _CONTACT_PATTERNS:
            modified = re.sub(pat, " ", modified, flags=flags)
        # Clean up hanging prepositions
        modified = re.sub(r"\s*(?:cho|đến|tới|vào|qua)\s*$", "", modified, flags=re.IGNORECASE)
        modified = re.sub(r"\s+", " ", modified).strip()

    elif pname in ("content", "message", "note_content", "body", "text"):
        for pat in _CONTENT_PATTERNS:
            modified = re.sub(pat, " ", modified, flags=re.IGNORECASE)
        modified = re.sub(r"\s+", " ", modified).strip()

    # If an explicit param_val was passed and is still present as whole token(s), remove it cleanly
    if param_val is not None:
        val_str = str(param_val).strip()
        if val_str and len(val_str) >= 2 and val_str.lower() in modified.lower():
            pattern = rf"(?<!\w){re.escape(val_str)}(?!\w)"
            modified = re.sub(pattern, " ", modified, flags=re.IGNORECASE)
            modified = re.sub(r"\s+", " ", modified).strip()

    # Clean up double punctuation or spaces
    modified = re.sub(r"(?:\bchút tiền\b[\s,]*){2,}", "chút tiền ", modified, flags=re.IGNORECASE)
    modified = re.sub(r"\s*,\s*,+", ",", modified)
    modified = re.sub(r"^[,.\s]+", "", modified)
    modified = re.sub(r"\s+([,.\?!:])", r"\1", modified)
    modified = re.sub(r"\s+", " ", modified).strip()
    return modified


def generate_when2call_contrastive_samples(
    train_samples: list[Sample],
    catalog: dict[str, Any],
    count: int = 500,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Synthesize verifiable ASK samples from ACT samples with required parameter omission.

    Directly adapts When2Call (NAACL 2025) tri-branch contrastive formulation:
    - ACT sample -> user drops required param -> sys.ask_missing_info(tool_name, params=[dropped])
    - Embeds 4-slot micro-<think> trace with verified 'Cần hỏi lại' boundary
    - Formats to MiniCPM5 ChatML step structure
    """
    rng = random.Random(seed)
    results: list[dict[str, Any]] = []

    candidates: list[tuple[Sample, ToolCall, str, Any]] = []
    for s in train_samples:
        if s.label is None:
            continue
        if isinstance(s.label, dict):
            calls = s.label.get("tool_calls", [])
        elif hasattr(s.label, "tool_calls"):
            calls = s.label.tool_calls
        else:
            calls = []

        # Step-0 ASK must not preempt multi-step prerequisite reads/parallel actions or conditionals.
        if len(calls) != 1 or _CONDITIONAL_RE.search(s.audio_text or ""):
            continue

        c = calls[0]
        if isinstance(c, dict):
            tname = c.get("name")
            args = c.get("arguments") or {}
        elif hasattr(c, "name"):
            tname = c.name
            args = c.arguments if hasattr(c, "arguments") else {}
        else:
            continue

        if not tname or tname == ASK_MISSING_INFO:
            continue

        spec = catalog.get(tname, {}).get("parameters", {})
        props = spec.get("properties") or {}
        required = spec.get("required") or []
        for req_param in required:
            if req_param not in args:
                continue
            # Never synthesize missing 'content'/'body' asks: 0/1800 real train samples ask content,
            # and dropping content fragments teaches aggressive false sys.ask_missing_info(['content']).
            if req_param.lower() in ("content", "message", "note_content", "body", "text"):
                continue
            prop_spec = props.get(req_param) or {}
            # Never synthesize step-0 ASK for internal x-ref / *_id fields (agent must lookup first).
            if req_param.endswith("_id") or "x-ref" in prop_spec:
                continue
            candidates.append(
                (
                    s,
                    ToolCall(name=tname, arguments=dict(args)),
                    req_param,
                    args[req_param],
                )
            )

    if not candidates:
        return results

    rng.shuffle(candidates)

    for idx, (sample, gold_call, param_to_drop, param_val) in enumerate(candidates):
        if len(results) >= count:
            break

        orig_text = sample.audio_text or ""
        masked_text = drop_required_param_from_utterance(orig_text, param_to_drop, param_val)

        # Skip candidate if utterance collapsed to an incomplete fragment or dangling punctuation
        if (
            masked_text == orig_text
            or len(masked_text.split()) < 4
            or masked_text.rstrip().endswith((",", ":", ";"))
        ):
            continue

        ask_call = ToolCall(
            name=ASK_MISSING_INFO,
            arguments={"tool_name": gold_call.name, "params": [param_to_drop]},
        )
        prompt_msgs = build_sample_prefix_messages(sample, catalog, user_text=masked_text)
        installed = resolve_installed_apps(sample, [gold_call.name])

        step_dict: dict[str, Any] = {
            "id": f"when2call_{sample.id}_{idx}",
            "sample_id": sample.id,
            "step_index": 0,
            "step_category": "ask",
            "prompt_messages": prompt_msgs,
            "step_calls": [ask_call.model_dump()],
            "tools": installed,
            "sample_context": sample.context,
        }

        # Synthesize verified 4-slot <think> trace with boundary 'Quyết định: Cần hỏi lại (Ask Missing Info) trước khi thực hiện.'
        think_trace = synthesize_deterministic_think(step_dict, sample.context, catalog)
        step_dict["think"] = think_trace

        results.append(step_dict)

    return results


def generate_post_lookup_ask_samples(
    train_samples: list[Sample],
    catalog: dict[str, dict[str, Any]],
    count: int = 300,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate multi-turn & post-lookup ASK steps (post-search ambiguity & multi-intent act-then-ask).

    Mirrors real benchmark ask patterns where >= 1 read/act step happens BEFORE sys.ask_missing_info.
    """
    rng = random.Random(seed)
    results: list[dict[str, Any]] = []

    # Mode A: Replay real multi-turn ask trajectories preserving intermediate tool results
    for s in train_samples:
        if len(results) >= count:
            break
        if not s.label or not s.label.tool_calls:
            continue
        calls = s.label.tool_calls
        ask_indices = [i for i, c in enumerate(calls) if c.name == ASK_MISSING_INFO]
        if not ask_indices or ask_indices[0] == 0:
            continue

        ask_idx = ask_indices[0]
        res_list = list(s.label.tool_results or [])
        prefix_calls = calls[:ask_idx]
        prefix_results = res_list[:ask_idx]
        steps = group_calls_into_steps(prefix_calls, prefix_results, catalog)

        prompt_msgs = build_sample_prefix_messages(s, catalog)
        for grp in steps:
            grp_calls = [c for c, _ in grp]
            prompt_msgs.append(build_assistant_history_message(grp_calls))
            for c, r in grp:
                obs = r if r is not None else {"ok": True}
                prompt_msgs.append({"role": "tool", "name": c.name, "content": serialize_observation(obs)})

        ask_call = call_to_dict(calls[ask_idx])
        installed = resolve_installed_apps(s, [c.name for c in calls])
        step_dict: dict[str, Any] = {
            "id": f"when2call_post_{s.id}",
            "sample_id": s.id,
            "step_index": len(steps),
            "step_category": "ask",
            "prompt_messages": prompt_msgs,
            "step_calls": [ask_call],
            "tools": installed,
            "sample_context": s.context,
        }
        step_dict["think"] = synthesize_deterministic_think(step_dict, s.context, catalog)
        results.append(step_dict)

    # Mode B: Synthesize ambiguous duplicate-contact ASK counterfactuals on phone.call / sms.send
    lookup_candidates: list[tuple[Sample, ToolCall, ToolCall]] = []
    for s in train_samples:
        if not s.label or not s.label.tool_calls or _CONDITIONAL_RE.search(s.audio_text or ""):
            continue
        calls = s.label.tool_calls
        for i in range(len(calls) - 1):
            c_r, c_w = calls[i], calls[i + 1]
            if c_r.name == "contacts.search" and c_w.name in ("phone.call", "sms.send"):
                q = str((c_r.arguments or {}).get("query") or "").strip()
                if q:
                    lookup_candidates.append((s, c_r, c_w))

    rng.shuffle(lookup_candidates)
    for idx, (s, c_r, c_w) in enumerate(lookup_candidates):
        if len(results) >= count:
            break
        q = str((c_r.arguments or {}).get("query") or "Minh")
        ambig_obs = {
            "ok": True,
            "contacts": [
                {"id": "c_dup_01", "name": f"{q} (Công ty)", "phone": "0901111111"},
                {"id": "c_dup_02", "name": f"{q} (Bạn học)", "phone": "0902222222"},
            ],
        }
        prompt_msgs = build_sample_prefix_messages(s, catalog)
        prompt_msgs.append(build_assistant_history_message([c_r]))
        prompt_msgs.append({"role": "tool", "name": c_r.name, "content": serialize_observation(ambig_obs)})

        ask_call = {"name": ASK_MISSING_INFO, "arguments": {"tool_name": c_w.name, "params": ["contact_id"]}}
        installed = resolve_installed_apps(s, [c_r.name, c_w.name])
        step_dict = {
            "id": f"when2call_ambig_{s.id}_{idx}",
            "sample_id": s.id,
            "step_index": 1,
            "step_category": "ask",
            "prompt_messages": prompt_msgs,
            "step_calls": [ask_call],
            "tools": installed,
            "sample_context": s.context,
        }
        step_dict["think"] = synthesize_deterministic_think(step_dict, s.context, catalog)
        results.append(step_dict)

    # Mode C: Multi-intent Act-then-Ask (complete first independent action, then ASK missing slot on second)
    act_pool = [
        s for s in train_samples
        if s.label
        and len(s.label.tool_calls) == 1
        and s.label.tool_calls[0].name != ASK_MISSING_INFO
        and not _CONDITIONAL_RE.search(s.audio_text or "")
    ]
    step0_asks = generate_when2call_contrastive_samples(train_samples, catalog, count=max(10, count // 3), seed=seed + 1)
    sample_by_id = {s.id: s for s in train_samples}
    if act_pool and step0_asks:
        rng.shuffle(act_pool)
        for idx, ask_step in enumerate(step0_asks):
            if len(results) >= count:
                break
            s1 = act_pool[idx % len(act_pool)]
            s2 = sample_by_id.get(ask_step["sample_id"])
            if s2 is None or s1.id == s2.id:
                continue
            c1 = call_to_dict(s1.label.tool_calls[0])
            s2_masked = str(ask_step["prompt_messages"][-1]["content"])
            t1 = (s1.audio_text or "").rstrip(".,!? \t\n")
            t2 = s2_masked.strip()
            if t2 and t2[0].isupper() and (len(t2) == 1 or not t2[1].isupper()):
                t2 = t2[0].lower() + t2[1:]
            merged_ctx = merge_sample_contexts(s1, s2)
            merged_sample = Sample(id=f"mi_ask_{s1.id}_{s2.id}", context=merged_ctx, history=s1.history, audio_text=f"{t1} và {t2}")
            prompt_msgs = build_sample_prefix_messages(merged_sample, catalog)
            prompt_msgs.append(build_assistant_history_message([c1]))
            obs1 = (s1.label.tool_results[0] if (s1.label and s1.label.tool_results) else None) or {"ok": True}
            prompt_msgs.append({"role": "tool", "name": c1["name"], "content": serialize_observation(obs1)})

            installed = sorted((merged_ctx.get("installed_apps") or {}).keys()) or resolve_installed_apps(merged_sample, [c1["name"]])
            step_dict = {
                "id": f"when2call_mi_{s1.id}_{s2.id}_{idx}",
                "sample_id": s2.id,
                "step_index": 1,
                "step_category": "ask",
                "prompt_messages": prompt_msgs,
                "step_calls": ask_step["step_calls"],
                "tools": installed,
                "sample_context": merged_ctx,
            }
            step_dict["think"] = synthesize_deterministic_think(step_dict, merged_ctx, catalog)
            results.append(step_dict)

    return results[:count]
