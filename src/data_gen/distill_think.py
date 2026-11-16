"""Tool-N1 (ICLR 2026) & APIGen-MT (NeurIPS 2025) verified 4-slot micro-<think> reasoning distillation.

Generates concise, context-grounded reasoning traces (40-95 tokens):
1. Ý định (User intent & self-correction check)
2. Dữ liệu (Time, profile, prior tool observations, and error recovery state)
3. Kế hoạch (Read-before-Write step, parameter binding, or stop reason)
4. Ranh giới (Act vs Ask vs OOS decision)
"""
from __future__ import annotations

import json
import re
from typing import Any

from src.core.schemas import ASK_MISSING_INFO


def format_4slot_think(
    intent: str,
    grounding: str,
    action_plan: str,
    boundary: str,
) -> str:
    """Format structured 4-slot micro-<think> block."""
    lines = [
        "<think>",
        f"1. Ý định: {intent.strip()}",
        f"2. Dữ liệu: {grounding.strip()}",
        f"3. Kế hoạch: {action_plan.strip()}",
        f"4. Ranh giới: {boundary.strip()}",
        "</think>",
    ]
    return "\n".join(lines)


def _summarize_call_args(step_calls: list[dict[str, Any]], max_len: int = 90) -> str:
    parts: list[str] = []
    for c in step_calls:
        name = c.get("name", "")
        if name == ASK_MISSING_INFO:
            continue
        args = c.get("arguments") or {}
        if args:
            kv = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in list(args.items())[:3])
            parts.append(f"{name}({kv})")
        else:
            parts.append(f"{name}()")
    text = "; ".join(parts)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _extract_observation_facts(recent_tool_msgs: list[dict[str, Any]]) -> str:
    """Extract concise scalar values, prices, bills, devices, and list cardinalities from recent tool observations."""
    facts: list[str] = []
    for m in recent_tool_msgs:
        raw = str(m.get("content") or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if isinstance(obj, list):
            facts.append(f"list(len={len(obj)})")
            continue
        if not isinstance(obj, dict):
            continue
        for k in ("balance", "amount", "total", "price", "fee"):
            if k in obj and isinstance(obj[k], (int, float)) and not isinstance(obj[k], bool):
                facts.append(f"{k}={obj[k]}")
        for lk, lv in obj.items():
            if not isinstance(lv, list):
                continue
            if lk == "prices" and lv and isinstance(lv[0], dict):
                p0 = lv[0]
                parts = [f"{k}={p0[k]}" for k in ("type", "buy", "sell") if k in p0]
                if parts:
                    facts.append(f"prices[{','.join(parts)}]")
                    continue
            if lk == "bills" and lv:
                bill_parts: list[str] = []
                for b in lv[:5]:
                    if isinstance(b, dict):
                        bid = b.get("id", "")
                        btype = b.get("type", "")
                        bamt = b.get("amount", "")
                        bdue = b.get("due_date", "")
                        bill_parts.append(f"{bid}:{btype}={bamt}({bdue})")
                if bill_parts:
                    facts.append(f"bills({len(lv)})[{';'.join(bill_parts)}]")
                    continue
            if lk == "devices" and lv:
                dev_parts: list[str] = []
                for d in lv[:8]:
                    if isinstance(d, dict):
                        did = d.get("id", "")
                        fl = d.get("floor", "")
                        st = d.get("state") or {}
                        pw = st.get("power") or st.get("mode") or ""
                        dev_parts.append(f"{did}@f{fl}:{pw}")
                if dev_parts:
                    facts.append(f"devices({len(lv)})[{','.join(dev_parts)}]")
                    continue
            facts.append(f"{lk}(n={len(lv)})")
            if lv and isinstance(lv[0], dict):
                first = lv[0]
                for subk in ("balance", "amount", "total"):
                    if subk in first and isinstance(first[subk], (int, float)) and not isinstance(first[subk], bool):
                        facts.append(f"{lk}[0].{subk}={first[subk]}")
    return "; ".join(facts[:6])


def synthesize_deterministic_think(
    step: dict[str, Any],
    sample_context: dict[str, Any] | None = None,
    catalog: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Synthesize a verified, observation-grounded 4-slot micro-<think> from step data and gold tool calls."""
    cat = step.get("step_category", "act")
    step_calls = step.get("step_calls", [])
    prompt_msgs = step.get("prompt_messages", [])
    recovery_type = step.get("recovery_type")

    user_text = ""
    for m in reversed(prompt_msgs):
        if m.get("role") == "user":
            content = str(m.get("content") or "")
            if not content.startswith("LỖI CÚ PHÁP"):
                user_text = content
                break

    # Collect tool observations since the last user message
    recent_tool_msgs: list[dict[str, Any]] = []
    for m in reversed(prompt_msgs):
        if m.get("role") == "user":
            break
        if m.get("role") == "tool":
            recent_tool_msgs.append(m)
    recent_tool_msgs.reverse()

    ctx = sample_context or {}
    cur_time = str(ctx.get("current_time") or "")
    time_prefix = f"Mốc {cur_time[:16]}. " if cur_time else ""
    math_note = str(step.get("math_note") or "").strip()
    history_cue = str(step.get("history_cue") or "").strip()
    sc_note = str(step.get("self_correction_note") or "").strip()
    mandatory_kw = [str(k).strip() for k in (step.get("mandatory_keywords") or []) if str(k).strip()]
    has_asr_hint = "[Gợi ý đồng âm ASR" in user_text
    tool_names = [c.get("name", "") for c in step_calls if c.get("name")]

    # 1. Intent slot (crisp scope + condition awareness; never mix Act with OOS wording)
    u_low = user_text.lower()
    is_cond = bool(re.search(r"\b(?:nếu(?:\s+như)?|thì\s+thôi|trường\s+hợp|mà\s+(?:trên|dưới|đủ|còn))\b", u_low))
    if cat == "oos":
        if sc_note:
            intent = "Người dùng tự hủy/dừng lệnh."
        else:
            intent = "Thiếu ứng dụng hỗ trợ yêu cầu."
    elif cat == "ask":
        intent = "Thiếu tham số bắt buộc hoặc kết quả tìm kiếm mơ hồ."
    elif is_cond and step_calls:
        intent = "Yêu cầu có điều kiện (app khả dụng đủ -> kiểm tra/xử lý trước, KHÔNG phải OOS)."
    elif step_calls:
        intent = f"Thực hiện bước {', '.join(tool_names[:3])}."
    else:
        intent = "Tổng hợp kết quả đã thực thi."
    if has_asr_hint:
        intent = f"{intent} Đối chiếu [Gợi ý đồng âm ASR]."

    # 2. Grounding slot & 3. Action plan slot (facts BEFORE action, ~50% fewer tokens)
    all_reads = bool(
        step_calls
        and catalog
        and all((catalog.get(c.get("name", "")) or {}).get("kind") == "read" for c in step_calls)
    )
    arg_summary = _summarize_call_args(step_calls)

    if recovery_type == "empty_search":
        first_tool = tool_names[0] if tool_names else ""
        if first_tool == "loyalty.get_points":
            grounding = f"{time_prefix}loyalty.get_points theo tên trả về rỗng []."
            action_plan = "Gọi loyalty.get_points({}) không tham số."
        else:
            q = str(((step_calls[0].get("arguments") or {}).get("query") if step_calls else "") or "")
            grounding = f"{time_prefix}contacts.search truy vấn dài trả về rỗng []."
            action_plan = f"Rút gọn tên gọi lại contacts.search(query=\"{q}\")."
    elif recovery_type == "xref_repair":
        grounding = f"{time_prefix}Lỗi Unknown ID trước đó."
        if all_reads:
            action_plan = f"Đọc lấy ID trước: {', '.join(tool_names)}."
        else:
            action_plan = f"Dùng ID đã tra cứu gọi {arg_summary or ', '.join(tool_names)}."
    elif not step_calls:
        if recent_tool_msgs:
            obs_names = ", ".join(dict.fromkeys(str(m.get("name") or "tool") for m in recent_tool_msgs))
            grounding = f"{time_prefix}Đã xong {obs_names}."
            action_plan = "Trả lời xác nhận ngắn gọn."
        else:
            installed = ", ".join(sorted((ctx.get("installed_apps") or {}).keys()))
            if sc_note:
                grounding = f"{time_prefix}{sc_note}"
            else:
                grounding = f"{time_prefix}App khả dụng ({installed or 'hiện tại'}) không hỗ trợ."
            action_plan = "Không gọi tool; từ chối/dừng ngắn gọn."
    elif any(c.get("name") == ASK_MISSING_INFO for c in step_calls):
        ask_call = next(c for c in step_calls if c.get("name") == ASK_MISSING_INFO)
        tname = (ask_call.get("arguments") or {}).get("tool_name", "")
        params = (ask_call.get("arguments") or {}).get("params", [])
        obs_facts = _extract_observation_facts(recent_tool_msgs)
        if recent_tool_msgs:
            obs_names = ", ".join(dict.fromkeys(str(m.get("name") or "tool") for m in recent_tool_msgs))
            fact_suffix = f" ({obs_facts})" if obs_facts else ""
            grounding = f"{time_prefix}{obs_names}{fact_suffix} -> trùng/thiếu {params}."
        else:
            grounding = f"{time_prefix}Thiếu tham số {params} cho {tname}."
        action_plan = f"Gọi sys.ask_missing_info({tname}, {params})."
    elif recent_tool_msgs:
        obs_names = ", ".join(dict.fromkeys(str(m.get("name") or "tool") for m in recent_tool_msgs))
        obs_facts = _extract_observation_facts(recent_tool_msgs)
        fact_suffix = f" [{obs_facts}]" if obs_facts else ""
        grounding = f"{time_prefix}Kết quả {obs_names}{fact_suffix}."
        action_plan = f"Gọi {arg_summary or ', '.join(tool_names)}."
    elif all_reads:
        grounding = f"{time_prefix}Cần tra cứu dữ liệu/ID trước."
        action_plan = f"Đọc {arg_summary or ', '.join(tool_names)}."
    else:
        grounding = f"{time_prefix}Đủ thông tin ngữ cảnh."
        action_plan = f"Gọi {arg_summary or ', '.join(tool_names)}."

    if history_cue:
        grounding = f"{grounding} {history_cue}".strip()
    if math_note:
        grounding = f"{grounding} Tính: {math_note}".strip()
    if mandatory_kw:
        kw_joined = ", ".join(mandatory_kw)
        action_plan = f"{action_plan} Từ khóa bắt buộc: [{kw_joined}]."

    # 4. Boundary slot
    if cat == "oos":
        boundary = "Quyết định: Ngoài phạm vi (OOS), trả về danh sách rỗng."
    elif cat == "ask":
        boundary = "Quyết định: Cần hỏi lại (Ask Missing Info) trước khi thực hiện."
    elif not step_calls:
        boundary = "Quyết định: Hoàn tất phiên, dừng lại."
    else:
        boundary = "Quyết định: Đủ điều kiện thực thi (Act)."

    remaining_subgoals = step.get("remaining_subgoals") or []
    if remaining_subgoals:
        subgoal_str = f" Còn: [{', '.join(str(s) for s in remaining_subgoals)}]."
    else:
        subgoal_str = ""
    boundary = f"{boundary}{subgoal_str}"

    return format_4slot_think(intent, grounding, action_plan, boundary)


def attach_think_to_completion(
    completion_xml: str,
    think_block: str,
) -> str:
    """Prepend verified <think> block to completion XML/text."""
    clean_comp = completion_xml.strip()
    if clean_comp.startswith("<think>"):
        # Strip existing think block
        clean_comp = re.sub(r"<think>.*?</think>\s*", "", clean_comp, flags=re.DOTALL)
    return f"{think_block}\n\n{clean_comp}".strip()
