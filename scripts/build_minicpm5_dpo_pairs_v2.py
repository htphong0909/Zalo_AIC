"""DPO V2 pair builder — Step-DPO 4-slot prefix-matched counterfactuals.

Generates strictly non-dev (`train_only`) preference pairs for MiniCPM5-2B post-training,
inheriting mechanisms from:
  - Step-DPO (2024/2025): preserve identical 4-slot `<think>` prefix (`1. Ý định` & `2. Dữ liệu`)
    between `chosen` and `rejected` so shared tokens cancel mathematically and gradients focus
    on diverging plans (`3. Kế hoạch`), boundary verdicts (`4. Ranh giới`), and XML tool calls.
  - When2Call (NAACL 2025): tri-branch calibration (`act` vs `ask` vs `oos` refusal) + post-lookup
    bidirectional disambiguation (ambiguous ask vs clear act).
  - ToolSandbox (NAACL 2025) & DiaTool-DPO (SIGDIAL 2025): trajectory horizon calibration
    (`horizon_stop_vs_overcall` balanced 1:1 with `anti_early_stop`), business rules
    (`from_account_id`, `bill_payment.pay.source`), sibling obs ID binding, explicit schema
    keys, datetime grounding, and concise fuzzy arguments.

Strictly NEVER touches `<split>_environment/` or `dev_v1` IDs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gzip
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_minicpm5_dataset_v2 import (  # noqa: E402
    convert_step_to_minicpm5_messages,
    format_minicpm5_assistant_content,
)
from src.agent.sft_dataset import build_sft_trajectory, unroll_trajectory_steps  # noqa: E402
from src.core.dataset import load_catalog, load_split  # noqa: E402
from src.core.schemas import ASK_MISSING_INFO  # noqa: E402
from src.data_gen.anchor_binding_gen import generate_anchor_binding_samples  # noqa: E402
from src.data_gen.distill_think import format_4slot_think  # noqa: E402
from src.data_gen.execution_verifier import verify_step_validity  # noqa: E402
from src.data_gen.when2call_gen import (  # noqa: E402
    generate_post_lookup_ask_samples,
    generate_when2call_contrastive_samples,
)

_THINK_BLOCK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
_SLOT1_RE = re.compile(r"^1\.\s*Ý định:\s*(.+)$", re.MULTILINE)
_SLOT2_RE = re.compile(r"^2\.\s*Dữ liệu:\s*(.+)$", re.MULTILINE)
_SLOT3_RE = re.compile(r"^3\.\s*Kế hoạch:\s*(.+)$", re.MULTILINE)
_SLOT4_RE = re.compile(r"^4\.\s*Ranh giới:\s*(.+)$", re.MULTILINE)
_ISO_DT_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?(.*)$")
_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

_EXPLICIT_OFTEN_DROPPED_KEYS = (
    "power",
    "date_from",
    "date_to",
    "near",
    "mode",
    "location",
    "source",
    "repeat",
    "days",
)

_SUBTOOL_SWAPS: dict[str, list[str]] = {
    "sms.send": ["chat_ott.send_message", "email.send"],
    "chat_ott.send_message": ["sms.send", "email.send"],
    "email.send": ["sms.send", "chat_ott.send_message"],
    "notes.create": ["reminders.create", "tasks.add"],
    "reminders.create": ["notes.create", "tasks.add", "alarm.create"],
    "tasks.add": ["reminders.create", "notes.create"],
    "alarm.create": ["reminders.create"],
    "calendar.create_event": ["reminders.create"],
}

_OVERCALL_COMPANION_TOOLS = (
    "reminders.create",
    "notes.create",
    "calendar.create_event",
    "tasks.add",
    "sms.send",
    "news.save",
    "calendar.list_events",
)

_VERBOSE_PREFIXES = (
    "Xin chào, tôi xin phép gửi thông báo chi tiết như sau: ",
    "Thông tin xác nhận đầy đủ gửi đến bạn: ",
    "Kính gửi anh/chị, nội dung cập nhật chi tiết là: ",
)
_VERBOSE_SUFFIXES = (
    " Vui lòng kiểm tra và phản hồi lại cho tôi sớm nhé!",
    " Cảm ơn bạn rất nhiều vì đã hỗ trợ xác nhận.",
    " Nếu có thay đổi nào khác vui lòng báo lại sau.",
)


def _extract_tool_names(tools: list[Any] | None) -> list[str]:
    """Extract tool names regardless of OpenAI `{function: {name}}` or flat `{name}` schema."""
    names: list[str] = []
    for t in tools or []:
        if isinstance(t, str) and t:
            names.append(t)
        elif isinstance(t, dict):
            fn = t.get("function") if isinstance(t.get("function"), dict) else t
            nm = str((fn or {}).get("name") or t.get("name") or "")
            if nm:
                names.append(nm)
    return names


def _build_step_metadata(sample: Any, catalog: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Attach `sample_context` and `user_goal` onto unrolled steps for formatting."""
    ctx = sample.context.model_dump() if hasattr(sample.context, "model_dump") else dict(sample.context or {})
    return {
        "sample_context": ctx,
        "user_goal": str(getattr(sample, "audio_text", "") or ""),
    }


def render_prompt_chatml(prompt_msgs: list[dict[str, Any]]) -> str:
    """Render prompt messages identically to scripts/train_minicpm5_sft.py TRAIN_CHAT_TEMPLATE."""
    parts: list[str] = []
    n = len(prompt_msgs)
    for i, msg in enumerate(prompt_msgs):
        role = msg.get("role", "")
        content = str(msg.get("content") or "")
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif role == "tool":
            if i == 0 or prompt_msgs[i - 1].get("role") != "tool":
                parts.append("<|im_start|>user")
            parts.append(f"\n<tool_response>\n{content}\n</tool_response>")
            if i + 1 == n or prompt_msgs[i + 1].get("role") != "tool":
                parts.append("<|im_end|>\n")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _normalize_calls(calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Canonical representation of tool calls for equivalence comparison."""
    out: list[tuple[str, str]] = []
    for c in calls:
        name = str(c.get("name") or "")
        args = c.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        out.append((name, json.dumps(args, ensure_ascii=False, sort_keys=True)))
    return out


def _extract_4slots(think_source: str) -> tuple[str, str, str, str]:
    """Parse 4-slot Vietnamese <think> lines if present."""
    m = _THINK_BLOCK_RE.search(think_source or "")
    body = m.group(1) if m else (think_source or "")
    s1 = _SLOT1_RE.search(body)
    s2 = _SLOT2_RE.search(body)
    s3 = _SLOT3_RE.search(body)
    s4 = _SLOT4_RE.search(body)
    return (
        s1.group(1).strip() if s1 else "",
        s2.group(1).strip() if s2 else "",
        s3.group(1).strip() if s3 else "",
        s4.group(1).strip() if s4 else "",
    )


def _summarize_rejected_calls(calls: list[dict[str, Any]]) -> str:
    """Compact summary of calls matching distill_think._summarize_call_args style."""
    parts: list[str] = []
    for c in calls[:3]:
        name = str(c.get("name") or "")
        args = c.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        kv = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in list(args.items())[:4])
        parts.append(f"{name}({kv})")
    return "; ".join(parts)


def _build_realistic_rejected_think(
    user_goal: str,
    rejected_calls: list[dict[str, Any]],
    rejected_final_text: str,
    prior_obs_note: str = "",
    *,
    chosen_think: str = "",
    action_plan_override: str = "",
    boundary_override: str = "",
) -> str:
    """Build 4-slot Vietnamese `<think>` matching chosen's Slots 1 & 2 (Step-DPO prefix)."""
    s1, s2, _, _ = _extract_4slots(chosen_think)
    intent = s1 or f"Thực hiện yêu cầu '{user_goal[:70]}' của người dùng."
    grounding = s2 or (
        prior_obs_note.strip()
        if prior_obs_note
        else "Kiểm tra ngữ cảnh hội thoại và thông tin hiện có để xác định bước tiếp theo."
    )

    if action_plan_override:
        action_plan = action_plan_override.strip()
    elif not rejected_calls:
        snippet = (rejected_final_text or "Hoàn tất phản hồi trực tiếp cho người dùng.").strip()[:60]
        action_plan = f"Không gọi thêm tool nào; trả lời ngắn gọn '{snippet}'."
    elif any(c.get("name") == ASK_MISSING_INFO for c in rejected_calls):
        ask_c = next(c for c in rejected_calls if c.get("name") == ASK_MISSING_INFO)
        ask_args = ask_c.get("arguments") or {}
        tname = ask_args.get("tool_name", "")
        params = ask_args.get("params", [])
        action_plan = f"Gọi sys.ask_missing_info cho '{tname}' về các tham số {params}."
    else:
        names = ", ".join(str(c.get("name") or "") for c in rejected_calls if c.get("name"))
        summary = _summarize_rejected_calls(rejected_calls)
        action_plan = f"Gọi {names} ({summary}) với các tham số đã xác định."

    if boundary_override:
        boundary = boundary_override.strip()
    elif not rejected_calls:
        boundary = "Quyết định: Hoàn tất phiên, dừng lại. Mục tiêu còn lại: Không còn."
    elif any(c.get("name") == ASK_MISSING_INFO for c in rejected_calls):
        boundary = "Quyết định: Cần hỏi lại (Ask Missing Info) trước khi thực hiện. Mục tiêu còn lại: Không còn."
    else:
        boundary = "Quyết định: Đủ điều kiện thực thi (Act). Mục tiêu còn lại: Không còn."

    return format_4slot_think(intent, grounding, action_plan, boundary)


def _step_to_pair(
    chosen_step: dict[str, Any],
    *,
    rejected_calls: list[dict[str, Any]],
    rejected_final_text: str,
    rejected_think: str | None,
    category: str,
    sample_id: str,
    catalog: dict[str, dict[str, Any]],
    rng: random.Random,
    max_tokens: int,
    max_prompt_tokens: int = 14000,
    tokenizer: Any = None,
    rej_plan_override: str = "",
    rej_boundary_override: str = "",
) -> dict[str, Any] | None:
    """Validate `chosen_step`, render Step-DPO prefix-aligned `chosen`/`rejected`, enforce token caps."""
    if not verify_step_validity(chosen_step, catalog):
        return None
    sample_ctx = chosen_step.get("sample_context")
    msgs = convert_step_to_minicpm5_messages(copy.deepcopy(chosen_step), sample_ctx, catalog, rng=rng)
    if len(msgs) < 2:
        return None
    prompt_msgs = msgs[:-1]
    chosen_resp = str(msgs[-1]["content"])

    sys_content = str(prompt_msgs[0].get("content", "")) if prompt_msgs and prompt_msgs[0].get("role") == "system" else ""
    if chosen_step.get("tools") and "Quy tắc bắt buộc" not in sys_content:
        return None

    user_goal = str(chosen_step.get("user_goal") or "")
    if not user_goal:
        for m in reversed(prompt_msgs):
            if m.get("role") == "user":
                user_goal = str(m.get("content", ""))
                break

    if not rejected_think:
        rejected_think = _build_realistic_rejected_think(
            user_goal,
            rejected_calls,
            rejected_final_text,
            chosen_think=chosen_resp,
            action_plan_override=rej_plan_override,
            boundary_override=rej_boundary_override,
        )
    elif "<think>" not in rejected_think:
        rejected_think = f"<think>\n{rejected_think.strip()}\n</think>"

    rejected_step = {
        **chosen_step,
        "step_calls": list(rejected_calls),
        "final_text": rejected_final_text,
        "think": rejected_think,
    }
    rejected_resp = format_minicpm5_assistant_content(rejected_step, sample_ctx, catalog)
    if chosen_resp.strip() == rejected_resp.strip():
        return None

    prompt_str = render_prompt_chatml(prompt_msgs)
    max_comp_str = chosen_resp if len(chosen_resp) >= len(rejected_resp) else rejected_resp
    if tokenizer is not None:
        prompt_tok_len = len(tokenizer.encode(prompt_str, add_special_tokens=False))
        tok_len = len(tokenizer.encode(prompt_str + max_comp_str + "<|im_end|>", add_special_tokens=False))
    else:
        prompt_tok_len = int(len(prompt_str) / 2.6) + 32
        tok_len = int(len(prompt_str + max_comp_str) / 2.6) + 64
    if prompt_tok_len > max_prompt_tokens or tok_len > max_tokens:
        return None

    return {
        "id": f"dpo_{sample_id}_{category}_{rng.randint(1000, 999999)}",
        "sample_id": sample_id,
        "category": category,
        "prompt": prompt_str,
        "chosen": chosen_resp,
        "rejected": rejected_resp,
        "prompt_messages": prompt_msgs,
        "est_prompt_tokens": prompt_tok_len,
        "est_tokens": tok_len,
    }


def _mutate_entity_id(val: str, rng: random.Random) -> str:
    """Create a realistic plausible-but-wrong entity ID."""
    m = re.match(r"^(.*?)(\d+)$", val)
    if m:
        prefix, digits = m.group(1), m.group(2)
        num = int(digits)
        delta = rng.choice([-2, -1, 1, 2])
        new_num = max(1, num + delta)
        if new_num == num:
            new_num = num + 1
        return f"{prefix}{str(new_num).zfill(len(digits))}"
    return f"{val}_alt"


def _parse_tool_obs_payloads(prompt_msgs: list[dict[str, Any]]) -> list[tuple[str, Any]]:
    """Collect parsed JSON payloads from recent `role == 'tool'` messages after the last user turn."""
    out: list[tuple[str, Any]] = []
    recent: list[dict[str, Any]] = []
    for m in reversed(prompt_msgs or []):
        if m.get("role") == "user":
            break
        if m.get("role") == "tool":
            recent.append(m)
    recent.reverse()
    for m in recent:
        tname = str(m.get("name") or "")
        raw = m.get("content")
        payload: Any = raw
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except Exception:
                payload = raw
        out.append((tname, payload))
    return out


def _collect_obs_records(payload: Any) -> list[dict[str, Any]]:
    """Flatten dict records nested in tool observations."""
    records: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        records.append(payload)
        for v in payload.values():
            if isinstance(v, (dict, list)):
                records.extend(_collect_obs_records(v))
    elif isinstance(payload, list):
        for item in payload:
            records.extend(_collect_obs_records(item))
    return records


def _extract_candidate_ids_from_obs(prompt_msgs: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Map ID-like keys (`id`, `contact_id`, `account_id`, ...) -> ordered unique values from obs."""
    by_key: dict[str, list[str]] = {}
    for _, payload in _parse_tool_obs_payloads(prompt_msgs):
        for rec in _collect_obs_records(payload):
            for k, v in rec.items():
                if not isinstance(v, (str, int)):
                    continue
                if k == "id" or k.endswith("_id") or k in ("channel", "from_account_id", "source"):
                    sval = str(v).strip()
                    if sval:
                        by_key.setdefault(k, [])
                        if sval not in by_key[k]:
                            by_key[k].append(sval)
    return by_key


def _pick_realistic_param_default(
    pname: str,
    pschema: dict[str, Any],
    sample_ctx: dict[str, Any] | None,
    candidate_ids: dict[str, list[str]],
    rng: random.Random,
    user_goal: str = "",
) -> Any:
    """Fill a schema parameter with a realistic plausible guess (avoiding placeholder literals)."""
    enums = pschema.get("enum") or []
    if enums:
        return rng.choice(list(enums))
    if pname in candidate_ids and candidate_ids[pname]:
        return candidate_ids[pname][0]
    if pname.endswith("_id") and candidate_ids.get("id"):
        return candidate_ids["id"][0]
    ptype = str(pschema.get("type") or "string")
    if ptype in ("integer", "number"):
        if "amount" in pname or "money" in pname or "price" in pname:
            return rng.choice([50000, 100000, 200000])
        return 1
    if ptype == "boolean":
        return True
    ctx = sample_ctx or {}
    cur_time = str(ctx.get("current_time") or "2026-06-15T09:00:00+07:00")
    if pname in ("date", "date_from", "date_to", "departure_date"):
        return cur_time[:10]
    if pname in ("time", "start", "due"):
        return cur_time[:16]
    if pname == "end":
        return f"{cur_time[:10]}T10:00"
    if pname == "channel":
        return "VTV1"
    if pname in ("origin", "destination", "location", "near"):
        return str((ctx.get("user_profile") or {}).get("home_address") or "Quận 1, TP.HCM")
    if pname.endswith("_id"):
        prefix = pname[:-3][:4] or "item"
        return f"{prefix}_01"
    goal_words = [w.strip(",.?!") for w in (user_goal or "").split() if w.strip(",.?!")]
    if goal_words:
        return " ".join(goal_words[:6])
    return "Xác nhận thông tin"


def _build_plausible_tool_args(
    tool_name: str,
    catalog: dict[str, dict[str, Any]],
    sample_ctx: dict[str, Any] | None,
    prompt_msgs: list[dict[str, Any]],
    rng: random.Random,
    *,
    partial_args: dict[str, Any] | None = None,
    user_goal: str = "",
) -> dict[str, Any]:
    """Construct complete schema-valid plausible arguments for a tool."""
    if not user_goal:
        for m in reversed(prompt_msgs or []):
            if m.get("role") == "user":
                user_goal = str(m.get("content") or "")
                break
    schema = (catalog.get(tool_name) or {}).get("parameters") or {}
    props = schema.get("properties") or {}
    required = list(schema.get("required") or [])
    candidate_ids = _extract_candidate_ids_from_obs(prompt_msgs)
    args = dict(partial_args or {})
    for pname in required:
        if pname not in args or args[pname] in (None, "", "không rõ"):
            args[pname] = _pick_realistic_param_default(
                pname, props.get(pname) or {}, sample_ctx, candidate_ids, rng, user_goal=user_goal
            )
    for k, v in list(args.items()):
        if v == "không rõ":
            args[k] = _pick_realistic_param_default(
                k, props.get(k) or {}, sample_ctx, candidate_ids, rng, user_goal=user_goal
            )
    return args


def generate_on_policy_trace_pairs(
    samples_by_id: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    traces_path: Path,
    per_sample_path: Path | None,
    *,
    root: str | Path,
    split: str,
    rng: random.Random,
    max_tokens: int,
    max_prompt_tokens: int = 14000,
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    """Extract true on-policy mistakes from SFT checkpoint traces aligned to gold trajectory steps."""
    if not traces_path.exists():
        return []

    failed_ids: set[str] | None = None
    if per_sample_path and per_sample_path.exists():
        failed_ids = set()
        with open(per_sample_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if float(rec.get("final_score", rec.get("score", 1.0))) < 0.999:
                        failed_ids.add(str(rec["id"]))

    traces_by_id: dict[str, list[dict[str, Any]]] = {}
    with open(traces_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                sid = str(rec.get("id") or "")
                if sid in samples_by_id and (failed_ids is None or sid in failed_ids):
                    traces_by_id[sid] = rec.get("trace") or []

    pairs: list[dict[str, Any]] = []
    for sid, trace in traces_by_id.items():
        sample = samples_by_id[sid]
        traj = build_sft_trajectory(sample, catalog)
        gold_steps = unroll_trajectory_steps(traj)
        meta = _build_step_metadata(sample, catalog)

        step_events: list[dict[str, Any]] = []
        for ev in trace:
            ev_type = ev.get("event")
            if ev_type in ("executed", "final_text", "schema_repair", "schema_abort", "duplicate_read_stop"):
                step_events.append(ev)

        gold_idx = 0
        for ev in step_events:
            if gold_idx >= len(gold_steps):
                break
            g_step = gold_steps[gold_idx]
            g_calls = g_step.get("step_calls") or []
            ev_type = ev.get("event")

            if ev_type == "executed":
                p_calls = ev.get("calls") or []
                if _normalize_calls(p_calls) == _normalize_calls(g_calls):
                    gold_idx += 1
                    continue
                enriched = {**g_step, **meta}
                pair = _step_to_pair(
                    enriched,
                    rejected_calls=p_calls,
                    rejected_final_text="",
                    rejected_think=ev.get("raw_output") if "<think>" in str(ev.get("raw_output") or "") else None,
                    category="on_policy_wrong_call",
                    sample_id=sid,
                    catalog=catalog,
                    rng=rng,
                    max_tokens=max_tokens,
                    max_prompt_tokens=max_prompt_tokens,
                    tokenizer=tokenizer,
                )
                if pair:
                    pairs.append(pair)
                break

            if ev_type == "final_text":
                if not g_calls:
                    gold_idx += 1
                    continue
                enriched = {**g_step, **meta}
                pair = _step_to_pair(
                    enriched,
                    rejected_calls=[],
                    rejected_final_text=str(ev.get("final_text") or ev.get("text") or "Đã hoàn tất yêu cầu của bạn."),
                    rejected_think=ev.get("raw_output") if "<think>" in str(ev.get("raw_output") or "") else None,
                    category="on_policy_early_stop",
                    sample_id=sid,
                    catalog=catalog,
                    rng=rng,
                    max_tokens=max_tokens,
                    max_prompt_tokens=max_prompt_tokens,
                    tokenizer=tokenizer,
                )
                if pair:
                    pairs.append(pair)
                break

            if ev_type in ("schema_repair", "schema_abort"):
                bad_call = ev.get("call")
                p_calls = [bad_call] if isinstance(bad_call, dict) else []
                if not p_calls:
                    break
                enriched = {**g_step, **meta}
                pair = _step_to_pair(
                    enriched,
                    rejected_calls=p_calls,
                    rejected_final_text="",
                    rejected_think=None,
                    category="on_policy_schema_err",
                    sample_id=sid,
                    catalog=catalog,
                    rng=rng,
                    max_tokens=max_tokens,
                    max_prompt_tokens=max_prompt_tokens,
                    tokenizer=tokenizer,
                )
                if pair:
                    pairs.append(pair)
                break

    return pairs


def generate_horizon_stop_vs_overcall_pairs(
    samples: list[Any],
    catalog: dict[str, dict[str, Any]],
    *,
    target_count: int,
    rng: random.Random,
    max_tokens: int,
    max_prompt_tokens: int = 14000,
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    """Pair `final_stop` steps (`[]`) against realistic extra write/duplicate/read overcalls."""
    candidates: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []
    for sample in samples:
        traj = build_sft_trajectory(sample, catalog)
        if traj.get("stratum", "").startswith(("ask|", "oos|")):
            continue
        steps = unroll_trajectory_steps(traj)
        meta = _build_step_metadata(sample, catalog)
        for st in steps:
            if st.get("step_category") == "final_stop" and (st.get("prefix_calls") or []):
                candidates.append((sample, st, meta))

    rng.shuffle(candidates)
    pairs: list[dict[str, Any]] = []
    for sample, st, meta in candidates:
        if len(pairs) >= target_count:
            break
        prefix_calls = st.get("prefix_calls") or []
        installed_names = _extract_tool_names(st.get("tools"))
        prefix_names = {str(c.get("name") or "") for c in prefix_calls}
        prompt_msgs = st.get("prompt_messages") or []
        sample_ctx = meta.get("sample_context")

        roll = rng.random()
        rej_call: dict[str, Any] | None = None
        if roll < 0.42:
            # Mode A: repeat last executed tool call (duplicate write/read overcall)
            last_c = copy.deepcopy(prefix_calls[-1])
            rej_call = {"name": last_c.get("name", ""), "arguments": last_c.get("arguments") or {}}
        elif roll < 0.80:
            # Mode B: unrequested companion write tool (e.g. reminders.create / notes.create / calendar.create_event)
            comp_pool = [t for t in _OVERCALL_COMPANION_TOOLS if t in installed_names and t not in prefix_names]
            if not comp_pool:
                comp_pool = [
                    t
                    for t in installed_names
                    if t not in prefix_names and (catalog.get(t) or {}).get("kind") != "read"
                ]
            if comp_pool:
                pick_t = rng.choice(comp_pool)
                rej_call = {
                    "name": pick_t,
                    "arguments": _build_plausible_tool_args(pick_t, catalog, sample_ctx, prompt_msgs, rng),
                }
        if rej_call is None:
            # Mode C: spurious trailing read tool after already completing task
            read_pool = [t for t in installed_names if (catalog.get(t) or {}).get("kind") == "read"]
            if read_pool:
                pick_r = rng.choice(read_pool)
                rej_call = {
                    "name": pick_r,
                    "arguments": _build_plausible_tool_args(pick_r, catalog, sample_ctx, prompt_msgs, rng),
                }
            else:
                last_c = copy.deepcopy(prefix_calls[-1])
                rej_call = {"name": last_c.get("name", ""), "arguments": last_c.get("arguments") or {}}

        r_name = str(rej_call.get("name") or "")
        enriched = {**st, **meta}
        pair = _step_to_pair(
            enriched,
            rejected_calls=[rej_call],
            rejected_final_text="",
            rejected_think=None,
            category="horizon_stop_vs_overcall",
            sample_id=sample.id,
            catalog=catalog,
            rng=rng,
            max_tokens=max_tokens,
            max_prompt_tokens=max_prompt_tokens,
            tokenizer=tokenizer,
            rej_plan_override=f"Gọi thêm {r_name} ({_summarize_rejected_calls([rej_call])}) để xử lý bổ sung.",
            rej_boundary_override=f"Quyết định: Tiếp tục thực thi bước bổ sung. Mục tiêu còn lại: [{r_name}].",
        )
        if pair:
            pairs.append(pair)
    return pairs


def generate_anti_early_stop_pairs(
    samples: list[Any],
    catalog: dict[str, dict[str, Any]],
    *,
    root: str | Path,
    split: str,
    target_count: int,
    rng: random.Random,
    max_tokens: int,
    max_prompt_tokens: int = 14000,
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    """Build anti-early-stop pairs weighted toward multi-intent tails and multi-item loops."""
    priority_candidates: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []
    regular_candidates: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []

    for sample in samples:
        traj = build_sft_trajectory(sample, catalog)
        if traj.get("stratum", "").startswith(("ask|", "oos|")):
            continue
        steps = unroll_trajectory_steps(traj)
        meta = _build_step_metadata(sample, catalog)
        for idx, st in enumerate(steps):
            if st.get("step_category") == "act" and st.get("step_calls") and idx >= 1:
                prefix_calls = st.get("prefix_calls") or []
                prefix_names = [str(c.get("name") or "") for c in prefix_calls]
                cur_names = [str(c.get("name") or "") for c in (st.get("step_calls") or [])]
                has_prior_write = any((catalog.get(n) or {}).get("kind") != "read" for n in prefix_names)
                is_multi_item_loop = any(n in prefix_names for n in cur_names)
                if has_prior_write or is_multi_item_loop:
                    priority_candidates.append((sample, st, meta))
                else:
                    regular_candidates.append((sample, st, meta))

    rng.shuffle(priority_candidates)
    rng.shuffle(regular_candidates)
    ordered = priority_candidates + regular_candidates

    pairs: list[dict[str, Any]] = []
    for sample, st, meta in ordered:
        if len(pairs) >= target_count:
            break
        enriched = {**st, **meta}
        next_tools = [c.get("name", "") for c in (st.get("step_calls") or [])]
        early_text = "Tôi đã hoàn tất xử lý yêu cầu của bạn."
        pair = _step_to_pair(
            enriched,
            rejected_calls=[],
            rejected_final_text=early_text,
            rejected_think=None,
            category="anti_early_stop",
            sample_id=sample.id,
            catalog=catalog,
            rng=rng,
            max_tokens=max_tokens,
            max_prompt_tokens=max_prompt_tokens,
            tokenizer=tokenizer,
            rej_plan_override=f"Bỏ qua bước còn lại ({', '.join(next_tools)}); dừng và trả lời người dùng ngay.",
            rej_boundary_override="Quyết định: Hoàn tất phiên, dừng lại. Mục tiêu còn lại: Không còn.",
        )
        if pair:
            pairs.append(pair)
    return pairs


def generate_post_lookup_dpo_pairs(
    samples: list[Any],
    catalog: dict[str, dict[str, Any]],
    *,
    root: str | Path,
    split: str,
    target_count: int,
    rng: random.Random,
    max_tokens: int,
    max_prompt_tokens: int = 14000,
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    """Bidirectional post-lookup pairs: ambiguous `ask` vs premature guess, AND clear `act` vs false ask."""
    ask_target = int(round(target_count * 0.50))
    clear_target = max(1, target_count - ask_target)

    # 1. Real post-lookup ASK steps FIRST (100% intent-coherent), plus domain-coherent synthetic ASK steps
    real_ask_steps: list[tuple[str, dict[str, Any]]] = []
    synth_ask_steps: list[tuple[str, dict[str, Any]]] = []
    clear_act_steps: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []

    for sample in samples:
        traj = build_sft_trajectory(sample, catalog)
        steps = unroll_trajectory_steps(traj)
        meta = _build_step_metadata(sample, catalog)
        for st in steps:
            calls = st.get("step_calls") or []
            p_msgs = st.get("prompt_messages") or []
            has_obs = any(m.get("role") == "tool" for m in p_msgs)
            if not has_obs or not calls:
                continue
            if any(c.get("name") == ASK_MISSING_INFO for c in calls):
                real_ask_steps.append((sample.id, {**st, **meta}))
            elif st.get("step_category") == "act" and all(c.get("name") != ASK_MISSING_INFO for c in calls):
                clear_act_steps.append((sample, st, meta))

    synth_post = generate_post_lookup_ask_samples(samples, catalog, count=max(60, ask_target * 2), seed=rng.randint(1, 99999))
    for st in synth_post:
        ask_c = next((c for c in (st.get("step_calls") or []) if c.get("name") == ASK_MISSING_INFO), None)
        if not ask_c:
            continue
        target_app = str((ask_c.get("arguments") or {}).get("tool_name") or "").split(".")[0]
        obs_apps = {str(m.get("name") or "").split(".")[0] for m in (st.get("prompt_messages") or []) if m.get("role") == "tool"}
        if target_app and (target_app in obs_apps or target_app in ("phone", "sms", "chat_ott", "email") and "contacts" in obs_apps):
            synth_ask_steps.append((str(st.get("sample_id") or "synth_post"), st))

    rng.shuffle(real_ask_steps)
    rng.shuffle(synth_ask_steps)
    ask_steps = real_ask_steps + synth_ask_steps
    rng.shuffle(clear_act_steps)

    pairs: list[dict[str, Any]] = []
    ask_count = 0
    for sid, st in ask_steps:
        if ask_count >= ask_target:
            break
        ask_call = next((c for c in (st.get("step_calls") or []) if c.get("name") == ASK_MISSING_INFO), None)
        if not ask_call:
            continue
        ask_args = ask_call.get("arguments") or {}
        tname = str(ask_args.get("tool_name") or "")
        params = list(ask_args.get("params") or [])
        if not tname or tname not in catalog:
            continue
        prompt_msgs = st.get("prompt_messages") or []
        sample_ctx = st.get("sample_context")
        # Build hardest premature ACT: bind real candidate #1 returned in <tool_response>
        rej_args = _build_plausible_tool_args(tname, catalog, sample_ctx, prompt_msgs, rng)
        rej_call = {"name": tname, "arguments": rej_args}
        pair = _step_to_pair(
            st,
            rejected_calls=[rej_call],
            rejected_final_text="",
            rejected_think=None,
            category="post_lookup_ask_vs_guess",
            sample_id=sid,
            catalog=catalog,
            rng=rng,
            max_tokens=max_tokens,
            max_prompt_tokens=max_prompt_tokens,
            tokenizer=tokenizer,
            rej_plan_override=f"Tự chọn ứng viên đầu tiên từ kết quả tra cứu ({_summarize_rejected_calls([rej_call])}) mà không hỏi xác nhận {params}.",
            rej_boundary_override="Quyết định: Đủ điều kiện thực thi (Act). Mục tiêu còn lại: Không còn.",
        )
        if pair:
            pairs.append(pair)
            ask_count += 1

    clear_count = 0
    for sample, st, meta in clear_act_steps:
        if clear_count >= clear_target:
            break
        calls = st.get("step_calls") or []
        first_c = calls[0]
        tname = str(first_c.get("name") or "")
        c_args = dict(first_c.get("arguments") or {})
        if not tname or not c_args or tname == ASK_MISSING_INFO:
            continue
        # Reject by falsely asking for an ID or primary parameter that IS already clear in obs/context
        id_keys = [k for k in c_args if k == "id" or k.endswith("_id")]
        false_param = id_keys[0] if id_keys else next(iter(c_args.keys()))
        rej_ask = {
            "name": ASK_MISSING_INFO,
            "arguments": {"tool_name": tname, "params": [false_param]},
        }
        enriched = {**st, **meta}
        pair = _step_to_pair(
            enriched,
            rejected_calls=[rej_ask],
            rejected_final_text="",
            rejected_think=None,
            category="post_lookup_clear_act_vs_false_ask",
            sample_id=sample.id,
            catalog=catalog,
            rng=rng,
            max_tokens=max_tokens,
            max_prompt_tokens=max_prompt_tokens,
            tokenizer=tokenizer,
            rej_plan_override=f"Gọi sys.ask_missing_info cho '{tname}' về ['{false_param}'] dù dữ liệu tra cứu đã đầy đủ.",
            rej_boundary_override="Quyết định: Cần hỏi lại (Ask Missing Info) trước khi thực hiện. Mục tiêu còn lại: Không còn.",
        )
        if pair:
            pairs.append(pair)
            clear_count += 1

    return pairs


def generate_anchor_binding_dpo_pairs(
    samples: list[Any],
    catalog: dict[str, dict[str, Any]],
    *,
    root: str | Path,
    split: str,
    target_count: int,
    rng: random.Random,
    max_tokens: int,
    max_prompt_tokens: int = 14000,
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    """Anchor-binding & business-rule DPO pairs: `from_account_id`, `bill_payment.pay.source`, sibling obs IDs, and subtool swaps."""
    acct_target = int(round(target_count * 0.28))
    source_target = int(round(target_count * 0.28))
    sibling_target = int(round(target_count * 0.28))
    subtool_target = max(1, target_count - acct_target - source_target - sibling_target)

    acct_steps: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []
    source_steps: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []
    sibling_steps: list[tuple[str, dict[str, Any]]] = []
    subtool_steps: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []

    for sample in samples:
        traj = build_sft_trajectory(sample, catalog)
        steps = unroll_trajectory_steps(traj)
        meta = _build_step_metadata(sample, catalog)
        for st in steps:
            calls = st.get("step_calls") or []
            if st.get("step_category") != "act" or not calls:
                continue
            first_c = calls[0]
            tname = str(first_c.get("name") or "")
            args = first_c.get("arguments") or {}
            if "from_account_id" in args:
                acct_steps.append((sample, st, meta))
            if tname == "bill_payment.pay" and "source" in args:
                source_steps.append((sample, st, meta))
            if any(k == "id" or k.endswith("_id") for k in args):
                sibling_steps.append((sample.id, {**st, **meta}))
            if tname in _SUBTOOL_SWAPS:
                installed = set(_extract_tool_names(st.get("tools")))
                if any(cand in installed for cand in _SUBTOOL_SWAPS[tname]):
                    subtool_steps.append((sample, st, meta))

    synth_ab = generate_anchor_binding_samples(samples, catalog, count=max(40, sibling_target), seed=rng.randint(1, 99999))
    for st in synth_ab:
        sibling_steps.append((str(st.get("sample_id") or "synth_ab"), st))

    rng.shuffle(acct_steps)
    rng.shuffle(source_steps)
    rng.shuffle(sibling_steps)
    rng.shuffle(subtool_steps)

    pairs: list[dict[str, Any]] = []

    # 1. `from_account_id` checking/savings / balance-rule counterfactuals
    acct_cnt = 0
    for sample, st, meta in acct_steps:
        if acct_cnt >= acct_target:
            break
        calls = copy.deepcopy(st["step_calls"])
        gold_val = str((calls[0].get("arguments") or {}).get("from_account_id") or "")
        if not gold_val:
            continue
        obs_ids = _extract_candidate_ids_from_obs(st.get("prompt_messages") or [])
        alt_pool = [
            cand
            for k in ("account_id", "from_account_id", "id")
            for cand in obs_ids.get(k, [])
            if cand != gold_val
        ]
        wrong_acct = alt_pool[0] if alt_pool else _mutate_entity_id(gold_val, rng)
        calls[0]["arguments"]["from_account_id"] = wrong_acct
        pair = _step_to_pair(
            {**st, **meta},
            rejected_calls=calls,
            rejected_final_text="",
            rejected_think=None,
            category="from_account_balance_rule",
            sample_id=sample.id,
            catalog=catalog,
            rng=rng,
            max_tokens=max_tokens,
            max_prompt_tokens=max_prompt_tokens,
            tokenizer=tokenizer,
            rej_plan_override=f"Chọn sai tài khoản nguồn from_account_id='{wrong_acct}' thay vì tài khoản phù hợp với số dư/loại tài khoản yêu cầu.",
        )
        if pair:
            pairs.append(pair)
            acct_cnt += 1

    # 2. `bill_payment.pay.source` (`bank` <-> `ewallet`)
    src_cnt = 0
    for sample, st, meta in source_steps:
        if src_cnt >= source_target:
            break
        calls = copy.deepcopy(st["step_calls"])
        gold_src = str((calls[0].get("arguments") or {}).get("source") or "")
        wrong_src = "ewallet" if gold_src == "bank" else "bank"
        calls[0]["arguments"]["source"] = wrong_src
        pair = _step_to_pair(
            {**st, **meta},
            rejected_calls=calls,
            rejected_final_text="",
            rejected_think=None,
            category="bill_source_choice",
            sample_id=sample.id,
            catalog=catalog,
            rng=rng,
            max_tokens=max_tokens,
            max_prompt_tokens=max_prompt_tokens,
            tokenizer=tokenizer,
            rej_plan_override=f"Chọn sai nguồn thanh toán bill_payment.pay(source='{wrong_src}') thay vì '{gold_src}'.",
        )
        if pair:
            pairs.append(pair)
            src_cnt += 1

    # 3. Real sibling entity ID from <tool_response> (or mutated ID fallback)
    sib_cnt = 0
    for sid, st in sibling_steps:
        if sib_cnt >= sibling_target:
            break
        calls = copy.deepcopy(st.get("step_calls") or [])
        if not calls:
            continue
        obs_ids = _extract_candidate_ids_from_obs(st.get("prompt_messages") or [])
        mutated = False
        wrong_desc = ""
        for c in calls:
            args = c.get("arguments") or {}
            for k, v in list(args.items()):
                if (k == "id" or k.endswith("_id")) and isinstance(v, str) and v.strip():
                    alt_pool = [
                        cand
                        for key in (k, "id")
                        for cand in obs_ids.get(key, [])
                        if cand != v and (len(v) <= 4 or cand[:3] == v[:3] or key == k)
                    ]
                    wrong_v = alt_pool[0] if alt_pool else _mutate_entity_id(v, rng)
                    args[k] = wrong_v
                    wrong_desc = f"{k}='{wrong_v}'"
                    mutated = True
                    break
            if mutated:
                break
        if not mutated:
            continue
        pair = _step_to_pair(
            st,
            rejected_calls=calls,
            rejected_final_text="",
            rejected_think=None,
            category="obs_sibling_entity_id",
            sample_id=sid,
            catalog=catalog,
            rng=rng,
            max_tokens=max_tokens,
            max_prompt_tokens=max_prompt_tokens,
            tokenizer=tokenizer,
            rej_plan_override=f"Gọi {calls[0]['name']} nhưng lấy nhầm mã {wrong_desc} thay vì đúng đối tượng từ kết quả tra cứu.",
        )
        if pair:
            pairs.append(pair)
            sib_cnt += 1

    # 4. Sub-tool disambiguation (`sms.send` vs `chat_ott.send_message`, `notes.create` vs `reminders.create`, etc.)
    sub_cnt = 0
    for sample, st, meta in subtool_steps:
        if sub_cnt >= subtool_target:
            break
        calls = copy.deepcopy(st["step_calls"])
        orig_name = str(calls[0].get("name") or "")
        installed = set(_extract_tool_names(st.get("tools")))
        cand_swaps = [t for t in _SUBTOOL_SWAPS.get(orig_name, []) if t in installed]
        if not cand_swaps:
            continue
        swap_tool = rng.choice(cand_swaps)
        orig_args = dict(calls[0].get("arguments") or {})
        swap_props = ((catalog.get(swap_tool) or {}).get("parameters") or {}).get("properties") or {}
        mapped_partial = {k: v for k, v in orig_args.items() if k in swap_props}
        rej_args = _build_plausible_tool_args(
            swap_tool,
            catalog,
            meta.get("sample_context"),
            st.get("prompt_messages") or [],
            rng,
            partial_args=mapped_partial,
        )
        rej_call = {"name": swap_tool, "arguments": rej_args}
        pair = _step_to_pair(
            {**st, **meta},
            rejected_calls=[rej_call],
            rejected_final_text="",
            rejected_think=None,
            category="subtool_confusion",
            sample_id=sample.id,
            catalog=catalog,
            rng=rng,
            max_tokens=max_tokens,
            max_prompt_tokens=max_prompt_tokens,
            tokenizer=tokenizer,
            rej_plan_override=f"Chọn nhầm công cụ {swap_tool} thay vì {orig_name} được yêu cầu.",
        )
        if pair:
            pairs.append(pair)
            sub_cnt += 1

    return pairs


def generate_balanced_when2call_step0_pairs(
    samples: list[Any],
    catalog: dict[str, dict[str, Any]],
    *,
    target_count: int,
    rng: random.Random,
    max_tokens: int,
    max_prompt_tokens: int = 14000,
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    """Tri-branch boundary pairs: `step0_ask`, `step0_act_vs_false_ask`, AND `oos_refusal`."""
    ask_target = int(round(target_count * 0.36))
    act_target = int(round(target_count * 0.32))
    oos_target = max(1, target_count - ask_target - act_target)

    pairs: list[dict[str, Any]] = []

    # 1. Step-0 ASK vs plausible schema-filled premature ACT
    real_step0_ask: list[tuple[str, dict[str, Any]]] = []
    act_step0_pool: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []
    oos_pool: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []

    for sample in samples:
        traj = build_sft_trajectory(sample, catalog)
        steps = unroll_trajectory_steps(traj)
        meta = _build_step_metadata(sample, catalog)
        for st in steps:
            sc = st.get("step_category")
            calls = st.get("step_calls") or []
            has_obs = any(m.get("role") == "tool" for m in (st.get("prompt_messages") or []))
            if sc == "oos":
                oos_pool.append((sample, st, meta))
            elif sc == "ask" and not has_obs and calls:
                real_step0_ask.append((sample.id, {**st, **meta}))
            elif sc == "act" and not has_obs and calls:
                act_step0_pool.append((sample, st, meta))

    synth_w2c = generate_when2call_contrastive_samples(samples, catalog, count=max(40, ask_target * 2), seed=rng.randint(1, 99999))
    for st in synth_w2c:
        if any(c.get("name") == ASK_MISSING_INFO for c in (st.get("step_calls") or [])):
            real_step0_ask.append((str(st.get("sample_id") or "synth_w2c"), st))

    rng.shuffle(real_step0_ask)
    rng.shuffle(act_step0_pool)
    rng.shuffle(oos_pool)

    ask_cnt = 0
    for sid, st in real_step0_ask:
        if ask_cnt >= ask_target:
            break
        ask_c = next((c for c in (st.get("step_calls") or []) if c.get("name") == ASK_MISSING_INFO), None)
        if not ask_c:
            continue
        tname = str((ask_c.get("arguments") or {}).get("tool_name") or "")
        params = list((ask_c.get("arguments") or {}).get("params") or [])
        if not tname or tname not in catalog:
            continue
        rej_args = _build_plausible_tool_args(
            tname, catalog, st.get("sample_context"), st.get("prompt_messages") or [], rng
        )
        rej_call = {"name": tname, "arguments": rej_args}
        pair = _step_to_pair(
            st,
            rejected_calls=[rej_call],
            rejected_final_text="",
            rejected_think=None,
            category="when2call_step0_ask",
            sample_id=sid,
            catalog=catalog,
            rng=rng,
            max_tokens=max_tokens,
            max_prompt_tokens=max_prompt_tokens,
            tokenizer=tokenizer,
            rej_plan_override=f"Tự đoán tham số còn thiếu {params} và gọi ngay {tname} thay vì hỏi lại người dùng.",
            rej_boundary_override="Quyết định: Đủ điều kiện thực thi (Act). Mục tiêu còn lại: Không còn.",
        )
        if pair:
            pairs.append(pair)
            ask_cnt += 1

    # 2. Clear Step-0 ACT vs false ASK
    act_cnt = 0
    for sample, st, meta in act_step0_pool:
        if act_cnt >= act_target:
            break
        calls = st.get("step_calls") or []
        first_c = calls[0]
        tname = str(first_c.get("name") or "")
        c_args = first_c.get("arguments") or {}
        if not tname or not c_args:
            continue
        false_param = rng.choice(list(c_args.keys()))
        rej_ask = {"name": ASK_MISSING_INFO, "arguments": {"tool_name": tname, "params": [false_param]}}
        pair = _step_to_pair(
            {**st, **meta},
            rejected_calls=[rej_ask],
            rejected_final_text="",
            rejected_think=None,
            category="when2call_step0_act",
            sample_id=sample.id,
            catalog=catalog,
            rng=rng,
            max_tokens=max_tokens,
            max_prompt_tokens=max_prompt_tokens,
            tokenizer=tokenizer,
            rej_plan_override=f"Gọi sys.ask_missing_info cho '{tname}' về ['{false_param}'] dù thông tin đã rõ trong ngữ cảnh.",
            rej_boundary_override="Quyết định: Cần hỏi lại (Ask Missing Info) trước khi thực hiện. Mục tiêu còn lại: Không còn.",
        )
        if pair:
            pairs.append(pair)
            act_cnt += 1

    # 3. Genuine OOS refusal (`[]`) vs hallucinated tool call or false `sys.ask_missing_info`
    oos_cnt = 0
    for sample, st, meta in oos_pool:
        if oos_cnt >= oos_target:
            break
        installed = _extract_tool_names(st.get("tools"))
        if not installed:
            continue
        pick_tool = rng.choice(installed)
        if rng.random() < 0.7:
            rej_call = {
                "name": pick_tool,
                "arguments": _build_plausible_tool_args(
                    pick_tool, catalog, meta.get("sample_context"), st.get("prompt_messages") or [], rng
                ),
            }
            rej_plan = f"Cố gắng gọi {pick_tool} dù ứng dụng không hỗ trợ yêu cầu ngoài phạm vi."
            rej_bound = "Quyết định: Đủ điều kiện thực thi (Act). Mục tiêu còn lại: Không còn."
        else:
            req = list(((catalog.get(pick_tool) or {}).get("parameters") or {}).get("required") or ["item"])
            rej_call = {"name": ASK_MISSING_INFO, "arguments": {"tool_name": pick_tool, "params": req[:1]}}
            rej_plan = f"Gọi sys.ask_missing_info cho '{pick_tool}' thay vì từ chối yêu cầu ngoài phạm vi."
            rej_bound = "Quyết định: Cần hỏi lại (Ask Missing Info) trước khi thực hiện. Mục tiêu còn lại: Không còn."
        pair = _step_to_pair(
            {**st, **meta},
            rejected_calls=[rej_call],
            rejected_final_text="",
            rejected_think=None,
            category="oos_refusal",
            sample_id=sample.id,
            catalog=catalog,
            rng=rng,
            max_tokens=max_tokens,
            max_prompt_tokens=max_prompt_tokens,
            tokenizer=tokenizer,
            rej_plan_override=rej_plan,
            rej_boundary_override=rej_bound,
        )
        if pair:
            pairs.append(pair)
            oos_cnt += 1

    return pairs


def _shift_datetime_value(val: str, rng: random.Random) -> str | None:
    """Shift ISO datetime by +-1h or +-1 day, or shift ISO date by +-1 day."""
    m = _ISO_DT_RE.match(val.strip())
    if m:
        yr, mo, dy, hr, mn, sc, tz = m.groups()
        ihr = int(hr)
        idy = int(dy)
        if rng.random() < 0.7:
            new_hr = (ihr + rng.choice([-1, 1])) % 24
            sec_part = f":{sc}" if sc is not None else ""
            return f"{yr}-{mo}-{dy}T{new_hr:02d}:{mn}{sec_part}{tz}"
        new_dy = min(28, max(1, idy + rng.choice([-1, 1])))
        if new_dy == idy:
            new_dy = idy + 1
        sec_part = f":{sc}" if sc is not None else ""
        return f"{yr}-{mo}-{new_dy:02d}T{hr}:{mn}{sec_part}{tz}"
    m2 = _ISO_DATE_RE.match(val.strip())
    if m2:
        yr, mo, dy = m2.groups()
        idy = int(dy)
        new_dy = min(28, max(1, idy + rng.choice([-1, 1])))
        if new_dy == idy:
            new_dy = idy + 1
        return f"{yr}-{mo}-{new_dy:02d}"
    return None


def generate_arg_schema_time_fuzzy_pairs(
    samples: list[Any],
    catalog: dict[str, dict[str, Any]],
    *,
    target_count: int,
    rng: random.Random,
    max_tokens: int,
    max_prompt_tokens: int = 14000,
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    """Counterfactual pairs targeting dropped explicit keys, datetime shifts, and verbose fuzzy text."""
    key_target = int(round(target_count * 0.36))
    dt_target = int(round(target_count * 0.32))
    fuzzy_target = max(1, target_count - key_target - dt_target)

    key_steps: list[tuple[Any, dict[str, Any], dict[str, Any], str]] = []
    dt_steps: list[tuple[Any, dict[str, Any], dict[str, Any], str]] = []
    fuzzy_steps: list[tuple[Any, dict[str, Any], dict[str, Any], str]] = []

    for sample in samples:
        traj = build_sft_trajectory(sample, catalog)
        steps = unroll_trajectory_steps(traj)
        meta = _build_step_metadata(sample, catalog)
        for st in steps:
            if st.get("step_category") != "act":
                continue
            calls = st.get("step_calls") or []
            if not calls:
                continue
            first_c = calls[0]
            tname = str(first_c.get("name") or "")
            args = first_c.get("arguments") or {}
            drop_hit = next((k for k in _EXPLICIT_OFTEN_DROPPED_KEYS if k in args), "")
            if drop_hit:
                key_steps.append((sample, st, meta, drop_hit))
            dt_hit = next(
                (k for k in ("time", "start", "end", "due", "date", "date_from", "departure_date") if isinstance(args.get(k), str)),
                "",
            )
            if dt_hit and (_ISO_DT_RE.match(str(args[dt_hit])) or _ISO_DATE_RE.match(str(args[dt_hit]))):
                dt_steps.append((sample, st, meta, dt_hit))
            if tname in ("sms.send", "chat_ott.send_message", "notes.create", "email.send", "calendar.create_event"):
                f_key = "title" if tname == "calendar.create_event" else "content"
                if isinstance(args.get(f_key), str) and args[f_key].strip():
                    fuzzy_steps.append((sample, st, meta, f_key))

    rng.shuffle(key_steps)
    rng.shuffle(dt_steps)
    rng.shuffle(fuzzy_steps)

    pairs: list[dict[str, Any]] = []

    # 1. Missing explicit schema key (`power`, `date_from`, `near`, `mode`, `location`, `source`, ...)
    k_cnt = 0
    for sample, st, meta, drop_k in key_steps:
        if k_cnt >= key_target:
            break
        calls = copy.deepcopy(st["step_calls"])
        calls[0]["arguments"].pop(drop_k, None)
        pair = _step_to_pair(
            {**st, **meta},
            rejected_calls=calls,
            rejected_final_text="",
            rejected_think=None,
            category="missing_explicit_key",
            sample_id=sample.id,
            catalog=catalog,
            rng=rng,
            max_tokens=max_tokens,
            max_prompt_tokens=max_prompt_tokens,
            tokenizer=tokenizer,
            rej_plan_override=f"Gọi {calls[0]['name']} nhưng bỏ sót tham số '{drop_k}' đã được yêu cầu.",
        )
        if pair:
            pairs.append(pair)
            k_cnt += 1

    # 2. Datetime off-by-1h / off-by-1day
    d_cnt = 0
    for sample, st, meta, dt_k in dt_steps:
        if d_cnt >= dt_target:
            break
        calls = copy.deepcopy(st["step_calls"])
        orig_dt = str(calls[0]["arguments"][dt_k])
        bad_dt = _shift_datetime_value(orig_dt, rng)
        if not bad_dt or bad_dt == orig_dt:
            continue
        calls[0]["arguments"][dt_k] = bad_dt
        pair = _step_to_pair(
            {**st, **meta},
            rejected_calls=calls,
            rejected_final_text="",
            rejected_think=None,
            category="datetime_shift",
            sample_id=sample.id,
            catalog=catalog,
            rng=rng,
            max_tokens=max_tokens,
            max_prompt_tokens=max_prompt_tokens,
            tokenizer=tokenizer,
            rej_plan_override=f"Gọi {calls[0]['name']} với mốc thời gian bị lệch {dt_k}='{bad_dt}' thay vì '{orig_dt}'.",
        )
        if pair:
            pairs.append(pair)
            d_cnt += 1

    # 3. Concise gold fuzzy `content`/`title` vs verbose bloated/filler text
    f_cnt = 0
    for sample, st, meta, f_key in fuzzy_steps:
        if f_cnt >= fuzzy_target:
            break
        calls = copy.deepcopy(st["step_calls"])
        orig_txt = str(calls[0]["arguments"][f_key]).strip()
        bloated = f"{rng.choice(_VERBOSE_PREFIXES)}{orig_txt}.{rng.choice(_VERBOSE_SUFFIXES)}"
        calls[0]["arguments"][f_key] = bloated
        pair = _step_to_pair(
            {**st, **meta},
            rejected_calls=calls,
            rejected_final_text="",
            rejected_think=None,
            category="fuzzy_concise_vs_verbose",
            sample_id=sample.id,
            catalog=catalog,
            rng=rng,
            max_tokens=max_tokens,
            max_prompt_tokens=max_prompt_tokens,
            tokenizer=tokenizer,
            rej_plan_override=f"Gọi {calls[0]['name']} nhưng viết {f_key} dài dòng dư thừa thay vì súc tích đúng trọng tâm.",
        )
        if pair:
            pairs.append(pair)
            f_cnt += 1

    return pairs


def _split_train_val_ids(non_dev_ids: list[str], val_ratio: float, rng: random.Random) -> tuple[set[str], set[str]]:
    ids = list(non_dev_ids)
    rng.shuffle(ids)
    val_n = max(1, int(round(len(ids) * val_ratio))) if val_ratio > 0 and len(ids) >= 10 else 0
    val_ids = set(ids[:val_n])
    train_ids = set(ids[val_n:])
    return train_ids, val_ids


def build_dpo_v2_dataset(
    *,
    root: str | Path = "data",
    split: str = "train",
    dev_split_path: str | Path = "configs/splits/dev_v1.json",
    out_dir: str | Path = "data/dpo",
    traces_path: str | Path | None = None,
    per_sample_path: str | Path | None = None,
    val_ratio: float = 0.1,
    max_tokens: int = 15800,
    max_prompt_tokens: int = 14000,
    seed: int = 42,
    use_hf_tokenizer: bool = False,
) -> tuple[Path, Path | None, dict[str, Any]]:
    """Build train/val DPO V2 gzipped JSONL files strictly excluding `dev_v1` IDs."""
    rng = random.Random(seed)
    all_samples = load_split(split, root)
    catalog = load_catalog(split, root)

    dev_path = Path(dev_split_path)
    dev_ids: set[str] = set()
    if dev_path.exists():
        raw_dev = json.loads(dev_path.read_text(encoding="utf-8"))
        if isinstance(raw_dev, dict):
            dev_ids = set(raw_dev.get("dev_ids") or raw_dev.get("ids") or [])
        else:
            dev_ids = set(raw_dev)

    non_dev_samples = [s for s in all_samples if s.id not in dev_ids]
    samples_by_id = {s.id: s for s in non_dev_samples}

    tokenizer = None
    if use_hf_tokenizer:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("OpenBMB/MiniCPM-o-4_5", trust_remote_code=True)
        except Exception:
            tokenizer = None

    train_ids, val_ids = _split_train_val_ids([s.id for s in non_dev_samples], val_ratio, rng)

    def _build_for_subset(sub_samples: list[Any], scale: float) -> list[dict[str, Any]]:
        sub_pairs: list[dict[str, Any]] = []
        sub_by_id = {s.id: s for s in sub_samples}
        if traces_path is not None:
            sub_pairs.extend(
                generate_on_policy_trace_pairs(
                    sub_by_id,
                    catalog,
                    Path(traces_path),
                    Path(per_sample_path) if per_sample_path else None,
                    root=root,
                    split=split,
                    rng=rng,
                    max_tokens=max_tokens,
                    max_prompt_tokens=max_prompt_tokens,
                    tokenizer=tokenizer,
                )
            )
        sub_pairs.extend(
            generate_horizon_stop_vs_overcall_pairs(
                sub_samples,
                catalog,
                target_count=max(4, int(round(130 * scale))),
                rng=rng,
                max_tokens=max_tokens,
                max_prompt_tokens=max_prompt_tokens,
                tokenizer=tokenizer,
            )
        )
        sub_pairs.extend(
            generate_anti_early_stop_pairs(
                sub_samples,
                catalog,
                root=root,
                split=split,
                target_count=max(4, int(round(130 * scale))),
                rng=rng,
                max_tokens=max_tokens,
                max_prompt_tokens=max_prompt_tokens,
                tokenizer=tokenizer,
            )
        )
        sub_pairs.extend(
            generate_post_lookup_dpo_pairs(
                sub_samples,
                catalog,
                root=root,
                split=split,
                target_count=max(6, int(round(210 * scale))),
                rng=rng,
                max_tokens=max_tokens,
                max_prompt_tokens=max_prompt_tokens,
                tokenizer=tokenizer,
            )
        )
        sub_pairs.extend(
            generate_anchor_binding_dpo_pairs(
                sub_samples,
                catalog,
                root=root,
                split=split,
                target_count=max(6, int(round(220 * scale))),
                rng=rng,
                max_tokens=max_tokens,
                max_prompt_tokens=max_prompt_tokens,
                tokenizer=tokenizer,
            )
        )
        sub_pairs.extend(
            generate_balanced_when2call_step0_pairs(
                sub_samples,
                catalog,
                target_count=max(6, int(round(195 * scale))),
                rng=rng,
                max_tokens=max_tokens,
                max_prompt_tokens=max_prompt_tokens,
                tokenizer=tokenizer,
            )
        )
        sub_pairs.extend(
            generate_arg_schema_time_fuzzy_pairs(
                sub_samples,
                catalog,
                target_count=max(6, int(round(195 * scale))),
                rng=rng,
                max_tokens=max_tokens,
                max_prompt_tokens=max_prompt_tokens,
                tokenizer=tokenizer,
            )
        )
        rng.shuffle(sub_pairs)
        return sub_pairs

    train_samples = [samples_by_id[i] for i in sorted(train_ids)]
    val_samples = [samples_by_id[i] for i in sorted(val_ids)]
    total_n = max(1, len(non_dev_samples))
    train_pairs = _build_for_subset(train_samples, len(train_samples) / total_n)
    val_pairs = _build_for_subset(val_samples, len(val_samples) / total_n) if val_samples else []

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    train_gz = out_path / "minicpm5_dpo_v2_train.jsonl.gz"
    val_gz: Path | None = out_path / "minicpm5_dpo_v2_val.jsonl.gz" if val_pairs else None

    with gzip.open(train_gz, "wt", encoding="utf-8") as f:
        for row in train_pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if val_gz is not None:
        with gzip.open(val_gz, "wt", encoding="utf-8") as f:
            for row in val_pairs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    all_rows = train_pairs + val_pairs
    stats = {
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "train_categories": dict(Counter(r["category"] for r in train_pairs)),
        "val_categories": dict(Counter(r["category"] for r in val_pairs)),
        "max_est_prompt_tokens": max((r["est_prompt_tokens"] for r in all_rows), default=0),
        "max_est_tokens": max((r["est_tokens"] for r in all_rows), default=0),
    }
    print(f"DPO V2 dataset written: {json.dumps(stats, ensure_ascii=False)}", flush=True)
    return train_gz, val_gz, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dev-split", default="configs/splits/dev_v1.json")
    parser.add_argument("--out-dir", default="data/dpo")
    parser.add_argument("--traces", default=None)
    parser.add_argument("--per-sample", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=15800)
    parser.add_argument("--max-prompt-tokens", type=int, default=14000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-hf-tokenizer", action="store_true")
    args = parser.parse_args()
    build_dpo_v2_dataset(
        root=args.root,
        split=args.split,
        dev_split_path=args.dev_split,
        out_dir=args.out_dir,
        traces_path=args.traces,
        per_sample_path=args.per_sample,
        val_ratio=args.val_ratio,
        max_tokens=args.max_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        seed=args.seed,
        use_hf_tokenizer=args.use_hf_tokenizer,
    )


if __name__ == "__main__":
    main()
