"""ReAct tool-calling loop with configurable pre-flight schema, x-ref, and ask_missing_info guardrails."""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.agent.boundary_verifier import BoundaryVerifier
from src.agent.history import build_messages, truncate_tool_result
from src.agent.llm_client import ChatBackend
from src.agent.prompts import format_tool_call_hint
from src.agent.tool_caller import (
    extract_final_text,
    has_malformed_tool_call,
    parse_tool_calls,
    sanitize_tool_call,
)
from src.agent.tool_selector import estimate_tools_tokens, select_tools
from src.core.config import GuardrailConfig
from src.core.schemas import ASK_MISSING_INFO, Sample, ToolCall
from src.env.safe_env import SafeEnv

_GENERIC_REL_WORDS = {
    "đồng",
    "nghiệp",
    "bạn",
    "cũ",
    "học",
    "em",
    "anh",
    "chị",
    "cô",
    "chú",
    "bác",
    "ông",
    "bà",
    "họ",
    "bên",
    "nhóm",
    "phòng",
    "phụ",
    "trách",
    "người",
    "quen",
}
_NUMBER_WORD_RE = re.compile(
    r"\d|\b(một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|chục|trăm|nghìn|ngàn|triệu|củ|lít)\b",
    re.IGNORECASE,
)


@dataclass
class AgentRunResult:
    id: str
    tool_calls: list[dict[str, Any]]
    state_hash: str
    steps: int
    repairs: int
    latency_s: float
    tools_tokens_est: int
    final_text: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)

    def to_submission_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool_calls": self.tool_calls,
            "state_hash": self.state_hash,
        }


def _call_signature(call: ToolCall) -> tuple[str, str]:
    return (
        call.name,
        json.dumps(call.arguments, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
    )


_LIST_KEY_TO_ID_PARAM: dict[str, str] = {
    "notes": "note_id",
    "devices": "device_id",
    "scenes": "scene_id",
    "alarms": "alarm_id",
    "tasks": "task_id",
}
_TOOL_TO_DEFAULT_ID_PARAM: dict[str, str] = {
    "notes.list": "note_id",
    "alarm.list": "alarm_id",
    "tasks.list": "task_id",
}
_TARGETED_ENTITY_ID_PARAMS: tuple[str, ...] = ("note_id", "device_id", "scene_id", "alarm_id", "task_id")


def _canonical_phone_variants(digits: str) -> set[str]:
    """Return normalized numeric phone variants (+84/84 <-> 0) for matching."""
    if not digits:
        return set()
    variants = {digits}
    if digits.startswith("84") and len(digits) >= 10:
        variants.add("0" + digits[2:])
    elif digits.startswith("0") and len(digits) >= 9:
        variants.add("84" + digits[1:])
    return variants


def _extract_phone_candidates(text: str) -> set[str]:
    """Extract normalized phone candidates from numeric strings and spoken Vietnamese digits."""
    import re
    from src.asr.itn import _DIGITS

    if not text:
        return set()
    low = text.lower()
    out: set[str] = set()

    def _add_if_phone(digits: str) -> None:
        if 8 <= len(digits) <= 13 and (digits.startswith("0") or digits.startswith("84")):
            out.update(_canonical_phone_variants(digits))

    for m in re.finditer(r"(?<!\d)(?:\+?84|0)\d{7,12}(?!\d)", low):
        _add_if_phone(re.sub(r"\D", "", m.group(0)))

    for m in re.finditer(r"(?<!\d)(?:\+?84|0)\d{1,3}(?:[.\s-]\d{2,4}){2,4}(?!\d)", low):
        _add_if_phone(re.sub(r"\D", "", m.group(0)))

    tokens = re.findall(r"[^\s,.;:!?()\[\]\"']+", low)
    cur_digits: list[str] = []
    for tok in tokens:
        if tok in _DIGITS:
            cur_digits.append(str(_DIGITS[tok]))
        elif tok.isdigit():
            cur_digits.append(tok)
        else:
            if cur_digits:
                _add_if_phone("".join(cur_digits))
                cur_digits.clear()
    if cur_digits:
        _add_if_phone("".join(cur_digits))

    return out


def _extract_entities_by_tool(
    tool_name: str,
    obj: Any,
    pools: dict[str, dict[str, dict[str, Any]]],
) -> None:
    """Populate entity pools keyed by ID param (note_id, device_id, scene_id, alarm_id, task_id)."""
    default_param = _TOOL_TO_DEFAULT_ID_PARAM.get(tool_name)

    def _add_item(id_param: str, item: Any) -> None:
        if not isinstance(item, dict):
            return
        eid = item.get("id") or item.get(id_param)
        if isinstance(eid, str) and eid:
            pools[id_param][eid] = item

    if isinstance(obj, dict):
        if default_param and isinstance(obj.get("id"), str):
            _add_item(default_param, obj)
        for k, v in obj.items():
            if k in _LIST_KEY_TO_ID_PARAM and isinstance(v, list):
                param = _LIST_KEY_TO_ID_PARAM[k]
                for item in v:
                    _add_item(param, item)
            elif default_param and isinstance(v, list):
                for item in v:
                    _add_item(default_param, item)
            elif isinstance(v, (dict, list)):
                _extract_entities_by_tool(tool_name, v, pools)
    elif isinstance(obj, list) and default_param:
        for item in obj:
            _add_item(default_param, item)


def _format_schema_hint(tool_name: str, catalog: dict[str, dict[str, Any]]) -> str:
    """Build an explicit parameter hint for schema_repair turns."""
    schema = (catalog.get(tool_name) or {}).get("parameters") or {}
    props = schema.get("properties") or {}
    required = schema.get("required") or []
    req_str = ", ".join(required) if required else "không có"
    opt_list = [p for p in props if p not in required]
    opt_str = ", ".join(opt_list) if opt_list else "không có"
    return (
        f"Sửa lại tool/tham số hoặc gọi tool tra cứu ID trước rồi gọi lại. "
        f"Schema của `{tool_name}`: required=[{req_str}], optional=[{opt_str}]."
    )


def _generate_with_optional_schema(
    llm: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    force_tool_schema: bool = False,
) -> str:
    """Call llm.generate with force_tool_schema when the backend accepts that keyword."""
    import inspect

    gen_fn = llm.generate
    try:
        sig = inspect.signature(gen_fn)
        accepts_flag = "force_tool_schema" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
    except (TypeError, ValueError):
        accepts_flag = False

    if accepts_flag:
        return gen_fn(messages, tools, force_tool_schema=force_tool_schema)
    return gen_fn(messages, tools)


_LIT_EMAIL_RE = re.compile(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}")


def _extract_email_candidates(text: str) -> set[str]:
    """Extract literal and Vietnamese/English dictated email addresses (lower-cased)."""
    if not text or not isinstance(text, str):
        return set()
    low = text.lower()
    found = set(_LIT_EMAIL_RE.findall(low))
    if any(tok in low for tok in ("a còng", "a cồng", "a móc", "a-còng", " at ", "@")):
        norm = low
        norm = re.sub(r"\s*(?:a[\s\-]*c[oòóỏõọồốổỗộ]ng|a[\s\-]*m[oóòỏõọ]c)\s*", "@", norm)
        norm = re.sub(r"\s+at\s+(?=[a-z0-9._\-]+\s+(?:chấm|dot)\s+[a-z]{2,})", "@", norm)
        norm = re.sub(r"\s+(?:chấm|dot)\s+", ".", norm)
        norm = re.sub(r"\s+gạch\s*dưới\s+", "_", norm)
        norm = re.sub(r"\s+gạch\s*ngang\s+", "-", norm)
        norm = re.sub(r"\s*@\s*", "@", norm)
        norm = re.sub(r"\s*\.\s*", ".", norm)
        found.update(_LIT_EMAIL_RE.findall(norm))
    return found


def _collect_ids_from_result(
    app_name: str,
    obj: Any,
    by_app: dict[str, set[str]],
    current_collection: str | None = None,
) -> None:
    coll = current_collection or app_name
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and (k == "id" or k.endswith("_id")):
                by_app[app_name].add(v)
                by_app[coll].add(v)
                if k == "contact_id":
                    by_app["contacts"].add(v)
                    by_app["contacts.contacts"].add(v)
                elif k == "device_id":
                    by_app["smart_home.devices"].add(v)
                elif k == "scene_id":
                    by_app["smart_home.scenes"].add(v)
                elif k == "track_id":
                    by_app["music.tracks"].add(v)
                elif k == "playlist_id":
                    by_app["music.playlists"].add(v)
                elif k == "flight_id":
                    by_app["flights.flights"].add(v)
                elif k == "booking_id":
                    by_app["flights.bookings"].add(v)
                elif k == "bill_id":
                    by_app["bill_payment.bills"].add(v)
                elif k == "event_id":
                    by_app["calendar.events"].add(v)
            elif isinstance(v, str):
                if k in ("phone", "phone_number"):
                    by_app["phones"].add(v)
                    import re

                    digits = re.sub(r"\D", "", v)
                    if digits:
                        by_app["phones"].update(_canonical_phone_variants(digits))
                by_app["phones"].update(_extract_phone_candidates(v))
                if k == "email" or "@" in v:
                    by_app["emails"].update(_extract_email_candidates(v))
            elif isinstance(v, list):
                nested_coll = f"{app_name}.{k}"
                for item in v:
                    _collect_ids_from_result(app_name, item, by_app, current_collection=nested_coll)
            elif isinstance(v, dict):
                _collect_ids_from_result(app_name, v, by_app, current_collection=coll)
    elif isinstance(obj, list):
        for item in obj:
            _collect_ids_from_result(app_name, item, by_app, current_collection=coll)
    elif isinstance(obj, str):
        by_app["phones"].update(_extract_phone_candidates(obj))
        if "@" in obj:
            by_app["emails"].update(_extract_email_candidates(obj))


def _is_rel_mentioned(rel: str, text_low: str) -> bool:
    rel_words = [
        w
        for w in re.findall(r"\w+", (rel or "").lower())
        if w not in _GENERIC_REL_WORDS and len(w) > 1
    ]
    t_words = set(re.findall(r"\w+", text_low))
    return bool(rel_words) and any(w in t_words for w in rel_words)


def _collect_ambiguous_from_result(
    tool_name: str,
    res: Any,
    text_low: str,
    ambig_contact_ids: set[str],
    ambig_phones: set[str],
    ambig_emails: set[str],
    ambig_chat_ids: set[str],
) -> None:
    payload = res.get("result") if isinstance(res, dict) and isinstance(res.get("result"), dict) else res
    if not isinstance(payload, dict):
        return

    if tool_name == "contacts.search" and isinstance(payload.get("contacts"), list):
        by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in payload["contacts"]:
            if isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"].strip():
                by_name[item["name"].strip().lower()].append(item)
        for nm, group in by_name.items():
            if len(group) >= 2 and nm in text_low:
                disambiguated = any(
                    _is_rel_mentioned(str(g.get("relation") or ""), text_low)
                    or (isinstance(g.get("phone"), str) and g["phone"] in text_low)
                    or (isinstance(g.get("email"), str) and g["email"].lower() in text_low)
                    for g in group
                )
                if not disambiguated:
                    for g in group:
                        if isinstance(g.get("id"), str):
                            ambig_contact_ids.add(g["id"])
                        if isinstance(g.get("phone"), str):
                            ambig_phones.add(g["phone"])
                        if isinstance(g.get("email"), str):
                            ambig_emails.add(g["email"].lower())

    elif tool_name == "chat_ott.list_chats" and isinstance(payload.get("chats"), list):
        by_name_chat: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in payload["chats"]:
            if (
                isinstance(item, dict)
                and item.get("type") == "direct"
                and isinstance(item.get("name"), str)
                and item["name"].strip()
            ):
                by_name_chat[item["name"].strip().lower()].append(item)
        for nm, group in by_name_chat.items():
            if len(group) >= 2 and nm in text_low:
                for g in group:
                    if isinstance(g.get("id"), str):
                        ambig_chat_ids.add(g["id"])


def _apply_ask_guard(
    call: ToolCall,
    text_low: str,
    ambig_contact_ids: set[str],
    ambig_phones: set[str],
    ambig_emails: set[str],
    ambig_chat_ids: set[str],
) -> ToolCall:
    if call.name == ASK_MISSING_INFO:
        return call

    # 1. Vague transfer amount ("chút tiền", "ít tiền")
    if call.name in ("ewallet.transfer", "banking.transfer") and "amount" in call.arguments:
        if "chút tiền" in text_low or "ít tiền" in text_low:
            return ToolCall(
                name=ASK_MISSING_INFO,
                arguments={"tool_name": call.name, "params": ["amount"]},
            )

    # 2. Ambiguous duplicate contact from contacts.search
    cid = call.arguments.get("contact_id")
    if isinstance(cid, str) and cid in ambig_contact_ids:
        return ToolCall(
            name=ASK_MISSING_INFO,
            arguments={"tool_name": call.name, "params": ["contact_id"]},
        )
    for pk in ("phone", "phone_number"):
        pv = call.arguments.get(pk)
        if isinstance(pv, str) and pv in ambig_phones:
            return ToolCall(
                name=ASK_MISSING_INFO,
                arguments={"tool_name": call.name, "params": ["contact_id"]},
            )
    if call.name in ("email.create_draft", "email.send"):
        to_val = call.arguments.get("to")
        emails = (
            [str(x).lower() for x in to_val]
            if isinstance(to_val, list)
            else ([to_val.lower()] if isinstance(to_val, str) else [])
        )
        if any(e in ambig_emails for e in emails):
            return ToolCall(
                name=ASK_MISSING_INFO,
                arguments={"tool_name": call.name, "params": ["to"]},
            )

    # 3. Ambiguous duplicate direct chat from chat_ott.list_chats
    if call.name == "chat_ott.send_message":
        chat_id = call.arguments.get("chat_id")
        if isinstance(chat_id, str) and chat_id in ambig_chat_ids:
            return ToolCall(
                name=ASK_MISSING_INFO,
                arguments={"tool_name": "chat_ott.send_message", "params": ["chat_id"]},
            )

    return call


def _init_known_ids(
    sample: Sample,
    catalog: dict[str, dict[str, Any]],
    history_texts: dict[int, str] | None = None,
) -> dict[str, set[str]]:
    by_app: dict[str, set[str]] = defaultdict(set)
    ctx_profile = ((getattr(sample, "context", None) or {}).get("user_profile") or {})
    if isinstance(ctx_profile, dict):
        for pv in ctx_profile.values():
            if isinstance(pv, str) and "@" in pv:
                by_app["emails"].update(_extract_email_candidates(pv))
    for idx, turn in enumerate(getattr(sample, "history", []) or []):
        turn_text = (
            (history_texts or {}).get(idx)
            or getattr(turn, "audio_text", None)
            or getattr(turn, "text", "")
            or ""
        )
        if turn_text:
            by_app["phones"].update(_extract_phone_candidates(turn_text))
            by_app["emails"].update(_extract_email_candidates(turn_text))
        for i, call in enumerate(getattr(turn, "tool_calls", []) or []):
            app = call.name.split(".", 1)[0]
            props = ((catalog.get(call.name) or {}).get("parameters") or {}).get("properties") or {}
            for k, v in call.arguments.items():
                if isinstance(v, str):
                    xref = (props.get(k) or {}).get("x-ref")
                    if xref:
                        by_app[xref].add(v)
                        by_app[xref.split(".", 1)[0]].add(v)
                    elif k in ("phone", "phone_number"):
                        by_app["phones"].add(v)
                    elif k == "email" or "@" in v:
                        by_app["emails"].update(_extract_email_candidates(v))
                    else:
                        by_app[app].add(v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, str) and (k in ("to", "cc", "bcc", "emails") or "@" in item):
                            by_app["emails"].update(_extract_email_candidates(item))
            if i < len(getattr(turn, "tool_results", []) or []):
                _collect_ids_from_result(app, turn.tool_results[i], by_app)
    return by_app


def _xref_error(
    call: ToolCall,
    catalog: dict[str, dict[str, Any]],
    by_app: dict[str, set[str]],
    text_low: str = "",
) -> str | None:
    import re

    if call.name in ("email.send", "email.create_draft"):
        known_emails = set(by_app.get("emails", set())) | _extract_email_candidates(text_low)
        for field in ("to", "cc", "bcc"):
            raw_val = call.arguments.get(field)
            if not raw_val:
                continue
            vals = raw_val if isinstance(raw_val, list) else ([raw_val] if isinstance(raw_val, str) else [])
            for item in vals:
                if not isinstance(item, str) or not item.strip():
                    continue
                if item.strip().lower() not in known_emails:
                    return (
                        f"sai tham số: {field}='{item}' không có trong câu nói, lịch sử hay kết quả tra cứu. "
                        f"Không được tự bịa địa chỉ email; hãy gọi contacts.search hoặc email.list/email.read trước để lấy đúng email."
                    )

    props = ((catalog.get(call.name) or {}).get("parameters") or {}).get("properties") or {}
    for k, v in call.arguments.items():
        if not isinstance(v, str):
            continue
        if k in ("phone", "phone_number"):
            known_phones = by_app.get("phones", set())
            if v in known_phones or v.lower() in text_low:
                continue
            v_digits = re.sub(r"\D", "", v)
            v_variants = _canonical_phone_variants(v_digits) if v_digits else set()
            all_known = set(known_phones) | _extract_phone_candidates(text_low)
            if v_variants & all_known:
                continue
            is_suffix_correction = any(
                marker in text_low for marker in ("đuôi", "nhầm", "sửa", "đổi số", "chỉnh")
            )
            if is_suffix_correction and v_variants:
                known_digit_forms = {
                    cand
                    for p in all_known
                    for cand in _canonical_phone_variants(re.sub(r"\D", "", p))
                }
                if any(
                    len(vv) >= 8
                    and len(kp) == len(vv)
                    and (kp[:-2] == vv[:-2] or kp[:-3] == vv[:-3])
                    for vv in v_variants
                    for kp in known_digit_forms
                ):
                    continue
            return (
                f"sai tham số: {k}='{v}' không có trong câu nói hay danh bạ đã tra cứu. "
                f"Hãy gọi contacts.search trước để lấy contact_id hoặc số điện thoại thực tế."
            )
        xref = (props.get(k) or {}).get("x-ref")
        if not xref:
            continue
        ref_app = xref.split(".", 1)[0]

        # Check cross-collection clash within same app
        if xref == "smart_home.devices" and v in by_app.get("smart_home.scenes", set()) and v not in by_app.get("smart_home.devices", set()):
            return f"sai tham số: {k}='{v}' là mã ngữ cảnh (scene_id), không phải thiết bị (device_id). Hãy gọi smart_home.get_status để lấy đúng device_id."
        if xref == "smart_home.scenes" and v in by_app.get("smart_home.devices", set()) and v not in by_app.get("smart_home.scenes", set()):
            return f"sai tham số: {k}='{v}' là mã thiết bị (device_id), không phải ngữ cảnh (scene_id). Hãy gọi smart_home.get_status để lấy đúng scene_id."
        if xref == "music.tracks" and v in by_app.get("music.playlists", set()) and v not in by_app.get("music.tracks", set()):
            return f"sai tham số: {k}='{v}' là mã danh sách phát (playlist_id), không phải bài hát (track_id). Hãy gọi music.list_tracks để lấy đúng track_id."
        if xref == "music.playlists" and v in by_app.get("music.tracks", set()) and v not in by_app.get("music.playlists", set()):
            return f"sai tham số: {k}='{v}' là mã bài hát (track_id), không phải danh sách phát (playlist_id). Hãy gọi music.list_playlists để lấy đúng playlist_id."
        if xref == "flights.flights" and v in by_app.get("flights.bookings", set()) and v not in by_app.get("flights.flights", set()):
            return f"sai tham số: {k}='{v}' là mã đặt chỗ (booking_id), không phải mã chuyến bay (flight_id)."
        if xref == "flights.bookings" and v in by_app.get("flights.flights", set()) and v not in by_app.get("flights.bookings", set()):
            return f"sai tham số: {k}='{v}' là mã chuyến bay (flight_id), không phải mã đặt chỗ (booking_id). Hãy gọi flights.list_bookings để tra cứu."

        # Existence check: either exact collection match or app match
        has_id_in_coll = (v in by_app.get(xref, set())) or (v in by_app.get(ref_app, set()))
        if not has_id_in_coll:
            if call.name == "banking.transfer" and (v in by_app.get("bill_payment", set()) or v in by_app.get("bill_payment.bills", set())):
                return (
                    f"sai tham số: {k}='{v}' là mã hóa đơn (bill_id), không phải người thụ hưởng ngân hàng. "
                    f"Để thanh toán hóa đơn bằng ngân hàng, hãy gọi bill_payment.pay(bill_id='{v}', source='bank'). "
                    f"Nếu chuyển khoản cho người thụ hưởng, hãy gọi banking.list_beneficiaries trước."
                )
            desc = (props.get(k) or {}).get("description") or f"tool đọc của {ref_app}"
            return (
                f"sai tham số: {k}='{v}' chưa có trong kết quả tra cứu ({desc}). "
                f"Không được tự đoán ID; hãy gọi tool đọc/danh sách của {ref_app} trước để lấy đúng ID thực tế."
            )
    return None


def _preflight_validate(
    call: ToolCall,
    env: SafeEnv,
    catalog: dict[str, dict[str, Any]],
    by_app: dict[str, set[str]],
    text_low: str = "",
    *,
    enable_xref_guard: bool = True,
) -> dict[str, Any]:
    val = env.validate(call.name, call.arguments)
    if not val.get("ok", False):
        return val
    if enable_xref_guard:
        xerr = _xref_error(call, catalog, by_app, text_low=text_low)
        if xerr is not None:
            return {"ok": False, "error": xerr}
    return {"ok": True, "error": None}


@dataclass
class BatchReactItem:
    sample: Sample
    env: SafeEnv
    catalog: dict[str, dict[str, Any]]
    current_text: str | None = None
    history_texts: dict[int, str] | None = None
    asr_phonetic_diff: str | None = None
    t_start: float | None = None


_CONTACT_HONORIFIC_RE = re.compile(
    r"^(?:anh|chị|cô|chú|bác|em|ông|bà|thầy|sếp|bạn)\s+(.+)$",
    re.IGNORECASE,
)

_SCALAR_FALSE_ASK_PARAMS: dict[str, set[str]] = {
    "reminders.create": {"content"},
    "notes.create": {"content", "title"},
    "email.create_draft": {"body", "subject"},
    "shopping_list.add": {"name"},
    "calendar.create_event": {"title"},
    "tasks.add": {"title"},
}

_SKIP_ENTITY_VERIFY_ID_PARAMS = {
    "contact_id",
    "chat_id",
    "note_id",
    "device_id",
    "scene_id",
    "alarm_id",
    "task_id",
}


def _is_verifiable_entity_id_param(
    target_tool: str,
    param: str,
    catalog: dict[str, dict[str, Any]],
) -> bool:
    """Return True for any x-ref or *_id param not already handled by contact/EntityGuards."""
    if param in _SKIP_ENTITY_VERIFY_ID_PARAMS:
        return False
    props = ((catalog.get(target_tool) or {}).get("parameters") or {}).get("properties") or {}
    pspec = props.get(param)
    if isinstance(pspec, dict) and pspec.get("x-ref"):
        return True
    return param.endswith("_id")


def _maybe_add_empty_search_hint(
    call: ToolCall,
    res: Any,
    catalog: dict[str, dict[str, Any]],
) -> Any:
    """Attach a concise recovery hint to tool messages when a filtered read returns empty."""
    if call.name == "loyalty.get_points" and call.arguments:
        is_empty_loyalty = (isinstance(res, list) and not res) or (
            isinstance(res, dict)
            and (
                res.get("ok") is False
                or (isinstance(res.get("points"), list) and not res.get("points"))
                or (isinstance(res.get("programs"), list) and not res.get("programs"))
            )
        )
        if is_empty_loyalty:
            hint_msg = (
                "Không tìm thấy chương trình theo tên vừa nhập; hãy gọi "
                "loyalty.get_points({}) không tham số để xem toàn bộ danh sách chương trình."
            )
            return {**res, "hint": hint_msg} if isinstance(res, dict) else {"points": res, "hint": hint_msg}

    if not isinstance(res, dict):
        return res

    if call.name == "contacts.search" and res.get("contacts") == []:
        q = str(call.arguments.get("query") or "").strip()
        m = _CONTACT_HONORIFIC_RE.match(q)
        if m:
            short_q = m.group(1).strip()
            hint_msg = (
                f"Không tìm thấy '{q}'; hãy thử contacts.search(query='{short_q}') "
                f"hoặc contacts.search({{}}) trước khi kết luận thiếu thông tin."
            )
            return {**res, "hint": hint_msg}
        if q:
            return {
                **res,
                "hint": f"Không tìm thấy '{q}'; có thể thử contacts.search({{}}) hoặc từ khoá khác.",
            }
        return res

    spec = catalog.get(call.name) or {}
    if spec.get("kind") == "read" and call.arguments:
        req = ((spec.get("parameters") or {}).get("required")) or []
        has_empty_list = any(isinstance(v, list) and len(v) == 0 for k, v in res.items() if k != "ok")
        if not req and has_empty_list:
            return {
                **res,
                "hint": f"Không có kết quả với bộ lọc vừa rồi; thử từ khoá ngắn hơn hoặc gọi {call.name}({{}}) không tham số để xem toàn bộ danh sách.",
            }
    return res


def _should_verify_ask(call: ToolCall, sess: "_ReactSession") -> str | None:
    """Return a 1-shot self-verification prompt when sys.ask_missing_info looks like a false ask."""
    if not sess.guards.enable_ask_guard or sess.ask_verified or sess.repairs >= sess.max_repairs:
        return None
    if call.name != ASK_MISSING_INFO:
        return None
    target_tool = str(call.arguments.get("tool_name") or "").strip()
    raw_params = call.arguments.get("params") or []
    params = [str(p).strip() for p in (raw_params if isinstance(raw_params, list) else [raw_params]) if p]
    if not target_tool or not params:
        return None

    scalar_watch = _SCALAR_FALSE_ASK_PARAMS.get(target_tool, set())
    if any(p in scalar_watch for p in params):
        return (
            "KIỂM TRA LẠI TRƯỚC KHI HỎI: Chỉ gọi `sys.ask_missing_info` nếu tham số đó THỰC SỰ KHÔNG CÓ "
            "trong câu nói người dùng, lịch sử hội thoại hay kết quả tool vừa đọc. Nếu đã có đủ thông tin, "
            f"hãy gọi trực tiếp `{target_tool}`; chỉ gọi lại `sys.ask_missing_info` nếu chắc chắn bị thiếu."
        )

    app = target_tool.split(".", 1)[0]
    if any(_is_verifiable_entity_id_param(target_tool, p, sess.catalog) for p in params) and sess.known_ids_by_app.get(app):
        return (
            "KIỂM TRA LẠI TRƯỚC KHI HỎI: Kết quả tra cứu trước đó đã có danh sách mục thực tế. "
            "Hãy đối chiếu kỹ các trường như `status` (ví dụ `active`), `time_min`, `name`, `date`, `from_time` "
            f"với yêu cầu/lịch sử người dùng; nếu xác định được đúng mục, hãy gọi ngay `{target_tool}`. "
            "Chỉ gọi lại `sys.ask_missing_info` nếu thực sự mơ hồ không thể phân biệt."
        )
    return None


class _ReactSession:
    """Single-sample ReAct state machine shared by sequential and batched runners."""

    def __init__(
        self,
        sample: Sample,
        env: SafeEnv,
        catalog: dict[str, dict[str, Any]],
        *,
        current_text: str | None = None,
        history_texts: dict[int, str] | None = None,
        asr_phonetic_diff: str | None = None,
        prompt_style: str = "v1",
        max_steps: int = 8,
        max_repairs: int = 2,
        timeout_s: float = 25.0,
        max_prompt_tokens: int = 12000,
        max_result_chars: int = 1500,
        guardrails: GuardrailConfig | None = None,
        t0: float | None = None,
    ) -> None:
        self.sample = sample
        self.env = env
        self.catalog = catalog
        self.current_text = current_text
        self.prompt_style = prompt_style
        self.max_steps = max_steps
        self.max_repairs = max_repairs
        self.timeout_s = timeout_s
        self.max_result_chars = max_result_chars
        self.guards = guardrails if guardrails is not None else GuardrailConfig()
        self.t0 = t0 if t0 is not None else time.perf_counter()
        self.end_ts: float | None = None
        self.done = False

        self.boundary_verifier = BoundaryVerifier(catalog=catalog)
        self.tools = select_tools(sample, catalog)
        self.tool_names = {
            name
            for t in self.tools
            if isinstance(t, dict)
            for name in (
                [t.get("name")]
                if t.get("name")
                else [((t.get("function") or {}) if isinstance(t.get("function"), dict) else {}).get("name")]
            )
            if isinstance(name, str) and name
        }
        self.tools_tokens_est = estimate_tools_tokens(self.tools)
        self.messages = build_messages(
            sample,
            current_text=current_text,
            history_texts=history_texts,
            asr_phonetic_diff=asr_phonetic_diff,
            prompt_style=prompt_style,
            max_prompt_tokens=max_prompt_tokens,
            max_result_chars=max_result_chars,
        )

        self.steps = 0
        self.repairs = 0
        self.final_text = ""
        self.trace: list[dict[str, Any]] = []
        self.seen_read_calls: set[tuple[str, str]] = set()
        self.history_reads: dict[tuple[str, str], Any] = {}
        self.entity_pools: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for turn in getattr(sample, "history", []) or []:
            for i, hc in enumerate(getattr(turn, "tool_calls", []) or []):
                kind = (catalog.get(hc.name) or {}).get("kind")
                if kind == "read":
                    sig = _call_signature(hc)
                    self.seen_read_calls.add(sig)
                    if i < len(getattr(turn, "tool_results", []) or []):
                        self.history_reads[sig] = turn.tool_results[i]
                if i < len(getattr(turn, "tool_results", []) or []):
                    _extract_entities_by_tool(hc.name, turn.tool_results[i], self.entity_pools)

        self.known_ids_by_app = _init_known_ids(sample, catalog, history_texts=history_texts)
        self.ambig_contact_ids: set[str] = set()
        self.ambig_phones: set[str] = set()
        self.ambig_emails: set[str] = set()
        self.ambig_chat_ids: set[str] = set()

        raw_context = getattr(sample, "context", None)
        if isinstance(raw_context, dict):
            cur_time = str(raw_context.get("current_time") or "")
        else:
            cur_time = str(getattr(raw_context, "current_time", "") or "")
        self.cur_date = cur_time[:10] if len(cur_time) >= 10 else None
        self.active_text = current_text if current_text is not None else (sample.audio_text or "")
        self.text_low = self.active_text.lower()
        self.force_schema_next = False
        self.ask_verified = False

    def can_step(self, now: float) -> bool:
        if self.done:
            return False
        if self.steps >= self.max_steps:
            self.done = True
            self.end_ts = now
            return False
        if (now - self.t0) >= self.timeout_s:
            self.trace.append({"event": "timeout", "elapsed_s": now - self.t0})
            self.done = True
            self.end_ts = now
            return False
        return True

    def next_request(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
        self.steps += 1
        use_force = self.force_schema_next
        self.force_schema_next = False
        return self.messages, self.tools, use_force

    def apply_generation(self, raw_output: str) -> None:
        raw_calls = parse_tool_calls(raw_output, self.catalog)
        parsed_calls: list[ToolCall] = []
        bv_forced_ask = False
        for c in raw_calls:
            if self.guards.enable_arg_sanitizer:
                c = sanitize_tool_call(c, self.sample, current_text=self.current_text)
            if self.guards.enable_boundary_verifier:
                c = self.boundary_verifier.verify_optional_args(
                    c, self.active_text, current_date=self.cur_date
                )
            if self.guards.enable_ask_guard:
                c = _apply_ask_guard(
                    c,
                    self.text_low,
                    self.ambig_contact_ids,
                    self.ambig_phones,
                    self.ambig_emails,
                    self.ambig_chat_ids,
                )
            if self.guards.enable_boundary_verifier and c.name != ASK_MISSING_INFO:
                for id_param in _TARGETED_ENTITY_ID_PARAMS:
                    chosen_id = c.arguments.get(id_param)
                    if not isinstance(chosen_id, str) or not chosen_id:
                        continue
                    pool_map = self.entity_pools.get(id_param) or {}
                    chosen_entity = pool_map.get(chosen_id)
                    if not isinstance(chosen_entity, dict):
                        continue
                    chosen_label = str(
                        chosen_entity.get("name") or chosen_entity.get("title") or ""
                    ).strip().lower()
                    if not chosen_label:
                        continue
                    same_label_entities = [
                        e
                        for e in pool_map.values()
                        if str(e.get("name") or e.get("title") or "").strip().lower() == chosen_label
                    ]
                    if len(same_label_entities) >= 2 and self.boundary_verifier.should_ask_clarification(
                        self.active_text,
                        same_label_entities,
                        c.name,
                        id_param,
                    ):
                        c = ToolCall(
                            name=ASK_MISSING_INFO,
                            arguments={"tool_name": c.name, "params": [id_param]},
                        )
                        bv_forced_ask = True
                        break
            parsed_calls.append(c)

        if not parsed_calls:
            if has_malformed_tool_call(raw_output, self.catalog) and self.repairs < self.max_repairs:
                self.repairs += 1
                self.force_schema_next = True
                self.messages.append({"role": "assistant", "content": raw_output})
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "LỖI CÚ PHÁP: Lệnh gọi tool bị lỗi định dạng. "
                            f"Hãy phát lại đúng cấu trúc {format_tool_call_hint(self.prompt_style)}."
                        ),
                    }
                )
                self.trace.append({"step": self.steps, "event": "malformed_repair", "raw_output": raw_output})
                return

            self.final_text = extract_final_text(raw_output)
            self.trace.append({"step": self.steps, "event": "final_text", "final_text": self.final_text, "raw_output": raw_output})
            self.done = True
            self.end_ts = time.perf_counter()
            return

        if self.guards.enable_boundary_verifier and self.steps < self.max_steps:
            parsed_names = {pc.name for pc in parsed_calls}
            prereq_call: ToolCall | None = None
            suppressed_ask: ToolCall | None = None
            for pc in parsed_calls:
                cand = self.boundary_verifier.find_missing_prereq_read(
                    pc,
                    seen_reads=self.seen_read_calls,
                    available_tools=self.tool_names,
                    known_ids_by_app=self.known_ids_by_app,
                )
                if cand is not None and cand.name not in parsed_names:
                    prereq_check = _preflight_validate(
                        cand,
                        self.env,
                        self.catalog,
                        self.known_ids_by_app,
                        text_low=self.text_low,
                        enable_xref_guard=self.guards.enable_xref_guard,
                    )
                    if prereq_check.get("ok", False):
                        prereq_call = cand
                        suppressed_ask = pc
                        break
            if prereq_call is not None:
                prereq_sig = _call_signature(prereq_call)
                if prereq_sig in self.history_reads:
                    prereq_res = self.history_reads[prereq_sig]
                else:
                    prereq_res = self.env.call(prereq_call.name, prereq_call.arguments)
                self.seen_read_calls.add(prereq_sig)
                _collect_ids_from_result(prereq_call.name.split(".", 1)[0], prereq_res, self.known_ids_by_app)
                _extract_entities_by_tool(prereq_call.name, prereq_res, self.entity_pools)
                _collect_ambiguous_from_result(
                    prereq_call.name,
                    prereq_res,
                    self.text_low,
                    self.ambig_contact_ids,
                    self.ambig_phones,
                    self.ambig_emails,
                    self.ambig_chat_ids,
                )
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": prereq_call.name, "arguments": prereq_call.arguments},
                            }
                        ],
                    }
                )
                self.messages.append(
                    {
                        "role": "tool",
                        "name": prereq_call.name,
                        "content": truncate_tool_result(prereq_res, max_chars=self.max_result_chars),
                    }
                )
                self.trace.append(
                    {
                        "step": self.steps,
                        "event": "prereq_read",
                        "call": prereq_call.model_dump(),
                        "suppressed_call": suppressed_ask.model_dump() if suppressed_ask else None,
                    }
                )
                return

        if (
            not bv_forced_ask
            and len(parsed_calls) == 1
            and parsed_calls[0].name == ASK_MISSING_INFO
            and self.steps < self.max_steps
        ):
            verify_msg = _should_verify_ask(parsed_calls[0], self)
            if verify_msg:
                self.ask_verified = True
                self.repairs += 1
                self.messages.append({"role": "assistant", "content": raw_output})
                self.messages.append({"role": "user", "content": verify_msg})
                self.trace.append(
                    {
                        "step": self.steps,
                        "event": "ask_verify",
                        "call": parsed_calls[0].model_dump(),
                        "raw_output": raw_output,
                    }
                )
                return

        validations = [
            _preflight_validate(
                c,
                self.env,
                self.catalog,
                self.known_ids_by_app,
                text_low=self.text_low,
                enable_xref_guard=self.guards.enable_xref_guard,
            )
            for c in parsed_calls
        ]
        any_invalid = any(not v.get("ok", False) for v in validations)
        any_valid = any(v.get("ok", False) for v in validations)

        # Case 1: All calls in this step failed pre-flight validation
        if any_invalid and not any_valid:
            if self.repairs < self.max_repairs:
                self.repairs += 1
                self.force_schema_next = True
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": extract_final_text(raw_output),
                        "tool_calls": [
                            {"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                            for c in parsed_calls
                        ],
                    }
                )
                for c, v in zip(parsed_calls, validations):
                    err_payload = {
                        "ok": False,
                        "error": v.get("error"),
                        "hint": _format_schema_hint(c.name, self.catalog),
                    }
                    self.messages.append(
                        {
                            "role": "tool",
                            "name": c.name,
                            "content": json.dumps(err_payload, ensure_ascii=False),
                        }
                    )
                self.trace.append(
                    {
                        "step": self.steps,
                        "event": "schema_repair",
                        "attempted_calls": [c.model_dump() for c in parsed_calls],
                        "validations": validations,
                        "raw_output": raw_output,
                    }
                )
                return

            self.trace.append(
                {
                    "step": self.steps,
                    "event": "schema_abort",
                    "validations": validations,
                    "raw_output": raw_output,
                }
            )
            self.done = True
            self.end_ts = time.perf_counter()
            return

        # Case 2: At least one valid call
        exec_items: list[tuple[ToolCall, dict[str, Any]]] = []
        for c, v in zip(parsed_calls, validations):
            if v.get("ok", False):
                kind = (self.catalog.get(c.name) or {}).get("kind")
                sig = _call_signature(c)
                if kind == "read" and sig in self.seen_read_calls and sig not in self.history_reads:
                    continue
                if kind == "read":
                    self.seen_read_calls.add(sig)
            exec_items.append((c, v))

        if not any(v.get("ok", False) for _, v in exec_items):
            self.trace.append({"step": self.steps, "event": "duplicate_read_stop", "raw_output": raw_output})
            self.done = True
            self.end_ts = time.perf_counter()
            return

        self.messages.append(
            {
                "role": "assistant",
                "content": extract_final_text(raw_output),
                "tool_calls": [
                    {"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                    for c, _ in exec_items
                ],
            }
        )

        step_results: list[dict[str, Any]] = []
        executed_calls: list[dict[str, Any]] = []
        asked_missing = False
        had_blocked_sibling = False

        for c, v in exec_items:
            if not v.get("ok", False):
                had_blocked_sibling = True
                err_payload = {
                    "ok": False,
                    "error": v.get("error"),
                    "hint": "Lệnh này chưa được thực thi. Hãy tra cứu đúng ID từ kết quả tool rồi gọi lại.",
                }
                self.messages.append(
                    {
                        "role": "tool",
                        "name": c.name,
                        "content": json.dumps(err_payload, ensure_ascii=False),
                    }
                )
                continue

            sig = _call_signature(c)
            kind = (self.catalog.get(c.name) or {}).get("kind")
            if kind == "read" and sig in self.history_reads:
                res = self.history_reads[sig]
            else:
                res = self.env.call(c.name, c.arguments)
                executed_calls.append(c.model_dump())

            app = c.name.split(".", 1)[0]
            _collect_ids_from_result(app, res, self.known_ids_by_app)
            _extract_entities_by_tool(c.name, res, self.entity_pools)
            _collect_ambiguous_from_result(
                c.name,
                res,
                self.text_low,
                self.ambig_contact_ids,
                self.ambig_phones,
                self.ambig_emails,
                self.ambig_chat_ids,
            )
            step_results.append(res)
            msg_payload = _maybe_add_empty_search_hint(c, res, self.catalog)
            self.messages.append(
                {
                    "role": "tool",
                    "name": c.name,
                    "content": truncate_tool_result(msg_payload, max_chars=self.max_result_chars),
                }
            )
            if c.name == ASK_MISSING_INFO:
                asked_missing = True

        if had_blocked_sibling:
            self.repairs += 1

        self.trace.append(
            {
                "step": self.steps,
                "event": "executed",
                "calls": executed_calls,
                "results": step_results,
                "raw_output": raw_output,
            }
        )

        if asked_missing:
            self.done = True
            self.end_ts = time.perf_counter()

    def finalize(self) -> AgentRunResult:
        end = self.end_ts if self.end_ts is not None else time.perf_counter()
        return AgentRunResult(
            id=self.sample.id,
            tool_calls=self.env.get_tool_calls(),
            state_hash=self.env.get_state_hash(),
            steps=self.steps,
            repairs=self.repairs,
            latency_s=end - self.t0,
            tools_tokens_est=self.tools_tokens_est,
            final_text=self.final_text,
            trace=self.trace,
        )


def run_react_batch(
    items: list[BatchReactItem],
    llm: ChatBackend,
    *,
    prompt_style: str = "v1",
    max_steps: int = 8,
    max_repairs: int = 2,
    timeout_s: float = 25.0,
    max_prompt_tokens: int = 12000,
    max_result_chars: int = 1500,
    guardrails: GuardrailConfig | None = None,
) -> list[AgentRunResult]:
    """Run lockstep multi-sample ReAct loop leveraging batched LLM generation when supported."""
    if not items:
        return []
    sessions = [
        _ReactSession(
            item.sample,
            item.env,
            item.catalog,
            current_text=item.current_text,
            history_texts=item.history_texts,
            asr_phonetic_diff=item.asr_phonetic_diff,
            prompt_style=prompt_style,
            max_steps=max_steps,
            max_repairs=max_repairs,
            timeout_s=timeout_s,
            max_prompt_tokens=max_prompt_tokens,
            max_result_chars=max_result_chars,
            guardrails=guardrails,
            t0=item.t_start,
        )
        for item in items
    ]
    has_batch_gen = hasattr(llm, "generate_batch")
    while True:
        now = time.perf_counter()
        active = [s for s in sessions if s.can_step(now)]
        if not active:
            break
        reqs = [s.next_request() for s in active]
        if has_batch_gen:
            raw_outputs = llm.generate_batch(reqs)  # type: ignore[attr-defined]
        else:
            raw_outputs = [
                _generate_with_optional_schema(llm, msgs, tools, force_tool_schema=force)
                for msgs, tools, force in reqs
            ]
        for s, raw_out in zip(active, raw_outputs):
            s.apply_generation(raw_out)
    return [s.finalize() for s in sessions]


def run_react_loop(
    sample: Sample,
    env: SafeEnv,
    catalog: dict[str, dict[str, Any]],
    llm: ChatBackend,
    *,
    current_text: str | None = None,
    history_texts: dict[int, str] | None = None,
    asr_phonetic_diff: str | None = None,
    prompt_style: str = "v1",
    max_steps: int = 8,
    max_repairs: int = 2,
    timeout_s: float = 25.0,
    max_prompt_tokens: int = 12000,
    max_result_chars: int = 1500,
    guardrails: GuardrailConfig | None = None,
) -> AgentRunResult:
    sess = _ReactSession(
        sample,
        env,
        catalog,
        current_text=current_text,
        history_texts=history_texts,
        asr_phonetic_diff=asr_phonetic_diff,
        prompt_style=prompt_style,
        max_steps=max_steps,
        max_repairs=max_repairs,
        timeout_s=timeout_s,
        max_prompt_tokens=max_prompt_tokens,
        max_result_chars=max_result_chars,
        guardrails=guardrails,
    )
    while sess.can_step(time.perf_counter()):
        msgs, tools, force = sess.next_request()
        raw_output = _generate_with_optional_schema(llm, msgs, tools, force_tool_schema=force)
        sess.apply_generation(raw_output)
    return sess.finalize()
