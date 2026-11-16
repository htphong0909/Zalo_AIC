"""Error-Driven Data Engine (EDDE) and Hammer V3 masking for SFT V3."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta
import json
import random
import re
import string
from typing import Any

from scripts.build_lfm3_data import apply_hammer_masking
from src.core.schemas import Sample, ToolCall
from src.data_gen.execution_verifier import verify_step_validity
from src.data_gen.prompt_utils import (
    build_assistant_history_message,
    build_sample_prefix_messages,
    call_to_dict,
    resolve_installed_apps,
    serialize_observation,
)

_WEEKDAY_VI = {
    0: "Thứ Hai",
    1: "Thứ Ba",
    2: "Thứ Tư",
    3: "Thứ Năm",
    4: "Thứ Sáu",
    5: "Thứ Bảy",
    6: "Chủ Nhật",
}

_SENSITIVE_DISTRACTOR_GROUPS: list[tuple[set[str], list[str]]] = [
    ({"reminders", "calendar", "alarm"}, ["notes", "tasks", "weather"]),
    ({"email", "chat_ott", "sms"}, ["notes", "contacts", "calculator"]),
    ({"banking", "ewallet", "bill_payment"}, ["calculator", "exchange_rate", "loyalty"]),
    ({"food_delivery", "ride_hailing", "bike_rental"}, ["maps", "shopping_list", "weather"]),
    ({"smart_home", "appliances", "home_automation"}, ["home_energy", "tasks", "notes"]),
]


def assert_not_excluded(sample_id: str, exclude_ids: set[str] | None) -> None:
    """Abort if sample_id belongs to any protected holdout/dev set (zero-leak guard)."""
    if exclude_ids and str(sample_id) in exclude_ids:
        raise ValueError(
            f"Zero-leak guard violated: sample_id='{sample_id}' is in excluded set."
        )


def _filter_eligible(
    train_samples: list[Sample],
    exclude_ids: set[str] | None,
) -> list[Sample]:
    """Return samples outside exclude_ids, asserting none of the returned IDs leak."""
    excl = {str(x) for x in (exclude_ids or set())}
    out: list[Sample] = []
    for s in train_samples:
        sid = str(s.id)
        if sid in excl:
            continue
        assert_not_excluded(sid, excl)
        out.append(s)
    return out


def _parse_iso_dt(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return datetime.fromisoformat(raw.strip())
    except ValueError:
        return None


def _sample_calls_and_results(sample: Sample) -> tuple[list[dict[str, Any]], list[Any]]:
    if not sample.label:
        return [], []
    if isinstance(sample.label, dict):
        calls = [call_to_dict(c) for c in (sample.label.get("tool_calls") or [])]
        res = list(sample.label.get("tool_results") or [])
        return calls, res
    calls = [call_to_dict(c) for c in (getattr(sample.label, "tool_calls", None) or [])]
    res = list(getattr(sample.label, "tool_results", None) or [])
    return calls, res


def _build_verified_step(
    *,
    step_id: str,
    sample_id: str,
    step_category: str,
    prompt_messages: list[dict[str, Any]],
    step_calls: list[dict[str, Any]],
    sample_context: dict[str, Any],
    tools: list[str] | list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
    exclude_ids: set[str] | None = None,
    step_index: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    assert_not_excluded(sample_id, exclude_ids)
    step: dict[str, Any] = {
        "id": step_id,
        "sample_id": str(sample_id),
        "step_index": step_index,
        "step_category": step_category,
        "prompt_messages": prompt_messages,
        "step_calls": step_calls,
        "tools": tools,
        "sample_context": sample_context,
    }
    if extra:
        step.update(extra)
    if not verify_step_validity(step, catalog):
        return None
    return step


def _extract_scalars(obj: Any, *, min_len: int = 3) -> list[str]:
    out: list[str] = []
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_extract_scalars(v, min_len=min_len))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_extract_scalars(item, min_len=min_len))
    elif isinstance(obj, str):
        s = obj.strip()
        if len(s) >= min_len:
            out.append(s)
    return out


def generate_time_and_math_edde(
    train_samples: list[Sample],
    catalog: dict[str, dict[str, Any]],
    count: int = 250,
    seed: int = 42,
    exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate schema-verified Vietnamese relative date/time and arithmetic EDDE steps."""
    rng = random.Random(seed)
    eligible = _filter_eligible(train_samples, exclude_ids)
    results: list[dict[str, Any]] = []
    if not eligible or count <= 0:
        return results

    # Prefer samples that already have time/math/finance tools installed
    time_candidates = [
        s for s in eligible
        if set((s.context or {}).get("installed_apps", {}).keys()) & {"reminders", "alarm", "calendar", "calculator"}
    ] or list(eligible)
    rng.shuffle(time_candidates)

    templates = [
        ("reminders.create", "advance_meeting"),
        ("alarm.create", "morning_offset"),
        ("calendar.create_event", "weekday_duration"),
        ("calendar.update_event", "calendar_shift"),
        ("calculator.calculate", "split_bill"),
    ]

    idx = 0
    attempts = 0
    max_attempts = max(count * 8, 64)
    while len(results) < count and attempts < max_attempts:
        attempts += 1
        s = time_candidates[idx % len(time_candidates)]
        mode_tool, pattern = templates[(attempts - 1) % len(templates)]
        if mode_tool not in catalog:
            idx += 1
            continue

        ctx = copy.deepcopy(s.context or {})
        base_dt = _parse_iso_dt(ctx.get("current_time")) or datetime.fromisoformat("2026-11-14T09:00:00+07:00")
        day_offset = rng.choice([1, 2, 3])
        lead_min = rng.choice([15, 20, 25, 30, 45])
        extra_prompt_msgs: list[dict[str, Any]] | None = None
        installed_override: list[str] | None = None

        step_calls: list[dict[str, Any]] = []
        user_text = ""
        math_note = ""

        if pattern == "advance_meeting":
            anchor_h = rng.choice([8, 9, 14, 16, 19, 20])
            anchor_m = rng.choice([0, 30])
            target_dt = (base_dt + timedelta(days=day_offset)).replace(
                hour=anchor_h, minute=anchor_m, second=0, microsecond=0
            )
            day_phrase = {1: "ngày mai", 2: "ngày kia", 3: f"{_WEEKDAY_VI[target_dt.weekday()]} tuần này"}[day_offset]
            topic = rng.choice(["uống thuốc", "họp nhóm", "ra sân bay", "kiểm tra hóa đơn"])
            if idx % 2 == 0:
                remind_dt = target_dt - timedelta(minutes=lead_min)
                user_text = (
                    f"Nhắc tôi {topic} sớm hơn {lead_min} phút trước lịch lúc "
                    f"{anchor_h:02d}:{anchor_m:02d} {day_phrase}."
                )
                math_note = (
                    f"{target_dt.strftime('%Y-%m-%d')} {anchor_h:02d}:{anchor_m:02d} "
                    f"- {lead_min}p (sớm hơn) = {remind_dt.strftime('%H:%M')}."
                )
            else:
                remind_dt = target_dt + timedelta(minutes=lead_min)
                shift_phrase = rng.choice([f"lùi lại {lead_min} phút", f"muộn thêm {lead_min} phút"])
                user_text = (
                    f"Nhắc tôi {topic} {shift_phrase} so với mốc "
                    f"{anchor_h:02d}:{anchor_m:02d} {day_phrase}."
                )
                math_note = (
                    f"{target_dt.strftime('%Y-%m-%d')} {anchor_h:02d}:{anchor_m:02d} "
                    f"+ {lead_min}p (lùi/muộn thêm) = {remind_dt.strftime('%H:%M')}."
                )
            step_calls = [
                {
                    "name": "reminders.create",
                    "arguments": {
                        "content": topic,
                        "time": remind_dt.isoformat(),
                    },
                }
            ]
        elif pattern == "morning_offset":
            base_h = rng.choice([6, 7])
            base_m = rng.choice([0, 30])
            delta = rng.choice([15, 20, 25, 30])
            target_dt = (base_dt + timedelta(days=1)).replace(
                hour=base_h, minute=base_m, second=0, microsecond=0
            )
            if idx % 2 == 0:
                alarm_dt = target_dt - timedelta(minutes=delta)
                user_text = (
                    f"Đặt báo thức sáng mai sớm hơn {delta} phút so với {base_h:02d} giờ "
                    f"{'rưỡi' if base_m == 30 else 'đúng'} giúp tôi."
                )
                math_note = (
                    f"Ngày mai {(base_dt + timedelta(days=1)).strftime('%Y-%m-%d')}; "
                    f"{base_h:02d}:{base_m:02d} - {delta}p = {alarm_dt.strftime('%H:%M')}."
                )
            else:
                alarm_dt = target_dt + timedelta(minutes=delta)
                shift_w = rng.choice(["lùi lại", "muộn hơn"])
                user_text = (
                    f"Đặt báo thức sáng mai {shift_w} {delta} phút so với {base_h:02d} giờ "
                    f"{'rưỡi' if base_m == 30 else 'đúng'} giúp tôi."
                )
                math_note = (
                    f"Ngày mai {(base_dt + timedelta(days=1)).strftime('%Y-%m-%d')}; "
                    f"{base_h:02d}:{base_m:02d} + {delta}p (lùi/muộn) = {alarm_dt.strftime('%H:%M')}."
                )
            step_calls = [
                {
                    "name": "alarm.create",
                    "arguments": {"time": alarm_dt.isoformat()},
                }
            ]
        elif pattern == "calendar_shift":
            orig_h = rng.choice([8, 9, 14, 15])
            orig_m = rng.choice([0, 30])
            shift_delta = rng.choice([30, 45, 60])
            is_delay = (idx % 3 != 0)  # emphasize Vietnamese 'lùi lại / muộn thêm' (+) while keeping '-' balance
            ev_start = (base_dt + timedelta(days=1)).replace(hour=orig_h, minute=orig_m, second=0, microsecond=0)
            ev_end = ev_start + timedelta(hours=1)
            new_start = ev_start + timedelta(minutes=shift_delta if is_delay else -shift_delta)
            new_end = ev_end + timedelta(minutes=shift_delta if is_delay else -shift_delta)
            title = rng.choice(["Họp dự án", "Khám sức khỏe", "Tập thể dục", "Trao đổi đối tác"])
            if is_delay:
                delta_str = "nửa tiếng" if shift_delta == 30 else ("một tiếng" if shift_delta == 60 else "45 phút")
                verb = rng.choice([f"lùi lại thêm {delta_str}", f"muộn thêm {delta_str}", f"dời sang muộn hơn {delta_str}"])
                user_text = f"Lịch {title.lower()} ngày mai cho {verb} giúp tôi nhé."
                math_note = f"{ev_start.strftime('%H:%M')} + {shift_delta}p (lùi lại = cộng thêm) -> {new_start.strftime('%H:%M')}."
            else:
                user_text = f"Đẩy lịch {title.lower()} ngày mai lên sớm hơn {shift_delta} phút giúp tôi."
                math_note = f"{ev_start.strftime('%H:%M')} - {shift_delta}p (sớm hơn = trừ đi) -> {new_start.strftime('%H:%M')}."
            ctx.setdefault("installed_apps", {})
            if isinstance(ctx["installed_apps"], dict) and "calendar" not in ctx["installed_apps"]:
                ctx["installed_apps"]["calendar"] = "calendar"
            synth_ev = Sample(id=str(s.id), audio_text=user_text, history=[], context=ctx)
            extra_prompt_msgs = build_sample_prefix_messages(synth_ev, catalog)
            extra_prompt_msgs.append(build_assistant_history_message([ToolCall(name="calendar.list_events", arguments={})]))
            ev_obs = {
                "ok": True,
                "events": [{"id": "evt_01", "title": title, "start": ev_start.isoformat(), "end": ev_end.isoformat()}],
            }
            extra_prompt_msgs.append({"role": "tool", "name": "calendar.list_events", "content": serialize_observation(ev_obs)})
            installed_override = resolve_installed_apps(synth_ev, ["calendar.list_events", "calendar.update_event"])
            step_calls = [
                {
                    "name": "calendar.update_event",
                    "arguments": {
                        "event_id": "evt_01",
                        "start": new_start.isoformat(),
                        "end": new_end.isoformat(),
                    },
                }
            ]
        elif pattern == "weekday_duration":
            dur_min = rng.choice([45, 60, 90])
            start_h = rng.choice([9, 10, 14, 15])
            start_dt = (base_dt + timedelta(days=day_offset)).replace(
                hour=start_h, minute=0, second=0, microsecond=0
            )
            end_dt = start_dt + timedelta(minutes=dur_min)
            title = rng.choice(["Báo cáo tiến độ", "Trao đổi khách hàng", "Họp kỹ thuật"])
            wday = _WEEKDAY_VI[start_dt.weekday()]
            user_text = (
                f"Tạo lịch {title} vào {wday} ({start_dt.strftime('%d/%m')}) bắt đầu từ "
                f"{start_h} giờ và kéo dài {dur_min} phút."
            )
            math_note = (
                f"Start = {start_dt.strftime('%Y-%m-%d %H:%M')}; "
                f"+ {dur_min} phút -> End = {end_dt.strftime('%H:%M')}."
            )
            step_calls = [
                {
                    "name": "calendar.create_event",
                    "arguments": {
                        "title": title,
                        "start": start_dt.isoformat(),
                        "end": end_dt.isoformat(),
                    },
                }
            ]
        else:
            people = rng.choice([3, 4, 5])
            total = people * rng.choice([85000, 120000, 145000, 180000])
            tip = rng.choice([15000, 20000, 30000])
            expr = f"({total} + {tip}) / {people}"
            per_person = (total + tip) // people
            user_text = (
                f"Tính giúp tôi tiền ăn {total:,} đồng cộng {tip:,} đồng phí phục vụ "
                f"chia đều cho {people} người thì mỗi người trả bao nhiêu?"
            ).replace(",", ".")
            math_note = f"{expr} = {per_person}."
            step_calls = [{"name": "calculator.calculate", "arguments": {"expression": expr}}]

        # Ensure context exposes the used app alongside existing installed apps
        app = mode_tool.split(".", 1)[0]
        ctx.setdefault("installed_apps", {})
        if isinstance(ctx["installed_apps"], dict) and app not in ctx["installed_apps"]:
            ctx["installed_apps"][app] = app

        if extra_prompt_msgs is not None and installed_override is not None:
            prompt_msgs = extra_prompt_msgs
            installed = installed_override
        else:
            synth_sample = Sample(
                id=str(s.id),
                audio_text=user_text,
                history=[],
                context=ctx,
            )
            prompt_msgs = build_sample_prefix_messages(synth_sample, catalog)
            installed = resolve_installed_apps(synth_sample, [mode_tool])
        st = _build_verified_step(
            step_id=f"edde_time_math_{s.id}_{idx}",
            sample_id=str(s.id),
            step_category="act",
            prompt_messages=prompt_msgs,
            step_calls=step_calls,
            sample_context=ctx,
            tools=installed,
            catalog=catalog,
            exclude_ids=exclude_ids,
            extra={"math_note": math_note, "edde_bucket": "time_and_math"},
        )
        idx += 1
        if st is not None:
            results.append(st)

    return results[:count]


def generate_long_history_coref_edde(
    train_samples: list[Sample],
    catalog: dict[str, dict[str, Any]],
    count: int = 220,
    seed: int = 42,
    exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate multi-turn long-history coreference EDDE steps anchored on prior turns/observations."""
    rng = random.Random(seed)
    eligible = _filter_eligible(train_samples, exclude_ids)
    results: list[dict[str, Any]] = []
    if not eligible or count <= 0:
        return results

    distractor_turns = [
        ("Thời tiết hôm nay thế nào?", "weather.get_current", {}, {"ok": True, "temp_c": 29, "condition": "nắng nhẹ"}, "Hôm nay trời nắng nhẹ 29 độ."),
        ("Kiểm tra tỷ giá USD hôm nay giúp tôi", "exchange_rate.get_rate", {"from_currency": "USD", "to_currency": "VND"}, {"ok": True, "rate": 25400}, "Tỷ giá USD hiện là 25.400 VND."),
    ]

    # Collect multi-turn or read->write candidates from eligible samples
    candidates: list[tuple[Sample, dict[str, Any], Any, dict[str, Any]]] = []
    for s in eligible:
        calls, res_list = _sample_calls_and_results(s)
        if len(calls) >= 2 and len(res_list) >= 1:
            read_c = calls[0]
            read_r = res_list[0]
            write_c = calls[-1]
            r_kind = str((catalog.get(read_c["name"]) or {}).get("kind") or "")
            if r_kind == "read" and write_c["name"] != "sys.ask_missing_info":
                candidates.append((s, read_c, read_r, write_c))

    rng.shuffle(candidates)
    for i, (s, read_c, read_r, write_c) in enumerate(candidates):
        if len(results) >= count:
            break
        ctx = copy.deepcopy(s.context or {})
        scalars = _extract_scalars(read_r, min_len=3)[:4]
        if not scalars:
            continue
        cue = f"[Lịch sử lượt 1 ({read_c['name']}): {', '.join(scalars)}]"
        prefix_msgs = build_sample_prefix_messages(s, catalog)
        # Replace final user prompt after inserting 1-2 unrelated turns between the early read and follow-up
        orig_user = prefix_msgs[-1]["content"] if prefix_msgs else (s.audio_text or "")
        history_msgs = list(prefix_msgs[:-1])
        history_msgs.append({"role": "user", "content": f"Kiểm tra giúp tôi thông tin cho yêu cầu: {orig_user}"})
        history_msgs.append(build_assistant_history_message([ToolCall(name=read_c["name"], arguments=read_c["arguments"])]))
        history_msgs.append({"role": "tool", "name": read_c["name"], "content": serialize_observation(read_r)})
        history_msgs.append({"role": "assistant", "content": f"Tôi đã kiểm tra xong ({scalars[0]})."})

        # Insert 2 distractor turns so coreference spans across intermediate turns
        for d_user, d_tool, d_args, d_obs, d_reply in distractor_turns:
            if d_tool in catalog:
                history_msgs.append({"role": "user", "content": d_user})
                history_msgs.append(build_assistant_history_message([ToolCall(name=d_tool, arguments=d_args)]))
                history_msgs.append({"role": "tool", "name": d_tool, "content": serialize_observation(d_obs)})
                history_msgs.append({"role": "assistant", "content": d_reply})

        follow_up = rng.choice(
            [
                f"Quay lại việc ban nãy ({orig_user}), thực hiện đúng theo thông tin lượt đầu giúp tôi.",
                f"Dùng lại dữ liệu lúc đầu vừa tra được để hoàn tất yêu cầu '{orig_user}' nhé.",
            ]
        )
        history_msgs.append({"role": "user", "content": follow_up})

        installed = resolve_installed_apps(s, [read_c["name"], write_c["name"], "weather.get_current", "exchange_rate.get_rate"])
        st = _build_verified_step(
            step_id=f"edde_hist_{s.id}_{i}",
            sample_id=str(s.id),
            step_category="act",
            prompt_messages=history_msgs,
            step_calls=[write_c],
            sample_context=ctx,
            tools=installed,
            catalog=catalog,
            exclude_ids=exclude_ids,
            step_index=len(distractor_turns) + 1,
            extra={
                "history_cue": cue,
                "mandatory_keywords": scalars[:3],
                "edde_bucket": "long_history_coref",
            },
        )
        if st is not None:
            results.append(st)

    return results[:count]


def generate_fuzzy_fidelity_edde(
    train_samples: list[Sample],
    catalog: dict[str, dict[str, Any]],
    count: int = 200,
    seed: int = 42,
    exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate verbatim fuzzy-field fidelity & loyalty.get_points({}) fallback EDDE steps."""
    rng = random.Random(seed)
    eligible = _filter_eligible(train_samples, exclude_ids)
    results: list[dict[str, Any]] = []
    if not eligible or count <= 0:
        return results

    # Part 1 (25%): loyalty.get_points empty-program -> loyalty.get_points({}) fallback recovery
    loyalty_target = max(1, count // 4)
    loyalty_samples = [
        s for s in eligible
        if "loyalty" in (s.context or {}).get("installed_apps", {})
    ] or list(eligible)
    rng.shuffle(loyalty_samples)

    brands = ["Highlands", "MinhMart", "VinID", "Phúc Long", "Starbucks"]
    for i, s in enumerate(loyalty_samples[:loyalty_target]):
        if "loyalty.get_points" not in catalog:
            break
        ctx = copy.deepcopy(s.context or {})
        ctx.setdefault("installed_apps", {})["loyalty"] = "loyalty"
        brand = brands[i % len(brands)]
        prompt_msgs = build_sample_prefix_messages(
            Sample(id=str(s.id), audio_text=f"Kiểm tra điểm tích lũy {brand} của tôi", context=ctx),
            catalog,
        )
        prompt_msgs.append(
            build_assistant_history_message([ToolCall(name="loyalty.get_points", arguments={"program": brand})])
        )
        prompt_msgs.append(
            {
                "role": "tool",
                "name": "loyalty.get_points",
                "content": json.dumps(
                    {
                        "ok": True,
                        "programs": [],
                        "hint": "Không tìm thấy chương trình theo tên vừa nhập; hãy gọi loyalty.get_points({}) không tham số để xem toàn bộ danh sách chương trình.",
                    },
                    ensure_ascii=False,
                ),
            }
        )
        installed = resolve_installed_apps(s, ["loyalty.get_points"])
        st = _build_verified_step(
            step_id=f"edde_loyalty_fb_{s.id}_{i}",
            sample_id=str(s.id),
            step_category="act",
            prompt_messages=prompt_msgs,
            step_calls=[{"name": "loyalty.get_points", "arguments": {}}],
            sample_context=ctx,
            tools=installed,
            catalog=catalog,
            exclude_ids=exclude_ids,
            step_index=1,
            extra={
                "recovery_type": "empty_search",
                "edde_bucket": "fuzzy_loyalty_fallback",
            },
        )
        if st is not None:
            results.append(st)

    # Part 2 (75%): Verbatim string/code/address/amount fidelity into notes.create or reminders.create
    verbatim_payloads = [
        ("Mã đơn DH-8492-VN giao tại 18 Quốc Hương, P. Thảo Điền, TP. Thủ Đức (COD 245.000đ)", "Ghi chú mã đơn và địa chỉ"),
        ("Mã voucher MM100-X9K giảm 100.000đ tại MinhMart số 42 Trần Duy Hưng, Cầu Giấy", "Lưu voucher MinhMart"),
        ("Số hợp đồng HD-2026-0774 của anh Trần Minh Khôi - SĐT 0912345678, phòng 1204B", "Thông tin hợp đồng"),
        ("Lịch bảo trì máy lạnh phòng họp Tầng 4 - Lô B2, khu CNC Hòa Lạc vào 14:30", "Chi tiết bảo trì"),
    ]
    shuffled = list(eligible)
    rng.shuffle(shuffled)
    for i, s in enumerate(shuffled):
        if len(results) >= count:
            break
        if "notes.create" not in catalog:
            break
        exact_text, title = verbatim_payloads[i % len(verbatim_payloads)]
        tokens = [t.strip() for t in re.findall(r"[A-Z]{1,4}-\d[\w-]*|\d{3}\.\d{3}đ|18 Quốc Hương|42 Trần Duy Hưng|0912345678", exact_text)]
        ctx = copy.deepcopy(s.context or {})
        ctx.setdefault("installed_apps", {})["notes"] = "notes"
        user_q = (
            f"Ghi chú nguyên văn toàn bộ nội dung sau với tiêu đề '{title}': {exact_text}"
        )
        synth = Sample(id=str(s.id), audio_text=user_q, context=ctx)
        prompt_msgs = build_sample_prefix_messages(synth, catalog)
        installed = resolve_installed_apps(synth, ["notes.create"])
        st = _build_verified_step(
            step_id=f"edde_fuzzy_{s.id}_{i}",
            sample_id=str(s.id),
            step_category="act",
            prompt_messages=prompt_msgs,
            step_calls=[{"name": "notes.create", "arguments": {"title": title, "content": exact_text}}],
            sample_context=ctx,
            tools=installed,
            catalog=catalog,
            exclude_ids=exclude_ids,
            extra={
                "mandatory_keywords": tokens or [title, exact_text],
                "edde_bucket": "fuzzy_fidelity",
            },
        )
        if st is not None:
            results.append(st)

    return results[:count]


def generate_self_correction_and_oos_edde(
    train_samples: list[Sample],
    catalog: dict[str, dict[str, Any]],
    count: int = 200,
    seed: int = 42,
    exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate Vietnamese self-correction repairs and missing-app conflict OOS [] EDDE steps."""
    rng = random.Random(seed)
    eligible = [s for s in _filter_eligible(train_samples, exclude_ids) if (s.audio_text or "").strip()]
    results: list[dict[str, Any]] = []
    if not eligible or count <= 0:
        return results

    rng.shuffle(eligible)
    cancel_suffixes = [
        " à thôi khoan đã, hủy toàn bộ yêu cầu đó đi giúp tôi, không cần làm nữa đâu.",
        " ủa nhầm, đừng làm nữa nhé, bỏ hết yêu cầu này giúp tôi.",
    ]

    for i, s in enumerate(eligible):
        if len(results) >= count:
            break
        if re.search(r"\b(?:nếu|thì\s+thôi)\b", s.audio_text or "", flags=re.IGNORECASE):
            continue
        calls, _ = _sample_calls_and_results(s)
        if not calls:
            continue
        used_apps = {c["name"].split(".", 1)[0] for c in calls if "." in c["name"]}
        ctx = copy.deepcopy(s.context or {})

        if i % 2 == 0:
            # Mode A: Spoken mid-utterance cancellation -> OOS []
            orig = (s.audio_text or "").rstrip(".!? ")
            repaired_u = f"{orig}{rng.choice(cancel_suffixes)}"
            synth = Sample(id=str(s.id), audio_text=repaired_u, history=list(s.history), context=ctx)
            prompt_msgs = build_sample_prefix_messages(synth, catalog)
            installed = resolve_installed_apps(synth, [c["name"] for c in calls])
            st = _build_verified_step(
                step_id=f"edde_sc_cancel_{s.id}_{i}",
                sample_id=str(s.id),
                step_category="oos",
                prompt_messages=prompt_msgs,
                step_calls=[],
                sample_context=ctx,
                tools=installed,
                catalog=catalog,
                exclude_ids=exclude_ids,
                extra={
                    "self_correction_note": "Người dùng tự sửa ý và hủy bỏ yêu cầu ở cuối câu -> trả về [].",
                    "edde_bucket": "self_correction_cancel",
                },
            )
            if st is not None:
                results.append(st)
        else:
            # Mode B: Remove required app and supply close-neighbor distractor apps -> OOS []
            inst = ctx.get("installed_apps")
            if not isinstance(inst, dict) or not used_apps:
                continue
            new_inst = {k: v for k, v in inst.items() if k not in used_apps}
            for grp_used, distractors in _SENSITIVE_DISTRACTOR_GROUPS:
                if used_apps & grp_used:
                    for d in distractors:
                        if d not in used_apps:
                            new_inst[d] = d
            if not new_inst:
                new_inst = {"notes": "notes", "calculator": "calculator"}
            ctx["installed_apps"] = new_inst
            synth = Sample(id=str(s.id), audio_text=s.audio_text, context=ctx)
            prompt_msgs = build_sample_prefix_messages(synth, catalog)
            st = _build_verified_step(
                step_id=f"edde_oos_conflict_{s.id}_{i}",
                sample_id=str(s.id),
                step_category="oos",
                prompt_messages=prompt_msgs,
                step_calls=[],
                sample_context=ctx,
                tools=list(new_inst.keys()),
                catalog=catalog,
                exclude_ids=exclude_ids,
                extra={"edde_bucket": "oos_conflict"},
            )
            if st is not None:
                results.append(st)

    return results[:count]


def _make_masked_default(old_val: Any, rng: random.Random) -> Any:
    if isinstance(old_val, bool):
        return not old_val
    if isinstance(old_val, int):
        return int(old_val) + rng.randint(17, 93)
    if isinstance(old_val, float):
        return round(float(old_val) + rng.uniform(1.5, 9.5), 2)
    suffix = "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(4))
    return f"def_{suffix}"


def _replace_token_in_text(text: str, old_val: str, new_val: str) -> str:
    if not text or not old_val:
        return text
    pattern = rf"(?<!\w){re.escape(old_val)}(?!\w)"
    return re.sub(pattern, new_val, text)


def mask_tools_and_params_hammer(
    step: dict[str, Any],
    *,
    rng: random.Random,
) -> dict[str, Any]:
    """Hammer V3 masking: obfuscate tool names, parameter names, and parameter default values."""
    masked = apply_hammer_masking(step, rng=rng)
    masked.pop("think", None)

    # Track (masked_tool_name, masked_param_name) -> (old_default, new_default)
    default_updates: dict[tuple[str, str], tuple[Any, Any]] = {}
    str_replacements: list[tuple[str, str]] = []

    for tool_entry in masked.get("tools") or []:
        fn = tool_entry.get("function") or {}
        t_name = str(fn.get("name") or "")
        props = ((fn.get("parameters") or {}).get("properties") or {})
        for p_alias, pspec in props.items():
            if not isinstance(pspec, dict) or "default" not in pspec:
                continue
            old_def = pspec["default"]
            if old_def is None or old_def == "":
                continue
            new_def = _make_masked_default(old_def, rng)
            pspec["default"] = new_def
            if isinstance(old_def, str) and isinstance(pspec.get("description"), str):
                pspec["description"] = _replace_token_in_text(pspec["description"], old_def, str(new_def))
            default_updates[(t_name, p_alias)] = (old_def, new_def)
            if isinstance(old_def, str) and len(old_def) >= 2:
                str_replacements.append((old_def, str(new_def)))

    if default_updates:
        def _update_call_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
            if not isinstance(args, dict):
                return args
            out = dict(args)
            for k, v in list(out.items()):
                pair = default_updates.get((name, k))
                if pair is not None and v == pair[0]:
                    out[k] = pair[1]
            return out

        for c in masked.get("step_calls") or []:
            c["arguments"] = _update_call_args(str(c.get("name") or ""), c.get("arguments") or {})
        for c in masked.get("prefix_calls") or []:
            c["arguments"] = _update_call_args(str(c.get("name") or ""), c.get("arguments") or {})

        for msg in masked.get("prompt_messages") or []:
            role = msg.get("role")
            if role in ("user", "system") and isinstance(msg.get("content"), str):
                txt = msg["content"]
                for old_s, new_s in str_replacements:
                    txt = _replace_token_in_text(txt, old_s, new_s)
                msg["content"] = txt
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function")
                if isinstance(fn, dict):
                    fn["arguments"] = _update_call_args(
                        str(fn.get("name") or ""), fn.get("arguments") or {}
                    )

    return masked
