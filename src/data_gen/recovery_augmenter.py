"""Synthetic in-loop error recovery trajectory augmentation for small LLM tool calling."""
from __future__ import annotations

import copy
from typing import Any

from src.core.schemas import ASK_MISSING_INFO

_ID_PARAM_NAMES = frozenset(
    {
        "device_id",
        "contact_id",
        "event_id",
        "alarm_id",
        "bill_id",
        "beneficiary_id",
        "from_account_id",
        "restaurant_id",
        "item_id",
        "track_id",
        "chat_id",
        "course_id",
        "note_id",
        "appliance_id",
        "bike_id",
        "subscription_id",
        "reminder_id",
    }
)

_READ_TO_PREMATURE_WRITE: dict[str, tuple[str, dict[str, Any]]] = {
    "smart_home.get_status": ("smart_home.light_set", {"device_id": "fake_dev_999", "power": "on"}),
    "calendar.list_events": ("calendar.update_event", {"event_id": "fake_dev_999"}),
    "bill_payment.list_bills": ("bill_payment.pay", {"bill_id": "fake_dev_999", "source": "bank"}),
    "banking.list_beneficiaries": (
        "banking.transfer",
        {"from_account_id": "fake_acc_999", "beneficiary_id": "fake_dev_999", "amount": 100000},
    ),
    "alarm.list": ("alarm.update", {"alarm_id": "fake_dev_999", "enabled": False}),
    "reminders.list": ("reminders.complete", {"reminder_id": "fake_dev_999"}),
}


def create_empty_search_recovery_step(
    base_step: dict[str, Any],
) -> dict[str, Any] | None:
    """Create an augmented step where assistant recovers from an empty initial search."""
    step_calls = base_step.get("step_calls", [])
    search_call = next((c for c in step_calls if c.get("name") == "contacts.search"), None)
    if not search_call:
        return None

    query = str((search_call.get("arguments") or {}).get("query", "")).strip()
    if len(query.split()) < 1:
        return None

    augmented = copy.deepcopy(base_step)
    augmented["id"] = f"{base_step['id']}_rec_empty"
    augmented["recovery_type"] = "empty_search"

    # Prepend a failed verbose search turn to prompt_messages; if other sibling calls were in step_calls,
    # treat them as already executed so target step_calls retries only the failed search.
    verbose_query = f"Chị {query} bên phòng nhân sự"
    augmented["failed_query"] = verbose_query
    failed_tool_call = {
        "id": "call_search_verbose",
        "type": "function",
        "function": {"name": "contacts.search", "arguments": {"query": verbose_query}},
    }
    sibling_calls = [c for c in step_calls if c is not search_call]
    assistant_calls = [
        {"type": "function", "function": {"name": str(c.get("name") or ""), "arguments": dict(c.get("arguments") or {})}}
        for c in sibling_calls
    ] + [failed_tool_call]
    tool_msgs = [
        {"role": "tool", "name": str(c.get("name") or ""), "content": '{"ok": true}'}
        for c in sibling_calls
    ] + [
        {"role": "tool", "name": "contacts.search", "content": "[]"}
    ]
    augmented["prompt_messages"].extend(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": assistant_calls,
            },
            *tool_msgs,
        ]
    )
    if sibling_calls:
        augmented["step_calls"] = [copy.deepcopy(search_call)]

    return augmented


def create_xref_repair_recovery_step(
    base_step: dict[str, Any],
    read_tool_name: str = "smart_home.get_status",
    *,
    require_grounded_context: bool = False,
    catalog: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Create an augmented step where assistant recovers from an invalid xref ID.

    Supports two causally sound recovery patterns:
    1. Step is a Read lookup tool (e.g. `smart_home.get_status`): prepends a premature write call
       with `fake_dev_999` + Unknown ID error, so the target action is calling the Read tool first.
    2. Step is a Write tool with an `*_id` argument (e.g. `device_id="light_1"`): prepends a failed
       write call with `fake_dev_999` + Unknown ID error, so the target action extracts the real ID
       from prior tool observations in `prompt_messages`.
    """
    step_calls = base_step.get("step_calls", [])
    if not step_calls:
        return None

    first_call = step_calls[0]
    first_name = str(first_call.get("name") or "")
    first_args = dict(first_call.get("arguments") or {})

    if first_name == ASK_MISSING_INFO:
        return None

    id_key = next((k for k in first_args if k in _ID_PARAM_NAMES or k.endswith("_id")), None)

    if first_name in _READ_TO_PREMATURE_WRITE:
        failed_name, raw_args = _READ_TO_PREMATURE_WRITE[first_name]
        failed_args = dict(raw_args)
    elif id_key is not None:
        id_val = str(first_args.get(id_key) or "")
        if require_grounded_context:
            prompt_text = " ".join(
                str(m.get("content") or "")
                for m in (base_step.get("prompt_messages") or [])
                if m.get("role") == "tool"
            )
            if not id_val or id_val not in prompt_text:
                return None
        failed_name = first_name
        failed_args = dict(first_args)
        failed_args[id_key] = "fake_dev_999"
    else:
        return None

    if catalog is not None:
        spec = catalog.get(failed_name)
        if not spec:
            return None
        props = ((spec.get("parameters") or {}).get("properties") or {})
        if props and any(k not in props for k in failed_args):
            return None

    augmented = copy.deepcopy(base_step)
    augmented["id"] = f"{base_step['id']}_rec_xref"
    augmented["recovery_type"] = "xref_repair"

    failed_tool_call = {
        "id": "call_premature_write",
        "type": "function",
        "function": {"name": failed_name, "arguments": failed_args},
    }
    err_payload = (
        '{"ok": false, "error": "Unknown ID: fake_dev_999", '
        '"hint": "Sửa lại tool/tham số hoặc gọi tool tra cứu ID trước rồi gọi lại."}'
    )
    augmented["prompt_messages"].extend(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [failed_tool_call],
            },
            {
                "role": "tool",
                "name": failed_name,
                "content": err_payload,
            },
        ]
    )

    return augmented
