"""Realistic unseen-domain schema aliasing for cross-app generalization SFT and probes."""
from __future__ import annotations

import copy
import random
import re
from typing import Any

from src.agent.tool_selector import compact_tool_spec
from src.core.schemas import ASK_MISSING_INFO

# Strictly disjoint from ALL 46 train apps AND all 11 public_test apps.
TRAIN_ALIAS_DOMAINS: tuple[str, ...] = (
    "agro_hub",
    "auto_dock",
    "book_shelf",
    "cargo_box",
    "civic_pass",
    "clinic_desk",
    "cloud_vault",
    "club_pass",
    "craft_hub",
    "diet_log",
    "doc_vault",
    "edu_port",
    "fleet_ops",
    "garden_care",
    "gear_shop",
    "health_band",
    "hotel_desk",
    "inv_ledger",
    "job_board",
    "laundry_go",
    "lease_hub",
    "mail_box",
    "meal_plan",
    "park_lot",
    "pay_desk",
    "pet_clinic",
    "print_hub",
    "pulse_care",
    "room_ops",
    "safe_box",
    "salon_book",
    "study_lab",
)

# Eval-only pool strictly disjoint from train, public_test, AND TRAIN_ALIAS_DOMAINS.
EVAL_ALIAS_DOMAINS: tuple[str, ...] = (
    "aqua_farm",
    "bakery_pos",
    "bike_dock",
    "cinema_box",
    "drone_port",
    "energy_grid",
    "ferry_pass",
    "gallery_ops",
    "gym_studio",
    "hostel_hub",
    "lab_queue",
    "marine_ops",
    "museum_go",
    "optik_care",
    "studio_rent",
    "wear_sync",
)

_READ_VERBS: tuple[str, ...] = (
    "fetch_records",
    "list_entries",
    "lookup_items",
    "query_status",
    "get_overview",
    "find_records",
    "browse_catalog",
    "inspect_state",
)

_WRITE_VERBS: tuple[str, ...] = (
    "submit_action",
    "create_record",
    "apply_update",
    "modify_entry",
    "remove_record",
    "dispatch_task",
    "register_item",
    "confirm_entry",
    "schedule_op",
    "cancel_entry",
)

_ID_PARAM_ALIASES: tuple[str, ...] = (
    "record_ref",
    "entry_uid",
    "target_id",
    "item_code",
    "slot_ref",
    "unit_ref",
    "resource_id",
)

_SCALAR_PARAM_ALIASES: dict[str, tuple[str, ...]] = {
    "query": ("keyword", "search_term", "filter_text"),
    "keyword": ("search_term", "filter_text"),
    "name": ("label", "item_title"),
    "title": ("heading", "subject_line"),
    "content": ("body_text", "memo_note", "details"),
    "body": ("message_text", "memo_note"),
    "subject": ("topic_line", "heading"),
    "date": ("target_date", "day_str"),
    "time": ("clock_time", "at_time"),
    "amount": ("total_value", "sum_val"),
    "location": ("place_name", "venue"),
    "destination": ("dropoff_place", "target_stop"),
    "origin": ("pickup_place", "start_point"),
    "source": ("pay_channel", "funding_method"),
    "category": ("group_tag", "section"),
    "status": ("state_flag", "mode_state"),
    "mode": ("run_mode", "op_mode"),
    "power": ("switch_state", "power_flag"),
    "enabled": ("is_active", "active_flag"),
    "brightness": ("light_level", "dim_value"),
    "temperature": ("target_temp", "temp_setting"),
}

_INSTALLED_LINE_RE = re.compile(r"^(-\s*installed_apps:\s*)(.*)$", re.MULTILINE)
_XREF_TAG_RE = re.compile(r"\[x-ref:\s*([^\]\s]+)\]")


def _replace_tool_and_app_text(
    text: str,
    tool_map: dict[str, str],
    app_map: dict[str, str],
) -> str:
    if not text:
        return text
    out = text
    for orig_t in sorted(tool_map, key=len, reverse=True):
        alias_t = tool_map[orig_t]
        out = out.replace(orig_t, alias_t)
    for orig_app in sorted(app_map, key=len, reverse=True):
        alias_app = app_map[orig_app]
        out = re.sub(rf"(?<![\w.]){re.escape(orig_app)}\.", f"{alias_app}.", out)
    return out


def _pick_param_alias(
    orig_param: str,
    pspec: dict[str, Any],
    used_params: set[str],
    rng: random.Random,
) -> str:
    is_id_like = orig_param.endswith("_id") or "x-ref" in pspec or "[x-ref:" in str(pspec.get("description") or "")
    if is_id_like:
        candidates = list(_ID_PARAM_ALIASES)
    elif orig_param in _SCALAR_PARAM_ALIASES:
        candidates = list(_SCALAR_PARAM_ALIASES[orig_param])
    else:
        candidates = [f"field_{orig_param}", f"opt_{orig_param}", f"val_{orig_param}"]
    rng.shuffle(candidates)
    for cand in candidates:
        if cand not in used_params:
            return cand
    idx = 2
    while f"param_{idx}" in used_params:
        idx += 1
    return f"param_{idx}"


def _remap_call(
    name: str,
    args: dict[str, Any],
    tool_map: dict[str, str],
    param_map: dict[str, dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    if name == ASK_MISSING_INFO:
        out_args = dict(args or {})
        orig_target = str(out_args.get("tool_name") or "").strip()
        if orig_target in tool_map:
            out_args["tool_name"] = tool_map[orig_target]
            pmap = param_map.get(orig_target, {})
            raw_params = out_args.get("params")
            if isinstance(raw_params, list):
                out_args["params"] = [pmap.get(str(p), str(p)) for p in raw_params]
        return ASK_MISSING_INFO, out_args

    alias_t = tool_map.get(name, name)
    pmap = param_map.get(name, {})
    new_args = {pmap.get(k, k): v for k, v in (args or {}).items()}
    return alias_t, new_args


def _extract_tool_schemas(
    raw_tools: list[Any],
    catalog: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in raw_tools:
        if isinstance(item, dict) and "function" in item:
            out.append(copy.deepcopy(item))
        elif isinstance(item, str) and item in catalog:
            out.append(compact_tool_spec(catalog[item]))
    return out


def alias_domain_step(
    step: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    *,
    rng: random.Random,
    domain_pool: tuple[str, ...] = TRAIN_ALIAS_DOMAINS,
    alias_params_prob: float = 0.75,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Rewrite a trajectory step onto realistic unseen domain/tool/parameter names.

    Returns ``(aliased_step, aliased_catalog)`` with consistent tool schemas, ``x-ref`` metadata,
    ``installed_apps`` context line, multi-turn history calls/observations, and ``step_calls``.
    """
    aliased = copy.deepcopy(step)
    tool_schemas = _extract_tool_schemas(aliased.get("tools") or [], catalog)

    apps_found: list[str] = []
    for t in tool_schemas:
        fn_name = str((t.get("function") or {}).get("name") or "")
        if "." in fn_name and not fn_name.startswith("sys."):
            app = fn_name.split(".", 1)[0]
            if app not in apps_found:
                apps_found.append(app)
    ctx_inst = (aliased.get("sample_context") or {}).get("installed_apps") or {}
    if isinstance(ctx_inst, dict):
        for app in ctx_inst:
            if app != "sys" and app not in apps_found:
                apps_found.append(app)

    pool = list(domain_pool)
    rng.shuffle(pool)
    app_map: dict[str, str] = {}
    for idx, orig_app in enumerate(apps_found):
        app_map[orig_app] = pool[idx] if idx < len(pool) else f"domain_ext_{idx}"

    tool_map: dict[str, str] = {}
    used_tools_by_app: dict[str, set[str]] = {}
    param_map: dict[str, dict[str, str]] = {}
    rename_params = rng.random() < alias_params_prob

    for t in tool_schemas:
        fn = t.get("function") or {}
        orig_name = str(fn.get("name") or "")
        if not orig_name or orig_name == ASK_MISSING_INFO or orig_name.startswith("sys."):
            continue
        orig_app = orig_name.split(".", 1)[0] if "." in orig_name else ""
        alias_app = app_map.get(orig_app, orig_app)
        used_names = used_tools_by_app.setdefault(alias_app, set())

        kind = str((catalog.get(orig_name) or {}).get("kind") or "")
        if not kind:
            is_read = any(tok in orig_name for tok in ("list", "search", "get", "query", "status", "history"))
            kind = "read" if is_read else "write"
        verb_pool = list(_READ_VERBS if kind == "read" else _WRITE_VERBS)
        rng.shuffle(verb_pool)
        chosen = next((v for v in verb_pool if f"{alias_app}.{v}" not in used_names), None)
        if chosen is None:
            chosen = f"op_{len(used_names) + 1}"
        alias_t = f"{alias_app}.{chosen}"
        used_names.add(alias_t)
        tool_map[orig_name] = alias_t

        props = ((fn.get("parameters") or {}).get("properties") or {})
        pmap: dict[str, str] = {}
        used_p: set[str] = set()
        for pk, pspec in props.items():
            if rename_params and isinstance(pspec, dict):
                new_p = _pick_param_alias(pk, pspec, used_p, rng)
            else:
                new_p = pk
            pmap[pk] = new_p
            used_p.add(new_p)
        param_map[orig_name] = pmap

    new_tools: list[dict[str, Any]] = []
    aliased_catalog: dict[str, dict[str, Any]] = {}
    for t in tool_schemas:
        fn = t.get("function") or {}
        orig_name = str(fn.get("name") or "")
        if orig_name not in tool_map:
            new_tools.append(t)
            if orig_name in catalog:
                aliased_catalog[orig_name] = copy.deepcopy(catalog[orig_name])
            continue

        alias_t = tool_map[orig_name]
        pmap = param_map[orig_name]
        orig_spec = catalog.get(orig_name) or {}
        orig_kind = str(orig_spec.get("kind") or "write")
        orig_props = ((orig_spec.get("parameters") or {}).get("properties") or {})

        fn["name"] = alias_t
        if isinstance(fn.get("description"), str):
            fn["description"] = _replace_tool_and_app_text(fn["description"], tool_map, app_map)

        params = fn.get("parameters") or {"type": "object", "properties": {}}
        props = params.get("properties") or {}
        new_props: dict[str, Any] = {}
        cat_props: dict[str, Any] = {}
        for old_p, pspec in props.items():
            new_p = pmap.get(old_p, old_p)
            ps = copy.deepcopy(pspec) if isinstance(pspec, dict) else pspec
            if isinstance(ps, dict):
                if isinstance(ps.get("description"), str):
                    desc = _replace_tool_and_app_text(ps["description"], tool_map, app_map)
                    for op_k, np_v in pmap.items():
                        if op_k != np_v and len(op_k) >= 4:
                            desc = re.sub(rf"(?<!\w){re.escape(op_k)}(?!\w)", np_v, desc)
                    ps["description"] = desc
                raw_xref = ps.get("x-ref") or (orig_props.get(old_p) or {}).get("x-ref")
                if raw_xref:
                    mapped_xref = _replace_tool_and_app_text(str(raw_xref), tool_map, app_map)
                else:
                    m_xref = _XREF_TAG_RE.search(str(ps.get("description") or ""))
                    mapped_xref = m_xref.group(1) if m_xref else None
                new_props[new_p] = ps
                cat_ps = copy.deepcopy(ps)
                if mapped_xref:
                    cat_ps["x-ref"] = mapped_xref
                cat_props[new_p] = cat_ps
            else:
                new_props[new_p] = ps
                cat_props[new_p] = ps

        params["properties"] = new_props
        if isinstance(params.get("required"), list):
            params["required"] = [pmap.get(r, r) for r in params["required"]]
        if isinstance(params.get("x-one-of"), list):
            params["x-one-of"] = [[pmap.get(r, r) for r in grp] for grp in params["x-one-of"]]
        fn["parameters"] = params
        new_tools.append(t)

        cat_params = copy.deepcopy(params)
        cat_params["properties"] = cat_props
        aliased_catalog[alias_t] = {
            "name": alias_t,
            "kind": orig_kind,
            "description": fn.get("description", ""),
            "parameters": cat_params,
            "app": app_map.get(orig_name.split(".", 1)[0], ""),
        }

    aliased["tools"] = new_tools

    # Update sample_context.installed_apps without disturbing Rule 1-9 text in system prompt
    if isinstance(aliased.get("sample_context"), dict):
        inst = aliased["sample_context"].get("installed_apps")
        if isinstance(inst, dict):
            new_inst = {}
            for k, v in inst.items():
                ak = app_map.get(k, k)
                if isinstance(v, dict) and isinstance(v.get("tools"), list):
                    nv = dict(v)
                    nv["tools"] = [tool_map.get(tn, tn) for tn in v["tools"]]
                    new_inst[ak] = nv
                else:
                    new_inst[ak] = ak if ak != k else v
            aliased["sample_context"]["installed_apps"] = new_inst

    def _remap_msg(msg: dict[str, Any]) -> None:
        role = msg.get("role")
        if role == "system" and isinstance(msg.get("content"), str):
            def _repl_installed(m: re.Match[str]) -> str:
                raw_apps = [a.strip() for a in m.group(2).split(",") if a.strip()]
                mapped_apps = sorted({app_map.get(a, a) for a in raw_apps})
                return f"{m.group(1)}{', '.join(mapped_apps)}"

            msg["content"] = _INSTALLED_LINE_RE.sub(_repl_installed, msg["content"])
        elif role == "tool":
            tname = str(msg.get("name") or "")
            if tname in tool_map:
                msg["name"] = tool_map[tname]
            if isinstance(msg.get("content"), str):
                msg["content"] = _replace_tool_and_app_text(msg["content"], tool_map, app_map)
        elif role == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") if isinstance(tc, dict) else None
                if isinstance(fn, dict) and fn.get("name"):
                    nn, na = _remap_call(str(fn["name"]), fn.get("arguments") or {}, tool_map, param_map)
                    fn["name"] = nn
                    fn["arguments"] = na

    for m in aliased.get("prompt_messages") or []:
        _remap_msg(m)
    if isinstance(aliased.get("completion_message"), dict):
        _remap_msg(aliased["completion_message"])

    aliased["step_calls"] = [
        {"name": nn, "arguments": na}
        for c in (aliased.get("step_calls") or [])
        for nn, na in [_remap_call(str(c.get("name") or ""), c.get("arguments") or {}, tool_map, param_map)]
    ]
    if isinstance(aliased.get("prefix_calls"), list):
        aliased["prefix_calls"] = [
            {"name": nn, "arguments": na}
            for c in aliased["prefix_calls"]
            for nn, na in [_remap_call(str(c.get("name") or ""), c.get("arguments") or {}, tool_map, param_map)]
        ]
    aliased["alias_app_map"] = app_map
    return aliased, aliased_catalog


def update_installed_apps_in_system(prompt_messages: list[dict[str, Any]], app_names: list[str]) -> None:
    """Update the `- installed_apps:` metadata line in system messages without touching prompt rules."""
    apps_str = ", ".join(sorted({a for a in app_names if a}))
    for m in prompt_messages:
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            m["content"] = _INSTALLED_LINE_RE.sub(lambda match: f"{match.group(1)}{apps_str}", m["content"])
