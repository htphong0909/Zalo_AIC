#!/usr/bin/env python3
"""Build anti-overfit, schema-generalizing LoRA dataset (`qwen35_train_v2r_general.jsonl.gz`)
and 100-sample Unseen-Schema Eval Probe (`unseen_schema_eval.jsonl`) with strict zero-leakage."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_lfm3_data import apply_hammer_masking, simulate_ctc_transcript
from src.agent.prompts import build_system_prompt
from src.agent.qwen35_sft import _THINK_STRIP_RE, step_to_qwen35_example
from src.agent.sft_dataset import build_sft_trajectory, unroll_trajectory_steps
from src.agent.tool_selector import compact_tool_spec
from src.core.dataset import load_catalog, load_split
from src.core.schemas import ASK_MISSING_INFO, Sample
from src.data_gen.edde_augmenter import assert_not_excluded
from src.data_gen.execution_verifier import verify_step_validity
from src.data_gen.prompt_utils import build_sample_prefix_messages, resolve_installed_apps
from src.data_gen.recovery_augmenter import (
    create_empty_search_recovery_step,
    create_xref_repair_recovery_step,
)
from src.data_gen.schema_aliasing import (
    EVAL_ALIAS_DOMAINS,
    TRAIN_ALIAS_DOMAINS,
    alias_domain_step,
    update_installed_apps_in_system,
)
from src.data_gen.when2call_gen import (
    generate_post_lookup_ask_samples,
    generate_when2call_contrastive_samples,
)

HELD_OUT_UNSEEN_APPS: tuple[str, ...] = (
    "appliances",
    "exchange_rate",
    "fitness",
    "library",
    "medication",
    "podcast_radio",
)

_SCALAR_CREATE_TOOLS: frozenset[str] = frozenset(
    {
        "reminders.create",
        "notes.create",
        "email.create_draft",
        "shopping_list.add",
        "calendar.create_event",
        "tasks.add",
        "sms.send",
    }
)

_CONDITIONAL_RE = re.compile(r"\b(nếu|trường hợp|không thì thôi|đừng|khỏi)\b", re.IGNORECASE)
_ID_ARG_RE = re.compile(r"(?:^id$|_id$|_ref$|_uid$|_code$)")


def _load_id_set(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {str(x) for x in data}
    if isinstance(data, dict):
        for key in ("dev_ids", "ids", "train_ids"):
            if key in data and isinstance(data[key], list):
                return {str(x) for x in data[key]}
    return set()


def _load_asr_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = str(row.get("id") or row.get("sample_id") or "").strip()
        if not sid:
            continue
        raw_hist = row.get("history_texts") or {}
        hist_int: dict[int, str] = {}
        if isinstance(raw_hist, dict):
            for k, v in raw_hist.items():
                try:
                    hist_int[int(k)] = str(v)
                except (TypeError, ValueError):
                    continue
        out[sid] = {
            "audio_text": str(row.get("audio_text") or row.get("transcript") or "").strip(),
            "history_texts": hist_int,
            "asr_phonetic_diff": str(row.get("asr_phonetic_diff") or row.get("phonetic_diff") or "").strip(),
        }
    return out


def _clean_prompt_messages_qwen35(
    msgs: list[dict[str, Any]],
    sample: Sample,
) -> list[dict[str, Any]]:
    """Ensure system prompt uses current qwen35 prompt (with Rule 9) & strips stale XML."""
    out: list[dict[str, Any]] = []
    q35_sys = build_system_prompt(Sample(id=sample.id, context=sample.context), prompt_style="qwen35")
    for m in msgs:
        nm = copy.deepcopy(m)
        role = nm.get("role")
        if role == "system":
            nm["content"] = q35_sys
        elif role == "assistant" and nm.get("tool_calls"):
            nm["content"] = _THINK_STRIP_RE.sub("", str(nm.get("content") or "")).strip()
        out.append(nm)
    return out


def _unroll_sample_steps(
    sample: Sample,
    catalog: dict[str, dict[str, Any]],
    *,
    current_text: str | None = None,
    history_texts: dict[int, str] | None = None,
    max_result_chars: int = 3500,
) -> list[dict[str, Any]]:
    traj = build_sft_trajectory(
        sample,
        catalog,
        current_text=current_text,
        history_texts=history_texts,
        max_result_chars=max_result_chars,
    )
    steps = unroll_trajectory_steps(traj)
    ctx = copy.deepcopy(sample.context or {})
    out: list[dict[str, Any]] = []
    for st in steps:
        st["sample_id"] = str(sample.id)
        st["sample_context"] = copy.deepcopy(ctx)
        st["prompt_messages"] = _clean_prompt_messages_qwen35(st.get("prompt_messages") or [], sample)
        if verify_step_validity(st, catalog):
            out.append(st)
    return out


def _apply_hammer_with_catalog(
    step: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    *,
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]] | None:
    masked = apply_hammer_masking(step, rng=rng)
    masked.pop("think", None)
    masked_cat: dict[str, dict[str, Any]] = {}
    pseudo_apps: list[str] = []
    for t in masked.get("tools") or []:
        fn = t.get("function") or {}
        name = str(fn.get("name") or "")
        if not name:
            continue
        if name == ASK_MISSING_INFO and name in catalog:
            masked_cat[name] = copy.deepcopy(catalog[name])
            continue
        pseudo_apps.append(name.split(".", 1)[0] if "." in name else name)
        params = copy.deepcopy(fn.get("parameters") or {"type": "object", "properties": {}})
        props = params.get("properties") or {}
        for ps in props.values():
            if isinstance(ps, dict) and "x-ref" not in ps:
                m = re.search(r"\[x-ref:\s*([^\]\s]+)\]", str(ps.get("description") or ""))
                if m:
                    ps["x-ref"] = m.group(1)
        masked_cat[name] = {
            "name": name,
            "kind": "write",
            "description": fn.get("description", ""),
            "parameters": params,
        }
    update_installed_apps_in_system(masked.get("prompt_messages") or [], pseudo_apps)
    if not verify_step_validity(masked, masked_cat):
        return None
    return masked, masked_cat


def _make_disambiguation_anti_false_ask(
    step: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    *,
    rng: random.Random,
    long_mode: bool = False,
) -> dict[str, Any] | None:
    """Create a post-lookup ACT step with decoy items (inactive / different date) or long JSON list."""
    if step.get("step_category") != "act":
        return None
    calls = step.get("step_calls") or []
    if len(calls) != 1:
        return None
    target = calls[0]
    tname = str(target.get("name") or "")
    targs = dict(target.get("arguments") or {})
    id_entry = next(((k, v) for k, v in targs.items() if _ID_ARG_RE.search(k) and isinstance(v, str) and v), None)
    if not id_entry:
        return None
    _, real_id = id_entry

    pmsgs = copy.deepcopy(step.get("prompt_messages") or [])
    tool_indices = [i for i, m in enumerate(pmsgs) if m.get("role") == "tool" and real_id in str(m.get("content") or "")]
    if not tool_indices:
        return None
    t_idx = tool_indices[-1]
    raw_content = str(pmsgs[t_idx].get("content") or "")
    try:
        parsed = json.loads(raw_content)
    except Exception:
        return None

    gold_item: dict[str, Any] = {"id": real_id, "name": "Mục đã chọn", "status": "active"}
    decoys = (
        [
            {"id": f"decoy_old_{rng.randint(10, 99)}", "name": "Mục cũ đã hoàn tất", "status": "ended", "date": "2020-01-01"},
            {"id": f"decoy_canc_{rng.randint(10, 99)}", "name": "Bản ghi đã hủy", "status": "cancelled", "date": "2020-01-02"},
        ]
        if not long_mode
        else [
            {
                "id": f"item_pad_{j:02d}",
                "name": f"Bản ghi lưu trữ số {j}",
                "status": "archived" if j % 2 == 0 else "ended",
                "note": "Dữ liệu lịch sử cũ không khớp yêu cầu hiện tại của người dùng",
            }
            for j in range(1, 13)
        ]
    )

    injected = False
    if isinstance(parsed, dict):
        for k, v in list(parsed.items()):
            if isinstance(v, list) and any(isinstance(el, dict) and real_id in json.dumps(el, ensure_ascii=False) for el in v):
                for el in v:
                    if isinstance(el, dict) and real_id in json.dumps(el, ensure_ascii=False) and "status" not in el:
                        el["status"] = "active"
                if long_mode:
                    # Place matching record later in the list while staying well under 3500 chars
                    parsed[k] = decoys[:9] + list(v) + decoys[9:]
                else:
                    parsed[k] = list(v) + decoys
                injected = True
                break
        if not injected:
            parsed["_candidates"] = decoys + [gold_item]
            injected = True
    elif isinstance(parsed, list):
        if long_mode:
            parsed = decoys[:9] + parsed + decoys[9:]
        else:
            parsed = parsed + decoys
        injected = True

    if not injected:
        return None
    serialized = json.dumps(parsed, ensure_ascii=False)
    if len(serialized) > 3300:
        return None
    pmsgs[t_idx]["content"] = serialized

    out = copy.deepcopy(step)
    out["id"] = f"{step['id']}_{'long' if long_mode else 'disambig'}_{rng.randint(100, 999)}"
    out["prompt_messages"] = pmsgs
    return out


def _make_generic_empty_search_recovery(
    step: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    *,
    rng: random.Random,
) -> dict[str, Any] | None:
    """Create recovery teaching `tool({})` after `tool(query/keyword=...)` returned empty list."""
    if step.get("step_category") != "act":
        return None
    calls = step.get("step_calls") or []
    if len(calls) != 1:
        return None
    c = calls[0]
    cname = str(c.get("name") or "")
    cargs = dict(c.get("arguments") or {})
    if cname == "contacts.search":
        return create_empty_search_recovery_step(step)
    spec = catalog.get(cname) or {}
    if spec.get("kind") != "read" or cargs:
        return None
    params = spec.get("parameters") or {}
    if params.get("required"):
        return None
    props = params.get("properties") or {}
    str_filter = next(
        (pk for pk, ps in props.items() if isinstance(ps, dict) and ps.get("type") == "string" and not ps.get("enum")),
        None,
    )
    if not str_filter:
        return None

    out = copy.deepcopy(step)
    out["id"] = f"{step['id']}_empty_read_{rng.randint(100, 999)}"
    out["prompt_messages"].extend(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": cname, "arguments": {str_filter: "mục cần tìm"}}}
                ],
            },
            {
                "role": "tool",
                "name": cname,
                "content": json.dumps(
                    {
                        "items": [],
                        "ok": True,
                        "hint": f"Không có kết quả với bộ lọc vừa rồi; thử từ khoá ngắn hơn hoặc gọi {cname}({{}}) không tham số để xem toàn bộ danh sách.",
                    },
                    ensure_ascii=False,
                ),
            },
        ]
    )
    return out


def _make_irrelevance_oos_step(
    sample: Sample,
    catalog: dict[str, dict[str, Any]],
    *,
    rng: random.Random,
) -> dict[str, Any] | None:
    calls = getattr(sample.label, "tool_calls", None) or []
    if not calls:
        return None
    used_apps = {c.name.split(".", 1)[0] for c in calls if "." in c.name}
    ctx = copy.deepcopy(sample.context or {})
    inst = ctx.get("installed_apps")
    if not isinstance(inst, dict) or not used_apps:
        return None
    reduced = {k: v for k, v in inst.items() if k not in used_apps}
    if not reduced:
        reduced = {"weather": "weather", "calculator": "calculator"}
    ctx["installed_apps"] = reduced
    mod = Sample(id=sample.id, audio_text=sample.audio_text, history=list(sample.history), context=ctx)
    pmsgs = build_sample_prefix_messages(mod, catalog, max_result_chars=3500)
    pmsgs = _clean_prompt_messages_qwen35(pmsgs, mod)
    tools = []
    seen_tools: set[str] = set()
    for tname, spec in catalog.items():
        app = tname.split(".", 1)[0] if "." in tname else ""
        if app in reduced or app == "sys":
            if tname not in seen_tools:
                tools.append(compact_tool_spec(spec))
                seen_tools.add(tname)
    return {
        "id": f"irr_oos_{sample.id}_{rng.randint(10, 99)}",
        "sample_id": str(sample.id),
        "step_index": 0,
        "step_category": "oos",
        "prompt_messages": pmsgs,
        "step_calls": [],
        "tools": tools,
        "sample_context": ctx,
    }


def _make_itn_email_variant(
    sample: Sample,
    catalog: dict[str, dict[str, Any]],
    *,
    rng: random.Random,
    asr_row: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build normalized spoken-email & real-PhoWhisper AURA phonetic-diff examples."""
    from src.asr.aura_nbest import extract_phonetic_diff

    calls = getattr(sample.label, "tool_calls", None) or []
    if not calls or any(c.name == ASK_MISSING_INFO for c in calls):
        return None
    if len(calls) == 1:
        c = calls[0]
        args = dict(c.arguments or {})
        email_key = next((k for k, v in args.items() if isinstance(v, str) and "@" in v and "." in v), None)
        if email_key:
            orig_email = args[email_key]
            new_email = rng.choice(["final@example.test", "lan.nguyen@company.vn", "support.team@hanoi.org"])
            new_args = dict(args)
            new_args[email_key] = new_email
            text = str(sample.audio_text or "").replace(orig_email, new_email)
            if new_email not in text:
                text = f"{text} (email: {new_email})"
            mod_sample = Sample(
                id=sample.id,
                audio_text=text,
                history=list(sample.history),
                context=copy.deepcopy(sample.context or {}),
                label={"tool_calls": [{"name": c.name, "arguments": new_args}], "tool_results": [{"ok": True}]},
            )
            steps = _unroll_sample_steps(mod_sample, catalog)
            if steps:
                steps[0]["id"] = f"itn_email_{sample.id}_{rng.randint(10, 99)}"
                return steps[0]

    # Real PhoWhisper ASR + AURA phonetic diff against clean transcript (simulates N-best ambiguity)
    gold_txt = str(sample.audio_text or "").strip()
    asr_txt = str((asr_row or {}).get("audio_text") or "").strip()
    if asr_txt and gold_txt and asr_txt != gold_txt:
        diff_hint = extract_phonetic_diff([asr_txt, gold_txt])
        if diff_hint:
            steps = _unroll_sample_steps(
                sample,
                catalog,
                current_text=f"{asr_txt}\n{diff_hint}",
                history_texts=(asr_row or {}).get("history_texts") or None,
            )
            if steps:
                steps[0]["id"] = f"aura_real_{sample.id}_{rng.randint(10, 99)}"
                return steps[0]

    words = gold_txt.split()
    if len(words) >= 5:
        noisy = simulate_ctc_transcript(gold_txt)
        diff_hint = extract_phonetic_diff([noisy, gold_txt]) if noisy else None
        if noisy and noisy != gold_txt and diff_hint:
            steps = _unroll_sample_steps(sample, catalog, current_text=f"{noisy}\n{diff_hint}")
            if steps:
                steps[0]["id"] = f"aura_diff_{sample.id}_{rng.randint(10, 99)}"
                return steps[0]
    return None


def _expand_tools_to_schemas(
    raw_tools: list[Any],
    catalog: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand mixed app names / tool names / dict specs into full OpenAI tool schema dicts."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    app_names: list[str] = []
    for t in raw_tools or []:
        if isinstance(t, dict) and "function" in t:
            name = str((t.get("function") or {}).get("name") or "")
            if name and name not in seen:
                out.append(t)
                seen.add(name)
        elif isinstance(t, str):
            if t in catalog:
                if t not in seen:
                    out.append(compact_tool_spec(catalog[t]))
                    seen.add(t)
            else:
                app_names.append(t)
    if app_names:
        app_set = set(app_names) | {"sys"}
        for tname, spec in catalog.items():
            if tname.split(".", 1)[0] in app_set and tname not in seen:
                out.append(compact_tool_spec(spec))
                seen.add(tname)
    return out


def _prune_step_distractor_apps(
    step: dict[str, Any],
    tools: list[dict[str, Any]],
    *,
    rng: random.Random,
    max_apps: int = 5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Keep all gold/history/target apps plus up to `max_apps` total apps to shrink long prompts safely."""
    req_apps: set[str] = set()

    def _collect_call(name: str, args: dict[str, Any]) -> None:
        if name == ASK_MISSING_INFO:
            tgt = str((args or {}).get("tool_name") or "")
            if "." in tgt:
                req_apps.add(tgt.split(".", 1)[0])
        elif "." in name and not name.startswith("sys."):
            req_apps.add(name.split(".", 1)[0])

    for c in step.get("step_calls") or []:
        _collect_call(str(c.get("name") or ""), c.get("arguments") or {})
    for c in step.get("prefix_calls") or []:
        _collect_call(str(c.get("name") or ""), c.get("arguments") or {})
    for m in step.get("prompt_messages") or []:
        if m.get("role") == "tool" and m.get("name"):
            tname = str(m["name"])
            if "." in tname and not tname.startswith("sys."):
                req_apps.add(tname.split(".", 1)[0])
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") if isinstance(tc, dict) else None
            if isinstance(fn, dict) and fn.get("name"):
                _collect_call(str(fn["name"]), fn.get("arguments") or {})

    avail_apps = sorted(
        {
            str((t.get("function") or {}).get("name") or "").split(".", 1)[0]
            for t in tools
            if "." in str((t.get("function") or {}).get("name") or "")
            and not str((t.get("function") or {}).get("name") or "").startswith("sys.")
        }
    )
    if len(avail_apps) <= max_apps:
        return step, tools

    distractors = [a for a in avail_apps if a not in req_apps]
    rng.shuffle(distractors)
    extra_slots = max(1 if not req_apps else 0, max_apps - len(req_apps))
    kept_apps = req_apps | set(distractors[:extra_slots])

    pruned_tools = [
        t for t in tools
        if (str((t.get("function") or {}).get("name") or "").split(".", 1)[0] in kept_apps)
        or str((t.get("function") or {}).get("name") or "") == ASK_MISSING_INFO
        or str((t.get("function") or {}).get("name") or "").startswith("sys.")
    ]
    out_step = copy.deepcopy(step)
    if isinstance(out_step.get("sample_context"), dict):
        inst = out_step["sample_context"].get("installed_apps")
        if isinstance(inst, dict):
            out_step["sample_context"]["installed_apps"] = {k: v for k, v in inst.items() if k in kept_apps}
    update_installed_apps_in_system(out_step.get("prompt_messages") or [], sorted(kept_apps))
    return out_step, pruned_tools


def _finalize_record(
    step: dict[str, Any],
    catalog_for_val: dict[str, dict[str, Any]],
    source: str,
    exclude_ids: set[str],
    rng: random.Random,
    *,
    max_prompt_apps: int | None = None,
) -> dict[str, Any] | None:
    sid = str(step.get("sample_id") or "")
    assert_not_excluded(sid, exclude_ids)
    if not verify_step_validity(step, catalog_for_val):
        return None
    tools = _expand_tools_to_schemas(step.get("tools") or [], catalog_for_val)
    if not tools:
        return None
    if max_prompt_apps is not None and source != "hammer_masked":
        step, tools = _prune_step_distractor_apps(step, tools, rng=rng, max_apps=max_prompt_apps)
    ctx = step.get("sample_context") or {}
    ex = step_to_qwen35_example(step, ctx, tools, catalog_for_val, enable_thinking=False, rng=rng)
    approx_chars = len(json.dumps(ex["prompt_messages"], ensure_ascii=False)) + len(json.dumps(ex["tools"], ensure_ascii=False))
    if approx_chars > 34000:
        return None
    ex["id"] = str(step.get("id") or f"{source}_{sid}")
    ex["sample_id"] = sid
    ex["step_category"] = str(step.get("step_category") or "act")
    ex["source"] = source
    ex["prompt_style"] = "qwen35"
    return ex


def build_unseen_schema_probe(
    held_out_samples: list[Sample],
    catalog: dict[str, dict[str, Any]],
    asr_map: dict[str, dict[str, Any]],
    *,
    forbidden_ids: set[str],
    seed: int = 2027,
) -> list[dict[str, Any]]:
    """Build deterministic 100-sample Unseen-Schema Probe strictly disjoint from train & Dev 299."""
    rng = random.Random(seed)
    probe: list[dict[str, Any]] = []

    def _alias_eval(st: dict[str, Any], sid: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        assert_not_excluded(sid, forbidden_ids)
        return alias_domain_step(
            st,
            catalog,
            rng=rng,
            domain_pool=EVAL_ALIAS_DOMAINS,
            alias_params_prob=1.0,
        )

    # 1. unseen_step0 (20 rows: step-0 read/direct write on held-out apps -> eval alias domain)
    step0_cands: list[tuple[Sample, dict[str, Any]]] = []
    post_rw_cands: list[tuple[Sample, dict[str, Any]]] = []
    anti_ask_cands: list[tuple[Sample, dict[str, Any]]] = []
    oos_source: list[Sample] = []

    for idx, s in enumerate(held_out_samples):
        assert_not_excluded(s.id, forbidden_ids)
        oos_source.append(s)
        asr = asr_map.get(s.id) if idx % 2 == 0 else None
        steps = _unroll_sample_steps(
            s,
            catalog,
            current_text=(asr["audio_text"] if asr and asr.get("audio_text") else None),
            history_texts=(asr["history_texts"] if asr and asr.get("history_texts") else None),
        )
        for st in steps:
            cat = st.get("step_category")
            if cat != "act":
                continue
            if st.get("step_index") == 0:
                step0_cands.append((s, st))
                calls = st.get("step_calls") or []
                if calls and calls[0].get("name") in _SCALAR_CREATE_TOOLS:
                    anti_ask_cands.append((s, st))
            else:
                post_rw_cands.append((s, st))

    rng.shuffle(step0_cands)
    for idx, (s, st) in enumerate(step0_cands[:30]):
        if sum(1 for r in probe if r["probe_category"] == "unseen_step0") >= 20:
            break
        aliased, acat = _alias_eval(st, s.id)
        aliased["id"] = f"probe_step0_{idx}_{s.id}"
        rec = _finalize_record(aliased, acat, "probe_unseen_step0", forbidden_ids, rng)
        if rec:
            rec["probe_category"] = "unseen_step0"
            probe.append(rec)

    # 2. unseen_post_read_write (20 rows)
    rng.shuffle(post_rw_cands)
    for idx, (s, st) in enumerate(post_rw_cands):
        if sum(1 for r in probe if r["probe_category"] == "unseen_post_read_write") >= 20:
            break
        aliased, acat = _alias_eval(st, s.id)
        aliased["id"] = f"probe_postrw_{idx}_{s.id}"
        rec = _finalize_record(aliased, acat, "probe_unseen_post_read_write", forbidden_ids, rng)
        if rec:
            rec["probe_category"] = "unseen_post_read_write"
            probe.append(rec)

    # Fallback top-up from step0_cands using _make_disambiguation_anti_false_ask or empty search if needed
    # 3. unseen_anti_false_ask (20 rows: scalar direct + unique status disambiguation + empty-search {})
    anti_pool: list[tuple[Sample, dict[str, Any]]] = list(anti_ask_cands)
    for s, st in post_rw_cands:
        d = _make_disambiguation_anti_false_ask(st, catalog, rng=rng)
        if d:
            anti_pool.append((s, d))
    for s, st in step0_cands:
        e = _make_generic_empty_search_recovery(st, catalog, rng=rng)
        if e:
            anti_pool.append((s, e))
    rng.shuffle(anti_pool)
    for idx, (s, st) in enumerate(anti_pool):
        if sum(1 for r in probe if r["probe_category"] == "unseen_anti_false_ask") >= 20:
            break
        aliased, acat = _alias_eval(st, s.id)
        aliased["id"] = f"probe_antifalseask_{idx}_{s.id}"
        rec = _finalize_record(aliased, acat, "probe_unseen_anti_false_ask", forbidden_ids, rng)
        if rec:
            rec["probe_category"] = "unseen_anti_false_ask"
            probe.append(rec)

    # 4. unseen_genuine_ask (20 rows: required non-content scalar omission + post-lookup duplicate ambiguity)
    w2c_asks = generate_when2call_contrastive_samples(held_out_samples, catalog, count=30, seed=seed + 11)
    post_asks = generate_post_lookup_ask_samples(held_out_samples, catalog, count=25, seed=seed + 12)
    ask_pool: list[dict[str, Any]] = list(w2c_asks) + list(post_asks)

    # Top up with generic post-lookup ambiguous duplicate ID cases from held-out multi-step trajectories
    for _, st in post_rw_cands:
        calls = st.get("step_calls") or []
        if len(calls) != 1:
            continue
        tname = str(calls[0].get("name") or "")
        id_k = next((k for k in (calls[0].get("arguments") or {}) if _ID_ARG_RE.search(k)), None)
        if not id_k:
            continue
        pmsgs = copy.deepcopy(st.get("prompt_messages") or [])
        t_idx = next((i for i in range(len(pmsgs) - 1, -1, -1) if pmsgs[i].get("role") == "tool"), None)
        if t_idx is None:
            continue
        pmsgs[t_idx]["content"] = json.dumps(
            {
                "ok": True,
                "items": [
                    {"id": "dup_item_1", "name": "Bản ghi trùng tên", "status": "active"},
                    {"id": "dup_item_2", "name": "Bản ghi trùng tên", "status": "active"},
                ],
            },
            ensure_ascii=False,
        )
        ask_pool.append(
            {
                "id": f"probe_dup_ambig_{st['sample_id']}_{rng.randint(10, 99)}",
                "sample_id": st["sample_id"],
                "step_index": st.get("step_index", 1),
                "step_category": "ask",
                "prompt_messages": pmsgs,
                "step_calls": [{"name": ASK_MISSING_INFO, "arguments": {"tool_name": tname, "params": [id_k]}}],
                "tools": copy.deepcopy(st.get("tools") or []),
                "sample_context": copy.deepcopy(st.get("sample_context") or {}),
            }
        )

    rng.shuffle(ask_pool)
    for idx, st in enumerate(ask_pool):
        if sum(1 for r in probe if r["probe_category"] == "unseen_genuine_ask") >= 20:
            break
        sid = str(st.get("sample_id") or "")
        sample_obj = next((x for x in held_out_samples if str(x.id) == sid), None)
        if sample_obj is None:
            continue
        st["tools"] = _expand_tools_to_schemas(st.get("tools") or [], catalog)
        st["prompt_messages"] = _clean_prompt_messages_qwen35(st.get("prompt_messages") or [], sample_obj)
        aliased, acat = _alias_eval(st, sid)
        aliased["id"] = f"probe_ask_{idx}_{sid}"
        rec = _finalize_record(aliased, acat, "probe_unseen_genuine_ask", forbidden_ids, rng)
        if rec:
            rec["probe_category"] = "unseen_genuine_ask"
            probe.append(rec)

    # 5. unseen_oos (20 rows)
    rng.shuffle(oos_source)
    for idx, s in enumerate(oos_source):
        if sum(1 for r in probe if r["probe_category"] == "unseen_oos") >= 20:
            break
        oos_st = _make_irrelevance_oos_step(s, catalog, rng=rng)
        if not oos_st:
            continue
        aliased, acat = _alias_eval(oos_st, s.id)
        aliased["id"] = f"probe_oos_{idx}_{s.id}"
        rec = _finalize_record(aliased, acat, "probe_unseen_oos", forbidden_ids, rng)
        if rec:
            rec["probe_category"] = "unseen_oos"
            probe.append(rec)

    return probe


def build_v2r_datasets(
    *,
    root: str | Path = "data",
    dev_split: str | Path = "configs/splits/dev_v1.json",
    fold0_split: str | Path = "configs/splits/fold_0_ids.json",
    f14_split: str | Path = "configs/splits/folds_1_to_4_ids.json",
    unseen_apps_out: str | Path = "configs/splits/unseen_apps_val.json",
    asr_cache: str | Path = "data/asr/train_only_phowhisper_loop38_nbest.jsonl",
    train_out: str | Path = "data/sft/qwen35_train_v2r_general.jsonl.gz",
    probe_out: str | Path = "data/sft/unseen_schema_eval.jsonl",
    seed: int = 42,
) -> dict[str, Any]:
    rng = random.Random(seed)
    catalog = load_catalog("train", root)
    all_samples = load_split("train", root)
    by_id = {str(s.id): s for s in all_samples}

    dev_ids = _load_id_set(Path(dev_split))
    fold0_ids = _load_id_set(Path(fold0_split))
    f14_ids = _load_id_set(Path(f14_split))

    held_set = set(HELD_OUT_UNSEEN_APPS)
    held_ids = sorted(
        s.id
        for s in all_samples
        if s.id in f14_ids
        and s.label
        and any(c.name.split(".", 1)[0] in held_set for c in (s.label.tool_calls or []) if c.name != ASK_MISSING_INFO)
    )
    unseen_val_path = Path(unseen_apps_out)
    unseen_val_path.parent.mkdir(parents=True, exist_ok=True)
    unseen_val_path.write_text(
        json.dumps({"held_out_apps": list(HELD_OUT_UNSEEN_APPS), "ids": held_ids}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    held_id_set = set(held_ids)
    exclude_ids = dev_ids | fold0_ids | held_id_set
    clean_train = [s for s in all_samples if str(s.id) in f14_ids and str(s.id) not in exclude_ids]
    held_samples = [by_id[sid] for sid in held_ids if sid in by_id]
    asr_map = _load_asr_map(Path(asr_cache))

    # Build 100-sample deterministic Unseen-Schema Probe (excluded from training IDs!)
    probe_forbidden = dev_ids | {str(s.id) for s in clean_train}
    probe_rows = build_unseen_schema_probe(
        held_samples,
        catalog,
        asr_map,
        forbidden_ids=probe_forbidden,
        seed=seed + 1985,
    )
    Path(probe_out).parent.mkdir(parents=True, exist_ok=True)
    with open(probe_out, "w", encoding="utf-8") as f:
        for row in probe_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Gather unrolled clean + real-PhoWhisper steps from the 895 clean_train samples
    clean_steps: list[dict[str, Any]] = []
    asr_steps: list[dict[str, Any]] = []
    for s in clean_train:
        assert_not_excluded(s.id, exclude_ids)
        clean_steps.extend(_unroll_sample_steps(s, catalog))
        arow = asr_map.get(s.id)
        if arow and arow.get("audio_text"):
            asr_steps.extend(
                _unroll_sample_steps(
                    s,
                    catalog,
                    current_text=arow["audio_text"],
                    history_texts=arow.get("history_texts") or None,
                )
            )

    records: list[dict[str, Any]] = []
    max_apps = 5

    def _add(rec: dict[str, Any] | None) -> bool:
        if rec is None:
            return False
        assert_not_excluded(rec["sample_id"], exclude_ids)
        records.append(rec)
        return True

    def _fin(st_obj: dict[str, Any], cat_obj: dict[str, dict[str, Any]], src_tag: str) -> dict[str, Any] | None:
        return _finalize_record(st_obj, cat_obj, src_tag, exclude_ids, rng, max_prompt_apps=max_apps)

    # 1. Seen anchors (~500 rows: ~50% real PhoWhisper ASR, capped ask proportion)
    anchor_pool: list[tuple[str, dict[str, Any]]] = [("seen_anchor_clean", st) for st in clean_steps if st.get("step_category") != "ask"]
    anchor_pool.extend(("seen_anchor_asr", st) for st in asr_steps if st.get("step_category") != "ask")
    rng.shuffle(anchor_pool)
    for src, st in anchor_pool[:500]:
        _add(_fin(st, catalog, src))

    # 2. Realistic aliased-domain unseen schemas (~480 rows: act + final_stop, clean + real ASR)
    act_or_stop = [
        ("alias_domain_clean", st) for st in clean_steps if st.get("step_category") in ("act", "final_stop")
    ] + [
        ("alias_domain_asr", st) for st in asr_steps if st.get("step_category") in ("act", "final_stop")
    ]
    rng.shuffle(act_or_stop)
    alias_count = 0
    for idx, (src, st) in enumerate(act_or_stop):
        if alias_count >= 480:
            break
        aliased, acat = alias_domain_step(st, catalog, rng=rng, domain_pool=TRAIN_ALIAS_DOMAINS, alias_params_prob=0.75)
        aliased["id"] = f"{src}_{st['sample_id']}_{st['step_index']}_{idx}"
        if _add(_fin(aliased, acat, src)):
            alias_count += 1

    # 3. Hammer schema-masked counterfactuals (~260 rows; prune apps BEFORE Hammer strips dots)
    rng.shuffle(act_or_stop)
    hammer_count = 0
    for idx, (_, st) in enumerate(act_or_stop):
        if hammer_count >= 260:
            break
        st_pruned, pruned_tools = _prune_step_distractor_apps(
            st, _expand_tools_to_schemas(st.get("tools") or [], catalog), rng=rng, max_apps=max_apps
        )
        st_pruned["tools"] = pruned_tools
        res = _apply_hammer_with_catalog(st_pruned, catalog, rng=rng)
        if not res:
            continue
        mstep, mcat = res
        mstep["id"] = f"hammer_{st['sample_id']}_{st['step_index']}_{idx}"
        if _add(_fin(mstep, mcat, "hammer_masked")):
            hammer_count += 1

    # 4a. Scalar Anti-False-Ask (~110 rows: half seen, half aliased unseen)
    scalar_steps = [
        st for st in (clean_steps + asr_steps)
        if st.get("step_category") == "act"
        and any(c.get("name") in _SCALAR_CREATE_TOOLS for c in (st.get("step_calls") or []))
    ]
    rng.shuffle(scalar_steps)
    scalar_cnt = 0
    for idx, st in enumerate(scalar_steps):
        if scalar_cnt >= 110:
            break
        if idx % 2 == 1:
            astep, acat = alias_domain_step(st, catalog, rng=rng, domain_pool=TRAIN_ALIAS_DOMAINS, alias_params_prob=0.8)
            astep["id"] = f"anti_ask_scalar_alias_{st['sample_id']}_{idx}"
            if _add(_fin(astep, acat, "anti_ask_scalar_alias")):
                scalar_cnt += 1
        else:
            cstep = copy.deepcopy(st)
            cstep["id"] = f"anti_ask_scalar_{st['sample_id']}_{idx}"
            if _add(_fin(cstep, catalog, "anti_ask_scalar")):
                scalar_cnt += 1

    # 4b. Post-Lookup Disambiguation Anti-False-Ask (~90 rows)
    post_act_steps = [st for st in (clean_steps + asr_steps) if st.get("step_category") == "act" and st.get("step_index", 0) > 0]
    rng.shuffle(post_act_steps)
    dis_cnt = 0
    for idx, st in enumerate(post_act_steps):
        if dis_cnt >= 90:
            break
        dstep = _make_disambiguation_anti_false_ask(st, catalog, rng=rng)
        if not dstep:
            continue
        if idx % 2 == 1:
            astep, acat = alias_domain_step(dstep, catalog, rng=rng, domain_pool=TRAIN_ALIAS_DOMAINS, alias_params_prob=0.8)
            if _add(_fin(astep, acat, "anti_ask_disambig_alias")):
                dis_cnt += 1
        else:
            if _add(_fin(dstep, catalog, "anti_ask_disambig")):
                dis_cnt += 1

    # 4c. Filtered Empty-Search `{}` Recovery (~80 rows)
    search_cands = [st for st in clean_steps if st.get("step_category") == "act"]
    rng.shuffle(search_cands)
    empty_cnt = 0
    for idx, st in enumerate(search_cands):
        if empty_cnt >= 80:
            break
        rec_st = _make_generic_empty_search_recovery(st, catalog, rng=rng)
        if not rec_st:
            continue
        if idx % 2 == 1 and (rec_st.get("step_calls") or [{}])[0].get("name") != "contacts.search":
            astep, acat = alias_domain_step(rec_st, catalog, rng=rng, domain_pool=TRAIN_ALIAS_DOMAINS, alias_params_prob=0.75)
            if _add(_fin(astep, acat, "recovery_empty_search_alias")):
                empty_cnt += 1
        else:
            if _add(_fin(rec_st, catalog, "recovery_empty_search")):
                empty_cnt += 1

    # 4d. X-Ref Repair Recovery (~60 rows)
    xref_cnt = 0
    for idx, st in enumerate(search_cands):
        if xref_cnt >= 60:
            break
        xstep = create_xref_repair_recovery_step(st, require_grounded_context=True, catalog=catalog)
        if not xstep:
            continue
        if idx % 2 == 1:
            astep, acat = alias_domain_step(xstep, catalog, rng=rng, domain_pool=TRAIN_ALIAS_DOMAINS, alias_params_prob=0.75)
            if _add(_fin(astep, acat, "recovery_xref_alias")):
                xref_cnt += 1
        else:
            if _add(_fin(xstep, catalog, "recovery_xref")):
                xref_cnt += 1

    # 5. Genuine Ask & When2Call (~165 rows max so ask stays ~8% of mix)
    real_asks = [st for st in (clean_steps + asr_steps) if st.get("step_category") == "ask"]
    rng.shuffle(real_asks)
    ask_cnt = 0
    for idx, st in enumerate(real_asks[:75]):
        if idx % 3 == 0:
            astep, acat = alias_domain_step(st, catalog, rng=rng, domain_pool=TRAIN_ALIAS_DOMAINS, alias_params_prob=0.75)
            if _add(_fin(astep, acat, "genuine_ask_alias")):
                ask_cnt += 1
        else:
            if _add(_fin(st, catalog, "genuine_ask_real")):
                ask_cnt += 1

    w2c_rows = generate_when2call_contrastive_samples(clean_train, catalog, count=75, seed=seed + 21)
    post_ask_rows = generate_post_lookup_ask_samples(clean_train, catalog, count=70, seed=seed + 22)
    combined_w2c = list(w2c_rows) + list(post_ask_rows)
    rng.shuffle(combined_w2c)
    for idx, st in enumerate(combined_w2c):
        if ask_cnt >= 165:
            break
        sid = str(st.get("sample_id") or "")
        sobj = by_id.get(sid)
        if not sobj:
            continue
        st["tools"] = _expand_tools_to_schemas(st.get("tools") or [], catalog)
        st["prompt_messages"] = _clean_prompt_messages_qwen35(st.get("prompt_messages") or [], sobj)
        if idx % 2 == 1:
            astep, acat = alias_domain_step(st, catalog, rng=rng, domain_pool=TRAIN_ALIAS_DOMAINS, alias_params_prob=0.75)
            if _add(_fin(astep, acat, "when2call_alias")):
                ask_cnt += 1
        else:
            if _add(_fin(st, catalog, "when2call_seen")):
                ask_cnt += 1

    # 6. OOS + Conditional + ITN Email / AURA ASR (~175 rows)
    oos_pool = list(clean_train)
    rng.shuffle(oos_pool)
    oos_cnt = 0
    for idx, s in enumerate(oos_pool):
        if oos_cnt >= 100:
            break
        ost = _make_irrelevance_oos_step(s, catalog, rng=rng)
        if not ost:
            continue
        if idx % 2 == 1:
            astep, acat = alias_domain_step(ost, catalog, rng=rng, domain_pool=TRAIN_ALIAS_DOMAINS, alias_params_prob=0.75)
            if _add(_fin(astep, acat, "oos_alias_domain")):
                oos_cnt += 1
        else:
            if _add(_fin(ost, catalog, "oos_seen")):
                oos_cnt += 1

    itn_pool = list(clean_train)
    rng.shuffle(itn_pool)
    itn_cnt = 0
    for idx, s in enumerate(itn_pool):
        if itn_cnt >= 75:
            break
        ist = _make_itn_email_variant(s, catalog, rng=rng, asr_row=asr_map.get(s.id))
        if not ist:
            continue
        if idx % 3 == 0:
            astep, acat = alias_domain_step(ist, catalog, rng=rng, domain_pool=TRAIN_ALIAS_DOMAINS, alias_params_prob=0.75)
            if _add(_fin(astep, acat, "asr_itn_aura_alias")):
                itn_cnt += 1
        else:
            if _add(_fin(ist, catalog, "asr_itn_aura")):
                itn_cnt += 1

    # 7. Long Tool Result Grounding (~100 rows testing up to 3,300 char observations)
    long_cnt = 0
    rng.shuffle(post_act_steps)
    for idx, st in enumerate(post_act_steps):
        if long_cnt >= 100:
            break
        lstep = _make_disambiguation_anti_false_ask(st, catalog, rng=rng, long_mode=True)
        if not lstep:
            continue
        if idx % 2 == 1:
            astep, acat = alias_domain_step(lstep, catalog, rng=rng, domain_pool=TRAIN_ALIAS_DOMAINS, alias_params_prob=0.75)
            if _add(_fin(astep, acat, "long_obs_alias")):
                long_cnt += 1
        else:
            if _add(_fin(lstep, catalog, "long_obs_seen")):
                long_cnt += 1

    rng.shuffle(records)
    out_p = Path(train_out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_p, "wt", encoding="utf-8") as gz:
        for row in records:
            gz.write(json.dumps(row, ensure_ascii=False) + "\n")

    train_sids = {r["sample_id"] for r in records}
    probe_sids = {r["sample_id"] for r in probe_rows}
    leak_dev = len(train_sids & dev_ids)
    leak_f0 = len(train_sids & fold0_ids)
    leak_held = len(train_sids & held_id_set)
    leak_probe_dev = len(probe_sids & dev_ids)
    assert leak_dev == 0 and leak_f0 == 0 and leak_held == 0 and leak_probe_dev == 0

    stats = {
        "total_train_rows": len(records),
        "unique_train_source_samples": len(train_sids),
        "held_out_unseen_apps": list(HELD_OUT_UNSEEN_APPS),
        "held_out_sample_count": len(held_ids),
        "probe_rows": len(probe_rows),
        "probe_categories": dict(Counter(r["probe_category"] for r in probe_rows)),
        "source_counts": dict(Counter(r["source"] for r in records)),
        "category_counts": dict(Counter(r["step_category"] for r in records)),
        "leaks": {
            "train_vs_dev299": leak_dev,
            "train_vs_fold0": leak_f0,
            "train_vs_unseen_apps_val": leak_held,
            "probe_vs_dev299": leak_probe_dev,
        },
    }
    stats_path = out_p.with_name("qwen35_train_v2r_general_stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data")
    parser.add_argument("--train-out", default="data/sft/qwen35_train_v2r_general.jsonl.gz")
    parser.add_argument("--probe-out", default="data/sft/unseen_schema_eval.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    stats = build_v2r_datasets(
        root=args.root,
        train_out=args.train_out,
        probe_out=args.probe_out,
        seed=args.seed,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
