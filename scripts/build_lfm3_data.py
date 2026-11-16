"""Generate step-wise unrolled LFM3 SFT, DPO, and GRPO datasets with PhoWhisper alignment, Hammer masking, and When2Call pairs."""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import string
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.sft_dataset import build_sft_trajectory, unroll_trajectory_steps
from src.core.dataset import load_catalog, load_split
from src.core.schemas import ASK_MISSING_INFO
from src.env.safe_env import SafeEnv

_DIGIT_WORDS = {
    0: "không",
    1: "một",
    2: "hai",
    3: "ba",
    4: "bốn",
    5: "năm",
    6: "sáu",
    7: "bảy",
    8: "tám",
    9: "chín",
}


def _int_to_vi_words(n: int) -> str:
    if n < 0:
        return str(n)
    if n < 10:
        return _DIGIT_WORDS[n]
    if n == 10:
        return "mười"
    if n < 20:
        unit = n % 10
        if unit == 5:
            return "mười lăm"
        return f"mười {_DIGIT_WORDS[unit]}"
    if n < 100:
        tens, unit = divmod(n, 10)
        tens_w = f"{_DIGIT_WORDS[tens]} mươi"
        if unit == 0:
            return tens_w
        if unit == 1:
            return f"{tens_w} mốt"
        if unit == 4:
            return f"{tens_w} tư"
        if unit == 5:
            return f"{tens_w} lăm"
        return f"{tens_w} {_DIGIT_WORDS[unit]}"
    if n < 1000:
        hundreds, rem = divmod(n, 100)
        h_w = f"{_DIGIT_WORDS[hundreds]} trăm"
        if rem == 0:
            return h_w
        if rem < 10:
            return f"{h_w} lẻ {_DIGIT_WORDS[rem]}"
        return f"{h_w} {_int_to_vi_words(rem)}"
    return str(n)


def simulate_ctc_transcript(text: str) -> str:
    """Convert written Vietnamese utterance (with digits/punctuation) into ChunkFormer-CTC spoken style."""
    s = (text or "").strip().lower()
    if not s:
        return ""

    s = re.sub(r"\bsms\b", "a ét mờ ét", s)
    s = re.sub(r"\bsjc\b", "ét giê xê", s)
    s = re.sub(r"\bdời\b", "rời", s)

    def _repl_dec_unit(m: re.Match[str]) -> str:
        whole = int(m.group(1))
        frac = int(m.group(2))
        unit = m.group(3)
        frac_w = "rưỡi" if frac == 5 else _int_to_vi_words(frac)
        return f"{_int_to_vi_words(whole)} {unit} {frac_w}"

    s = re.sub(r"\b(\d+)[.,](\d)\s*(triệu|nghìn|ngàn|tỷ|tiếng)\b", _repl_dec_unit, s)

    def _repl_hm(m: re.Match[str]) -> str:
        hh = int(m.group(1))
        mm = int(m.group(2))
        if mm == 30:
            return f"{_int_to_vi_words(hh)} rưỡi"
        if mm == 0:
            return f"{_int_to_vi_words(hh)} giờ"
        return f"{_int_to_vi_words(hh)} giờ {_int_to_vi_words(mm)}"

    s = re.sub(r"\b(\d{1,2})\s*[h:]\s*(\d{1,2})\b", _repl_hm, s)
    s = re.sub(r"\b(\d{1,2})\s*h\b", lambda m: f"{_int_to_vi_words(int(m.group(1)))} giờ", s)
    s = re.sub(r"\b(\d{1,3})\b", lambda m: _int_to_vi_words(int(m.group(1))), s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def format_lfm3_completion_xml(completion_msg: dict[str, Any]) -> str:
    """Format an assistant completion message into LFM3's XML tool-call or text format."""
    tcs = completion_msg.get("tool_calls") or []
    if not tcs:
        return str(completion_msg.get("content") or "").strip()

    blocks: list[str] = []
    for tc in tcs:
        fn = tc.get("function") or {}
        name = fn.get("name", "")
        args = fn.get("arguments") or {}
        param_lines: list[str] = []
        for k, v in args.items():
            val_str = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, bool)) else str(v)
            param_lines.append(f"<parameter={k}>\n{val_str}\n</parameter>")
        body = "\n".join(param_lines)
        if body:
            blocks.append(f"<function={name}>\n{body}\n</function>")
        else:
            blocks.append(f"<function={name}>\n</function>")
    inner = "\n".join(blocks)
    return f"<tool_call>\n{inner}\n</tool_call>"


def _fallback_render_prompt(prompt_messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    tools_injected = False
    tools_block = (
        f"<|tool_list_start|>{json.dumps(tools, ensure_ascii=False)}<|tool_list_end|>"
        if tools
        else ""
    )
    for m in prompt_messages:
        role = m.get("role", "user")
        if role == "assistant" and m.get("tool_calls"):
            content = format_lfm3_completion_xml(m)
        else:
            content = str(m.get("content") or "")
        if role == "system" and tools_block and not tools_injected:
            content = f"{content}\n{tools_block}" if content else tools_block
            tools_injected = True
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    if tools_block and not tools_injected:
        parts.insert(0, f"<|im_start|>system\n{tools_block}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def render_step_prompt_completion(
    step: dict[str, Any],
    tokenizer: Any | None = None,
) -> tuple[str, str]:
    """Render (prompt_str, completion_str) using LFM3's native chat template without double BOS/EOS."""
    prompt_msgs = step["prompt_messages"]
    comp_msg = step["completion_message"]
    tools = step.get("tools") or []

    if tokenizer is not None:
        prompt_str = tokenizer.apply_chat_template(
            prompt_msgs,
            tools=tools or None,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        full_str = tokenizer.apply_chat_template(
            [*prompt_msgs, comp_msg],
            tools=tools or None,
            add_generation_prompt=False,
            tokenize=False,
            enable_thinking=False,
        )
        if full_str.startswith(prompt_str):
            comp_str = full_str[len(prompt_str) :]
        else:
            comp_str = format_lfm3_completion_xml(comp_msg)
    else:
        prompt_str = _fallback_render_prompt(prompt_msgs, tools)
        comp_str = format_lfm3_completion_xml(comp_msg)

    # Strip explicit leading BOS tag so TRL/tokenizer doesn't duplicate BOS
    if prompt_str.startswith("<|startoftext|>"):
        prompt_str = prompt_str[len("<|startoftext|>") :]

    # Strip explicit trailing EOS tag so TRL/tokenizer doesn't duplicate EOS
    comp_str = re.sub(r"<\|im_end\|>\s*$", "", comp_str).strip()

    return prompt_str, comp_str


def build_sft_step_record(
    step: dict[str, Any],
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Build a prompt-completion and conversational record for TRL SFTTrainer."""
    prompt_str, comp_str = render_step_prompt_completion(step, tokenizer=tokenizer)
    return {
        "id": step["id"],
        "step_index": step["step_index"],
        "step_category": step["step_category"],
        "prompt": prompt_str,
        "completion": comp_str,
        "messages": [*step["prompt_messages"], step["completion_message"]],
        "tools": step.get("tools") or [],
    }


def build_grpo_step_record(
    step: dict[str, Any],
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Build a prompt + verifiable metadata record for TRL GRPOTrainer."""
    prompt_str, _ = render_step_prompt_completion(step, tokenizer=tokenizer)
    step_hash = step.get("step_gold_state_hash") or step.get("gold_state_hash")
    return {
        "sample_id": step["id"],
        "step_index": step["step_index"],
        "step_category": step["step_category"],
        "is_terminal": bool(step.get("is_terminal", False)),
        "gold_state_hash": step_hash,
        "step_gold_state_hash": step_hash,
        "prompt": prompt_str,
        "calls": json.dumps(step["step_calls"], ensure_ascii=False),
        "prefix_calls": json.dumps(step["prefix_calls"], ensure_ascii=False),
        "tools": json.dumps(step.get("tools") or [], ensure_ascii=False),
    }


def _random_ident(prefix: str, rng: random.Random, length: int = 5) -> str:
    alphabet = string.ascii_lowercase + string.digits
    suffix = "".join(rng.choice(alphabet) for _ in range(length))
    return f"{prefix}_{suffix}"


def apply_hammer_masking(
    step: dict[str, Any],
    *,
    rng: random.Random,
) -> dict[str, Any]:
    """Hammer (ICLR 2025) function & parameter name masking to force schema-grounded reasoning."""
    masked = copy.deepcopy(step)
    fn_map: dict[str, str] = {}
    param_map: dict[str, dict[str, str]] = {}

    new_tools: list[dict[str, Any]] = []
    for t in masked.get("tools") or []:
        fn = t.get("function") or {}
        orig_name = fn.get("name", "")
        if not orig_name or orig_name == ASK_MISSING_INFO:
            new_tools.append(t)
            continue
        alias_fn = _random_ident("fn", rng)
        fn_map[orig_name] = alias_fn
        pmap: dict[str, str] = {}
        params = fn.get("parameters") or {}
        props = params.get("properties") or {}
        new_props: dict[str, Any] = {}
        for pk, pspec in props.items():
            alias_p = _random_ident("arg", rng, length=4)
            pmap[pk] = alias_p
            new_props[alias_p] = pspec
        param_map[orig_name] = pmap

        fn["name"] = alias_fn
        params["properties"] = new_props
        if "required" in params and isinstance(params["required"], list):
            params["required"] = [pmap.get(r, r) for r in params["required"]]
        new_tools.append(t)

    # Remap any x-ref or description referencing masked tool names inside properties
    for t in new_tools:
        props = ((t.get("function") or {}).get("parameters") or {}).get("properties") or {}
        for pk, pspec in props.items():
            if isinstance(pspec, dict):
                for orig, alias in fn_map.items():
                    if "x-ref" in pspec and orig in pspec["x-ref"]:
                        pspec["x-ref"] = pspec["x-ref"].replace(orig, alias)
                    if "description" in pspec and orig in pspec["description"]:
                        pspec["description"] = pspec["description"].replace(orig, alias)

    def _remap_call_dict(name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if name == ASK_MISSING_INFO:
            out_args = dict(args)
            tname = str(out_args.get("tool_name") or "")
            if tname in fn_map:
                pm = param_map.get(tname, {})
                out_args["tool_name"] = fn_map[tname]
                if isinstance(out_args.get("params"), list):
                    out_args["params"] = [pm.get(str(x), str(x)) for x in out_args["params"]]
            return name, out_args
        new_name = fn_map.get(name, name)
        pm = param_map.get(name, {})
        new_args = {pm.get(k, k): v for k, v in (args or {}).items()}
        return new_name, new_args

    def _remap_msg(msg: dict[str, Any]) -> None:
        if msg.get("role") == "system" and isinstance(msg.get("content"), str):
            for orig, alias in fn_map.items():
                if orig in msg["content"]:
                    msg["content"] = msg["content"].replace(orig, alias)
        if msg.get("role") == "tool" and msg.get("name") in fn_map:
            msg["name"] = fn_map[msg["name"]]
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function")
            if isinstance(fn, dict) and "name" in fn:
                n_name, n_args = _remap_call_dict(fn["name"], fn.get("arguments") or {})
                fn["name"] = n_name
                fn["arguments"] = n_args

    for m in masked.get("prompt_messages") or []:
        _remap_msg(m)
    if "completion_message" in masked:
        _remap_msg(masked["completion_message"])

    masked["step_calls"] = [
        {"name": n_name, "arguments": n_args}
        for c in (masked.get("step_calls") or [])
        for n_name, n_args in [_remap_call_dict(c["name"], c.get("arguments") or {})]
    ]
    masked["prefix_calls"] = [
        {"name": n_name, "arguments": n_args}
        for c in (masked.get("prefix_calls") or [])
        for n_name, n_args in [_remap_call_dict(c["name"], c.get("arguments") or {})]
    ]

    # Shuffle tools to eliminate positional bias (Hammer ICLR 2025)
    rng.shuffle(new_tools)
    masked["tools"] = new_tools
    return masked


def _dummy_value_for_spec(prop_spec: dict[str, Any]) -> Any:
    if "enum" in prop_spec and prop_spec["enum"]:
        return prop_spec["enum"][0]
    ptype = prop_spec.get("type")
    if ptype == "integer":
        return 100000
    if ptype == "number":
        return 1.0
    if ptype == "boolean":
        return True
    if ptype == "array":
        return ["unknown"]
    if ptype == "object":
        return {}
    fmt = prop_spec.get("format")
    if fmt == "date":
        return "2026-11-20"
    if fmt == "time":
        return "08:00"
    if fmt == "date-time":
        return "2026-11-20T08:00:00+07:00"
    return "id_unknown_01"


def build_dpo_pairs_for_step(
    step: dict[str, Any],
    *,
    tokenizer: Any | None = None,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Construct When2Call 3-way contrast preference pairs (act vs ask vs oos/stop) for LFM3 DPO."""
    prompt_str, chosen_str = render_step_prompt_completion(step, tokenizer=tokenizer)
    cat = step["step_category"]
    tools = step.get("tools") or []
    tool_by_name = {
        t["function"]["name"]: t["function"]
        for t in tools
        if isinstance(t.get("function"), dict) and "name" in t["function"]
    }
    pairs: list[dict[str, Any]] = []

    def _render_rejected_msg(rej_msg: dict[str, Any]) -> str:
        fake_step = {
            "prompt_messages": step["prompt_messages"],
            "completion_message": rej_msg,
            "tools": tools,
        }
        _, rej_str = render_step_prompt_completion(fake_step, tokenizer=tokenizer)
        return rej_str

    if cat == "act" and step["step_calls"]:
        # Contrast 1: Gold tool call (chosen) vs False sys.ask_missing_info (rejected) -> cures false_ask
        first_call = step["step_calls"][0]
        arg_keys = list((first_call.get("arguments") or {}).keys())
        if not arg_keys:
            fn_props = ((tool_by_name.get(first_call["name"]) or {}).get("parameters") or {}).get("properties") or {}
            arg_keys = list(fn_props.keys())[:1] or ["id"]
        rej_ask_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": ASK_MISSING_INFO,
                        "arguments": {"tool_name": first_call["name"], "params": arg_keys[:2]},
                    },
                }
            ],
        }
        rej_str = _render_rejected_msg(rej_ask_msg)
        if rej_str != chosen_str:
            pairs.append(
                {
                    "id": step["id"],
                    "step_index": step["step_index"],
                    "step_category": cat,
                    "contrast": "act_vs_false_ask",
                    "prompt": prompt_str,
                    "chosen": chosen_str,
                    "rejected": rej_str,
                }
            )

    elif cat in ("oos", "final_stop"):
        # Contrast 2: Clean stop / OOS refusal (chosen) vs Hallucinated tool call (rejected) -> cures missed_oos & loops
        cand_names = [n for n in tool_by_name if n != ASK_MISSING_INFO]
        halluc_call: dict[str, Any] | None = None
        if cat == "final_stop" and step.get("prefix_calls"):
            last_prev = step["prefix_calls"][-1]
            halluc_call = {"name": last_prev["name"], "arguments": dict(last_prev.get("arguments") or {})}
        elif cand_names:
            chosen_fn = rng.choice(cand_names)
            fn_spec = tool_by_name[chosen_fn]
            props = (fn_spec.get("parameters") or {}).get("properties") or {}
            req = (fn_spec.get("parameters") or {}).get("required") or []
            dummy_args = {rk: _dummy_value_for_spec(props.get(rk) or {}) for rk in req}
            halluc_call = {"name": chosen_fn, "arguments": dummy_args}

        if halluc_call is not None:
            rej_tool_msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": halluc_call}],
            }
            rej_str = _render_rejected_msg(rej_tool_msg)
            if rej_str != chosen_str:
                pairs.append(
                    {
                        "id": step["id"],
                        "step_index": step["step_index"],
                        "step_category": cat,
                        "contrast": f"{cat}_vs_hallucinated_call",
                        "prompt": prompt_str,
                        "chosen": chosen_str,
                        "rejected": rej_str,
                    }
                )

    elif cat == "ask" and step["step_calls"]:
        # Contrast 3: Gold sys.ask_missing_info (chosen) vs Guessed write call with dummy args (rejected) -> cures missed_ask
        ask_call = next((c for c in step["step_calls"] if c["name"] == ASK_MISSING_INFO), step["step_calls"][0])
        target_fn = str((ask_call.get("arguments") or {}).get("tool_name") or "")
        missing_params = list((ask_call.get("arguments") or {}).get("params") or [])
        if target_fn:
            fn_spec = tool_by_name.get(target_fn) or {}
            props = (fn_spec.get("parameters") or {}).get("properties") or {}
            req = list((fn_spec.get("parameters") or {}).get("required") or [])
            fill_keys = list(dict.fromkeys([*req, *[str(p) for p in missing_params]]))
            guessed_args = {k: _dummy_value_for_spec(props.get(k) or {}) for k in fill_keys}
            rej_guess_msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": target_fn, "arguments": guessed_args}}],
            }
            rej_str = _render_rejected_msg(rej_guess_msg)
            if rej_str != chosen_str:
                pairs.append(
                    {
                        "id": step["id"],
                        "step_index": step["step_index"],
                        "step_category": cat,
                        "contrast": "ask_vs_guessed_call",
                        "prompt": prompt_str,
                        "chosen": chosen_str,
                        "rejected": rej_str,
                    }
                )

    return pairs


def _filter_training_tools(
    sample: Any,
    full_tools: list[dict[str, Any]],
    *,
    max_distractor_apps: int | None,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Keep all gold, history, and OOS installed apps; optionally subsample distractors only on non-OOS samples."""
    if max_distractor_apps is None:
        return full_tools

    # Never prune installed_apps on OOS samples (prevents test-time missed_oos under full catalog)
    if not (sample.label and sample.label.tool_calls):
        return full_tools

    req_apps: set[str] = set()
    for c in sample.label.tool_calls:
        req_apps.add(c.name.split(".", 1)[0])
        if c.name == ASK_MISSING_INFO:
            tname = str(c.arguments.get("tool_name") or "")
            if "." in tname:
                req_apps.add(tname.split(".", 1)[0])

    for turn in getattr(sample, "history", []) or []:
        for hc in getattr(turn, "tool_calls", []) or []:
            req_apps.add(hc.name.split(".", 1)[0])

    installed_apps = list((sample.context or {}).get("installed_apps") or {})
    distractor_candidates = [a for a in installed_apps if a not in req_apps]
    if len(distractor_candidates) > max_distractor_apps:
        chosen_distractors = set(rng.sample(distractor_candidates, max_distractor_apps))
    else:
        chosen_distractors = set(distractor_candidates)

    keep_apps = req_apps | chosen_distractors | {"sys"}
    return [
        t
        for t in full_tools
        if t.get("function", {}).get("name", "").split(".", 1)[0] in keep_apps
    ]


def oversample_grpo_records(
    records: list[dict[str, Any]],
    *,
    oversample_categories: tuple[str, ...] = ("ask", "oos"),
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Oversample no-action / clarify steps 2x as in lfm3_grpo_with_trl.ipynb."""
    extra = [dict(r) for r in records if r.get("step_category") in oversample_categories]
    combined = list(records) + extra
    rng = random.Random(seed)
    rng.shuffle(combined)
    return combined


def _load_exclude_ids(path: str | Path | None) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return set(raw)
    return set(raw.get("dev_ids") or raw.get("ids") or [])


def _load_asr_map(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[row["id"]] = row
    return out


def _compute_gold_state_hash(sample: Any, split: str, root: str) -> str | None:
    if not sample.label:
        return None
    try:
        env = SafeEnv(split, sample.id, root)
        for c in sample.label.tool_calls:
            env.call(c.name, c.arguments)
        return env.get_state_hash()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--root", default="data")
    parser.add_argument("--exclude-ids", default="configs/splits/dev_v1.json")
    parser.add_argument(
        "--asr-transcripts",
        default="experiments/exp_20261107_phowhisper_train/transcripts.jsonl",
    )
    parser.add_argument("--asr-mode", choices=("phowhisper", "ctc"), default="phowhisper")
    parser.add_argument("--model-id", default=None, help="Optional HF model ID to pre-render with AutoTokenizer")
    parser.add_argument("--max-distractor-apps", type=int, default=None)
    parser.add_argument("--final-stop-keep-prob", type=float, default=1.0)
    parser.add_argument("--hammer-mask-prob", type=float, default=0.25)
    parser.add_argument("--out-dir", default="data/sft")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = None
    if args.model_id:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, padding_side="left")

    exclude_ids = _load_exclude_ids(args.exclude_ids)
    asr_map = _load_asr_map(args.asr_transcripts)
    catalog = load_catalog(args.split, args.root)
    samples = [s for s in load_split(args.split, args.root) if s.id not in exclude_ids]

    rng = random.Random(args.seed)
    sft_records: list[dict[str, Any]] = []
    dpo_records: list[dict[str, Any]] = []
    grpo_base_records: list[dict[str, Any]] = []
    asr_augmented_samples = 0
    hammer_masked_steps = 0

    for sample in samples:
        g_hash = _compute_gold_state_hash(sample, args.split, args.root)
        clean_traj = build_sft_trajectory(sample, catalog, gold_state_hash=g_hash)
        clean_traj["tools"] = _filter_training_tools(
            sample, clean_traj["tools"], max_distractor_apps=args.max_distractor_apps, rng=rng
        )
        clean_steps = unroll_trajectory_steps(clean_traj, split=args.split, root=args.root)

        asr_entry = asr_map.get(sample.id)
        if asr_entry:
            asr_text = asr_entry.get("audio_text")
            raw_hist = asr_entry.get("history_texts") or {}
            hist_texts = {int(k): v for k, v in raw_hist.items()}
        elif args.asr_mode == "ctc":
            asr_text = simulate_ctc_transcript(sample.audio_text or "")
            hist_texts = {
                i: simulate_ctc_transcript(t.audio_text or t.text or "")
                for i, t in enumerate(sample.history)
                if t.role == "user" and (t.audio_text or t.text)
            }
        else:
            asr_text = sample.audio_text or ""
            hist_texts = None

        has_noisy = bool(asr_text and asr_text.strip() != (sample.audio_text or "").strip())
        if has_noisy:
            asr_augmented_samples += 1
            primary_traj = build_sft_trajectory(
                sample,
                catalog,
                current_text=asr_text,
                history_texts=hist_texts,
                gold_state_hash=g_hash,
            )
            primary_traj["tools"] = clean_traj["tools"]
            primary_steps = unroll_trajectory_steps(primary_traj, split=args.split, root=args.root)
        else:
            primary_steps = clean_steps

        for st in primary_steps:
            cat = st["step_category"]
            if cat == "final_stop" and rng.random() > args.final_stop_keep_prob:
                continue
            sft_records.append(build_sft_step_record(st, tokenizer=tokenizer))
            dpo_records.extend(build_dpo_pairs_for_step(st, tokenizer=tokenizer, rng=rng))
            grpo_base_records.append(build_grpo_step_record(st, tokenizer=tokenizer))

            if cat == "act" and args.hammer_mask_prob > 0 and rng.random() < args.hammer_mask_prob:
                masked_st = apply_hammer_masking(st, rng=rng)
                sft_records.append(build_sft_step_record(masked_st, tokenizer=tokenizer))
                hammer_masked_steps += 1

        # If real ASR differed from written gold, also include clean written steps for robustness
        if has_noisy:
            for st in clean_steps:
                if st["step_category"] != "final_stop":
                    sft_records.append(build_sft_step_record(st, tokenizer=tokenizer))

    rng.shuffle(sft_records)
    rng.shuffle(dpo_records)
    grpo_records = oversample_grpo_records(grpo_base_records, seed=args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sft_path = out_dir / "lfm3_train_sft.jsonl"
    dpo_path = out_dir / "lfm3_train_dpo.jsonl"
    grpo_path = out_dir / "lfm3_train_grpo.jsonl"

    with open(sft_path, "w", encoding="utf-8") as f_sft:
        for r in sft_records:
            f_sft.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(dpo_path, "w", encoding="utf-8") as f_dpo:
        for r in dpo_records:
            f_dpo.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(grpo_path, "w", encoding="utf-8") as f_grpo:
        for r in grpo_records:
            f_grpo.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "train_only_samples": len(samples),
                "excluded_dev_samples": len(exclude_ids),
                "asr_augmented_samples": asr_augmented_samples,
                "hammer_masked_steps": hammer_masked_steps,
                "sft_step_records": len(sft_records),
                "dpo_pair_records": len(dpo_records),
                "grpo_step_records": len(grpo_records),
                "sft_out": str(sft_path),
                "dpo_out": str(dpo_path),
                "grpo_out": str(grpo_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
