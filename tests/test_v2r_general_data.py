"""Unit tests for unseen-domain schema aliasing, V2R data pipeline, zero-leak guards, and probe scorer."""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

from scripts.build_qwen35_v2r_general_data import build_v2r_datasets
from scripts.eval_schema_probe import score_probe_predictions
from src.agent.qwen35_sft import format_qwen35_tool_calls_xml, step_to_qwen35_example
from src.agent.sft_dataset import build_sft_trajectory, unroll_trajectory_steps
from src.core.dataset import load_catalog, load_split
from src.core.schemas import ASK_MISSING_INFO, Sample
from src.data_gen.execution_verifier import verify_step_validity
from src.data_gen.schema_aliasing import (
    EVAL_ALIAS_DOMAINS,
    TRAIN_ALIAS_DOMAINS,
    alias_domain_step,
)


def test_alias_domains_strictly_disjoint_from_train_and_pub():
    train_apps = {t.split(".", 1)[0] for t in load_catalog("train", "data")}
    pub_apps = {t.split(".", 1)[0] for t in load_catalog("public_test", "data")}
    known = train_apps | pub_apps
    train_pool = set(TRAIN_ALIAS_DOMAINS)
    eval_pool = set(EVAL_ALIAS_DOMAINS)
    assert len(train_pool) == len(TRAIN_ALIAS_DOMAINS)
    assert len(eval_pool) == len(EVAL_ALIAS_DOMAINS)
    assert not (train_pool & known)
    assert not (eval_pool & known)
    assert not (train_pool & eval_pool)


def test_alias_domain_step_rewrites_tools_xref_ask_and_installed_apps():
    catalog = load_catalog("train", "data")
    samples = load_split("train", "data")
    # Pick a sample with smart_home.light_set or notes/reminders multi-step
    target_sample = next(
        s for s in samples
        if s.label and any(c.name == "smart_home.light_set" for c in (s.label.tool_calls or []))
    )
    traj = build_sft_trajectory(target_sample, catalog, max_result_chars=3500)
    steps = unroll_trajectory_steps(traj)
    assert steps
    st = copy.deepcopy(steps[-1])
    st["sample_context"] = copy.deepcopy(target_sample.context or {})
    rng = random.Random(7)

    aliased, acat = alias_domain_step(
        st,
        catalog,
        rng=rng,
        domain_pool=EVAL_ALIAS_DOMAINS,
        alias_params_prob=1.0,
    )
    assert verify_step_validity(aliased, acat)
    sys_msg = next(m["content"] for m in aliased["prompt_messages"] if m["role"] == "system")
    assert "smart_home" not in sys_msg.split("- installed_apps:", 1)[1].splitlines()[0]
    assert aliased["alias_app_map"]["smart_home"] in EVAL_ALIAS_DOMAINS

    # Check ASK aliasing preserves sys.ask_missing_info while remapping target tool_name & params
    ask_step = {
        "id": "ask_test",
        "step_category": "ask",
        "prompt_messages": copy.deepcopy(st["prompt_messages"][:2]),
        "step_calls": [{"name": ASK_MISSING_INFO, "arguments": {"tool_name": "smart_home.light_set", "params": ["device_id"]}}],
        "tools": copy.deepcopy(st["tools"]),
        "sample_context": copy.deepcopy(st["sample_context"]),
    }
    aliased_ask, acat_ask = alias_domain_step(
        ask_step,
        catalog,
        rng=random.Random(11),
        domain_pool=EVAL_ALIAS_DOMAINS,
        alias_params_prob=1.0,
    )
    assert verify_step_validity(aliased_ask, acat_ask)
    ask_call = aliased_ask["step_calls"][0]
    assert ask_call["name"] == ASK_MISSING_INFO
    mapped_target = ask_call["arguments"]["tool_name"]
    assert mapped_target in acat_ask
    assert ask_call["arguments"]["params"][0] in (acat_ask[mapped_target]["parameters"]["properties"])


def test_eval_schema_probe_scorer_metrics():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "aqua_farm.submit_action",
                "parameters": {"type": "object", "properties": {"record_ref": {"type": "string"}}, "required": ["record_ref"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": ASK_MISSING_INFO,
                "parameters": {"type": "object", "properties": {"tool_name": {"type": "string"}, "params": {"type": "array"}}},
            },
        },
    ]
    rows = [
        {
            "probe_category": "unseen_anti_false_ask",
            "tools": tools,
            "completion": format_qwen35_tool_calls_xml([{"name": "aqua_farm.submit_action", "arguments": {"record_ref": "r1"}}]) + "<|im_end|>",
        },
        {
            "probe_category": "unseen_genuine_ask",
            "tools": tools,
            "completion": format_qwen35_tool_calls_xml([{"name": ASK_MISSING_INFO, "arguments": {"tool_name": "aqua_farm.submit_action", "params": ["record_ref"]}}]) + "<|im_end|>",
        },
        {
            "probe_category": "unseen_oos",
            "tools": tools,
            "completion": "Yêu cầu chưa được hỗ trợ.<|im_end|>",
        },
    ]
    preds = [
        '<tool_call>\n<function=aqua_farm.submit_action>\n<parameter=record_ref>\nr1\n</parameter>\n</function>\n</tool_call>',
        '<tool_call>\n<function=sys.ask_missing_info>\n<parameter=tool_name>\naqua_farm.submit_action\n</parameter>\n<parameter=params>\n["record_ref"]\n</parameter>\n</function>\n</tool_call>',
        'Xin lỗi, ứng dụng chưa có trên thiết bị.',
    ]
    res = score_probe_predictions(rows, preds)
    assert res["tool_name_em"] == 1.0
    assert res["exact_call_match"] == 1.0
    assert res["ask_f1"] == 1.0
    assert res["anti_false_ask_trigger_rate"] == 0.0


def test_generated_v2r_train_and_probe_artifacts_zero_leak_and_valid():
    import gzip

    train_p = Path("data/sft/qwen35_train_v2r_general.jsonl.gz")
    probe_p = Path("data/sft/unseen_schema_eval.jsonl")
    assert train_p.exists() and probe_p.exists()

    dev_ids = set(json.loads(Path("configs/splits/dev_v1.json").read_text(encoding="utf-8"))["dev_ids"])
    f0_ids = set(json.loads(Path("configs/splits/fold_0_ids.json").read_text(encoding="utf-8"))["ids"])
    held_ids = set(json.loads(Path("configs/splits/unseen_apps_val.json").read_text(encoding="utf-8"))["ids"])

    train_rows = [json.loads(line) for line in gzip.open(train_p, "rt", encoding="utf-8") if line.strip()]
    probe_rows = [json.loads(line) for line in probe_p.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(train_rows) >= 1800
    assert len(probe_rows) == 100

    train_sids = {r["sample_id"] for r in train_rows}
    probe_sids = {r["sample_id"] for r in probe_rows}
    assert not (train_sids & dev_ids)
    assert not (train_sids & f0_ids)
    assert not (train_sids & held_ids)
    assert not (probe_sids & dev_ids)
    assert not (train_sids & probe_sids)

    for r in train_rows:
        assert r["tools"], f"empty tools in train row {r['id']}"
        assert not r.get("enable_thinking", False)
        sys_txt = next(m["content"] for m in r["prompt_messages"] if m["role"] == "system")
        assert "Tra cứu mở rộng & tránh hỏi thừa" in sys_txt

    for r in probe_rows:
        assert r["tools"], f"empty tools in probe row {r['id']}"
        sys_txt = next(m["content"] for m in r["prompt_messages"] if m["role"] == "system")
        assert "Tra cứu mở rộng & tránh hỏi thừa" in sys_txt
        inst_line = next(line for line in sys_txt.splitlines() if line.startswith("- installed_apps:"))
        for train_dom in TRAIN_ALIAS_DOMAINS:
            assert train_dom not in inst_line

