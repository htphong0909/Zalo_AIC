"""Unit tests for Tier-1 Stage-2 On-Policy DPO V2 builder & per-step health guards."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from scripts.build_minicpm5_dpo_pairs_v2 import (
    build_dpo_v2_dataset,
    generate_on_policy_trace_pairs,
    render_prompt_chatml,
)
from scripts.train_minicpm5_dpo_v2 import evaluate_step_health
from src.core.dataset import load_catalog, load_split


def test_render_prompt_chatml_wraps_tool_responses():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": '<think>\nx\n</think>\n\n<function name="a"></function>'},
        {"role": "tool", "name": "a", "content": '{"ok": true}'},
        {"role": "tool", "name": "b", "content": '{"ok": false}'},
    ]
    out = render_prompt_chatml(msgs)
    assert "<|im_start|>tool" not in out
    assert out.count("<|im_start|>user") == 2
    assert "<tool_response>\n{\"ok\": true}\n</tool_response>" in out
    assert out.endswith("<|im_start|>assistant\n")


def test_build_dpo_v2_dataset_zero_dev_leak_and_no_strawman(tmp_path: Path):
    dev_raw = json.loads(Path("configs/splits/dev_v1.json").read_text(encoding="utf-8"))
    dev_ids = set(dev_raw.get("dev_ids") or dev_raw.get("ids") or [])

    train_gz, val_gz, stats = build_dpo_v2_dataset(
        root="data",
        split="train",
        dev_split_path="configs/splits/dev_v1.json",
        out_dir=tmp_path,
        val_ratio=0.10,
        seed=42,
    )
    assert train_gz.exists() and val_gz is not None and val_gz.exists()
    assert stats["max_est_tokens"] <= 15800

    train_sids = set()
    val_sids = set()
    for p, bucket in [(train_gz, train_sids), (val_gz, val_sids)]:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                bucket.add(r["sample_id"])
                assert r["sample_id"] not in dev_ids
                assert r["chosen"] != r["rejected"]
                assert "Quyết định sai lệch" not in r["rejected"]
                assert "<think>" in r["chosen"] and "</think>" in r["chosen"]
                assert "<think>" in r["rejected"] and "</think>" in r["rejected"]
                assert "Quy tắc bắt buộc" in r["prompt"]
                assert r["prompt"].endswith("<|im_start|>assistant\n")
    assert not (train_sids & val_sids)


def test_on_policy_trace_pair_extraction(tmp_path: Path):
    samples = load_split("train", root="data")
    catalog = load_catalog("train", root="data")
    target = next(s for s in samples if s.label and len(s.label.tool_calls) >= 2)

    from src.agent.sft_dataset import build_sft_trajectory, unroll_trajectory_steps

    traj = build_sft_trajectory(target, catalog)
    steps = unroll_trajectory_steps(traj, split="train", root="data")
    assert len(steps) >= 2
    step0_calls = steps[0]["step_calls"]

    traces_file = tmp_path / "train_traces.jsonl"
    per_sample_file = tmp_path / "per_sample.jsonl"
    trace_row = {
        "id": target.id,
        "trace": [
            {"step": 1, "event": "executed", "calls": step0_calls, "results": []},
            {"step": 2, "event": "final_text", "final_text": "Xong bước 1 rồi ạ."},
        ],
    }
    traces_file.write_text(json.dumps(trace_row, ensure_ascii=False) + "\n", encoding="utf-8")
    per_sample_file.write_text(json.dumps({"id": target.id, "score": 0.0}) + "\n", encoding="utf-8")

    import random
    pairs = generate_on_policy_trace_pairs(
        {target.id: target},
        catalog,
        traces_file,
        per_sample_file,
        root="data",
        split="train",
        rng=random.Random(7),
        max_tokens=15800,
    )
    assert len(pairs) == 1
    assert pairs[0]["category"] == "on_policy_early_stop"
    assert "Xong bước 1 rồi ạ." in pairs[0]["rejected"]


def test_evaluate_step_health_guards():
    s, b, d = evaluate_step_health({"loss": 0.3, "grad_norm": 0.4, "logps/chosen": -10.0}, baseline_chosen_logp=None, max_chosen_logp_drop=3.0)
    assert s == "OK" and b == -10.0 and d == 0.0

    s2, b2, d2 = evaluate_step_health({"loss": 0.2, "grad_norm": 0.3, "logps/chosen": -13.5}, baseline_chosen_logp=b, max_chosen_logp_drop=3.0)
    assert s2 == "COLLAPSE" and d2 == -3.5

    s3, _, _ = evaluate_step_health({"loss": float("nan"), "grad_norm": 0.1, "logps/chosen": -10.0}, baseline_chosen_logp=b, max_chosen_logp_drop=3.0)
    assert s3 == "NAN"


def test_dpo_v2_stepdpo_prefix_parity_and_upgraded_categories():
    """Verify generated train/val pairs enforce Step-DPO 4-slot parity, zero leak, and upgraded categories."""
    import re

    train_gz = Path("data/dpo/minicpm5_dpo_v2_train.jsonl.gz")
    val_gz = Path("data/dpo/minicpm5_dpo_v2_val.jsonl.gz")
    assert train_gz.exists() and val_gz.exists()

    dev_raw = json.loads(Path("configs/splits/dev_v1.json").read_text(encoding="utf-8"))
    dev_ids = set(dev_raw["dev_ids"])

    tr_rows = [json.loads(line) for line in gzip.open(train_gz, "rt", encoding="utf-8")]
    va_rows = [json.loads(line) for line in gzip.open(val_gz, "rt", encoding="utf-8")]
    tr_sids = {r["sample_id"] for r in tr_rows}
    va_sids = {r["sample_id"] for r in va_rows}
    assert not (tr_sids & dev_ids)
    assert not (va_sids & dev_ids)
    assert not (tr_sids & va_sids)

    cats = {r["category"] for r in tr_rows}
    expected_cats = {
        "horizon_stop_vs_overcall",
        "anti_early_stop",
        "post_lookup_ask_vs_guess",
        "post_lookup_clear_act_vs_false_ask",
        "when2call_step0_ask",
        "when2call_step0_act",
        "oos_refusal",
        "from_account_balance_rule",
        "bill_source_choice",
        "obs_sibling_entity_id",
        "subtool_confusion",
        "missing_explicit_key",
        "datetime_shift",
        "fuzzy_concise_vs_verbose",
    }
    assert expected_cats.issubset(cats)

    for r in tr_rows + va_rows:
        assert r["est_prompt_tokens"] <= 14000
        assert r["est_tokens"] <= 15800
        rej = r["rejected"]
        chs = r["chosen"]
        assert "1. Observation:" not in rej
        assert "2. Intent & State:" not in rej
        assert "3. Action Plan:" not in rej
        assert "Quyết định sai lệch" not in rej
        c1 = re.search(r"^1\.\s*Ý định:\s*(.+)$", chs, re.MULTILINE)
        c2 = re.search(r"^2\.\s*Dữ liệu:\s*(.+)$", chs, re.MULTILINE)
        r1 = re.search(r"^1\.\s*Ý định:\s*(.+)$", rej, re.MULTILINE)
        r2 = re.search(r"^2\.\s*Dữ liệu:\s*(.+)$", rej, re.MULTILINE)
        assert c1 and c2 and r1 and r2
        assert c1.group(1) == r1.group(1)
        assert c2.group(1) == r2.group(1)
