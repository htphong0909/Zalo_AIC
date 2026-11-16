"""TDD test suite for Tier 1 data synthesis, runtime guardrails, and schema verification."""
from __future__ import annotations

import json
import pytest

from src.agent.react_loop import run_react_loop
from src.core.config import GuardrailConfig, load_config
from src.core.schemas import ASK_MISSING_INFO, Label, Sample, ToolCall, Turn
from src.data_gen.anchor_binding_gen import generate_anchor_binding_samples
from src.data_gen.apigen_mt_gen import generate_apigen_multi_intent_samples
from src.data_gen.distill_think import synthesize_deterministic_think
from src.data_gen.execution_verifier import validate_step_with_reason, verify_step_validity
from src.data_gen.when2call_gen import (
    generate_post_lookup_ask_samples,
    generate_when2call_contrastive_samples,
)
from src.env.safe_env import SafeEnv
from tests.test_react_loop import SequenceFakeBackend


@pytest.fixture
def tier1_catalog() -> dict[str, dict]:
    return {
        "contacts.search": {
            "name": "contacts.search",
            "kind": "read",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": [],
            },
        },
        "phone.call": {
            "name": "phone.call",
            "kind": "write",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_id": {"type": "string", "x-ref": "contacts.contacts"},
                    "phone_number": {"type": "string"},
                },
                "required": [],
                "x-one-of": [["contact_id"], ["phone_number"]],
            },
        },
        "sms.send": {
            "name": "sms.send",
            "kind": "write",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_id": {"type": "string", "x-ref": "contacts.contacts"},
                    "phone_number": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["content"],
                "x-one-of": [["contact_id"], ["phone_number"]],
            },
        },
        "alarm.create": {
            "name": "alarm.create",
            "kind": "write",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {"type": "string"},
                    "label": {"type": "string"},
                    "repeat": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["time"],
            },
        },
        "bill_payment.list_bills": {
            "name": "bill_payment.list_bills",
            "kind": "read",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        "bill_payment.pay": {
            "name": "bill_payment.pay",
            "kind": "write",
            "parameters": {
                "type": "object",
                "properties": {
                    "bill_id": {"type": "string", "x-ref": "bill_payment.bills"},
                    "source": {"type": "string", "enum": ["bank", "ewallet"]},
                },
                "required": ["bill_id", "source"],
            },
        },
        "ewallet.transfer": {
            "name": "ewallet.transfer",
            "kind": "write",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_id": {"type": "string"},
                    "phone": {"type": "string"},
                    "amount": {"type": "integer"},
                },
                "required": ["amount"],
                "x-one-of": [["contact_id"], ["phone"]],
            },
        },
        ASK_MISSING_INFO: {
            "name": ASK_MISSING_INFO,
            "kind": "sys",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string"},
                    "params": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["tool_name", "params"],
            },
        },
    }


def _rich_sample_with_history() -> Sample:
    return Sample(
        id="train_tier1_01",
        audio_text="Nhắn cho Lan giúp anh 5 giờ chiều họp nhé",
        context={
            "current_time": "2026-11-28T07:10:00+07:00",
            "timezone": "Asia/Ho_Chi_Minh",
            "location": {"lat": 16.0544, "lng": 108.2022},
            "user_profile": {"name": "Minh"},
            "installed_apps": {
                "contacts": {"tools": ["contacts.search"]},
                "sms": {"tools": ["sms.send"]},
                "phone": {"tools": ["phone.call"]},
                "alarm": {"tools": ["alarm.create"]},
                "bill_payment": {"tools": ["bill_payment.list_bills", "bill_payment.pay"]},
                "ewallet": {"tools": ["ewallet.transfer"]},
            },
        },
        history=[
            Turn(role="user", audio_text="Hôm nay mấy giờ có lịch?"),
            Turn(role="assistant", text="Sáng nay anh có lịch 9 giờ."),
        ],
        label=Label(
            tool_calls=[
                ToolCall(name="alarm.create", arguments={"time": "06:30"}),
                ToolCall(name="contacts.search", arguments={"query": "Lan"}),
                ToolCall(name="sms.send", arguments={"contact_id": "c_01", "content": "5 giờ chiều họp nhé"}),
            ],
            tool_results=[
                {"ok": True, "id": "al_1"},
                {"ok": True, "contacts": [{"id": "c_01", "name": "Lan Công Ty"}]},
                {"ok": True},
            ],
        ),
    )


def test_when2call_preserves_system_prompt_history_and_full_apps(tier1_catalog):
    s = _rich_sample_with_history()
    s_amount = Sample(
        id="train_tier1_02",
        audio_text="Chuyển 200000 qua ví cho số 0912345678",
        context=s.context,
        history=s.history,
        label=Label(
            tool_calls=[ToolCall(name="ewallet.transfer", arguments={"phone": "0912345678", "amount": 200000})],
        ),
    )
    out = generate_when2call_contrastive_samples([s_amount], tier1_catalog, count=2, seed=42)
    assert len(out) >= 1
    step = out[0]
    sys_content = step["prompt_messages"][0]["content"]
    assert "Ngữ cảnh hiện tại" in sys_content
    assert "Quy tắc bắt buộc" in sys_content
    user_msgs = [m for m in step["prompt_messages"] if m["role"] == "user"]
    assert len(user_msgs) == 2
    assert user_msgs[0]["content"] == "Hôm nay mấy giờ có lịch?"
    assert set(step["tools"]) == set(s.context["installed_apps"].keys())


def test_when2call_post_lookup_ambiguity_and_multi_intent_ask(tier1_catalog):
    s_post_ask = Sample(
        id="train_post_ask_01",
        audio_text="Đặt báo thức 6 rưỡi sáng mai với gọi cho Minh giúp tôi.",
        context=_rich_sample_with_history().context,
        label=Label(
            tool_calls=[
                ToolCall(name="alarm.create", arguments={"time": "06:30"}),
                ToolCall(name="contacts.search", arguments={"query": "Minh"}),
                ToolCall(name=ASK_MISSING_INFO, arguments={"tool_name": "phone.call", "params": ["contact_id"]}),
            ],
            tool_results=[
                {"ok": True, "id": "al_1"},
                {
                    "ok": True,
                    "contacts": [
                        {"id": "c_01", "name": "Minh Bạn"},
                        {"id": "c_02", "name": "Minh Sếp"},
                    ],
                },
                {"asked": True},
            ],
        ),
    )
    out = generate_post_lookup_ask_samples([s_post_ask], tier1_catalog, count=4, seed=7)
    assert len(out) >= 1
    step = out[0]
    assert step["step_category"] == "ask"
    assert step["step_calls"][0]["name"] == ASK_MISSING_INFO
    assert any(m["role"] == "tool" and "Minh Sếp" in m["content"] for m in step["prompt_messages"])
    assert "Ngữ cảnh hiện tại" in step["prompt_messages"][0]["content"]


def test_anchor_binding_extracts_later_edges_and_preserves_context(tier1_catalog):
    s = _rich_sample_with_history()
    out = generate_anchor_binding_samples([s], tier1_catalog, count=4, seed=42)
    assert len(out) == 1
    step = out[0]
    assert step["step_calls"][0]["name"] == "sms.send"
    assert step["step_calls"][0]["arguments"]["contact_id"] == "c_01"
    assert "Ngữ cảnh hiện tại" in step["prompt_messages"][0]["content"]
    assert any(m["role"] == "user" and m["content"] == "Hôm nay mấy giờ có lịch?" for m in step["prompt_messages"])
    assert set(step["tools"]) == set(s.context["installed_apps"].keys())


def test_apigen_mt_does_not_mix_dependent_read_write_in_same_step(tier1_catalog):
    s1 = Sample(
        id="t1",
        audio_text="Đặt báo thức 6 rưỡi",
        context={"installed_apps": {"alarm": {}}},
        label=Label(tool_calls=[ToolCall(name="alarm.create", arguments={"time": "06:30"})]),
    )
    s2 = Sample(
        id="t2",
        audio_text="Kiểm tra hóa đơn rồi thanh toán bill_01 bằng ngân hàng",
        context={"installed_apps": {"bill_payment": {}}},
        label=Label(
            tool_calls=[
                ToolCall(name="bill_payment.list_bills", arguments={}),
                ToolCall(name="bill_payment.pay", arguments={"bill_id": "bill_01", "source": "bank"}),
            ],
            tool_results=[
                {"ok": True, "bills": [{"id": "bill_01", "amount": 200000}]},
                {"ok": True},
            ],
        ),
    )
    out = generate_apigen_multi_intent_samples([s1, s2], tier1_catalog, count=2, seed=42, include_step_1=True)
    for st in out:
        names = [c["name"] for c in st["step_calls"]]
        assert not ("bill_payment.list_bills" in names and "bill_payment.pay" in names)
        assert "Quy tắc bắt buộc" in st["prompt_messages"][0]["content"]


def test_execution_verifier_checks_required_enum_and_x_one_of(tier1_catalog):
    bad_req = {"step_calls": [{"name": "bill_payment.pay", "arguments": {"bill_id": "bill_01"}}]}
    ok, reason = validate_step_with_reason(bad_req, tier1_catalog)
    assert not ok and "source" in reason
    assert verify_step_validity(bad_req, tier1_catalog) is False

    bad_enum = {"step_calls": [{"name": "bill_payment.pay", "arguments": {"bill_id": "bill_01", "source": "crypto"}}]}
    ok, reason = validate_step_with_reason(bad_enum, tier1_catalog)
    assert not ok and "enum" in reason.lower()

    bad_one_of = {"step_calls": [{"name": "sms.send", "arguments": {"content": "Alo"}}]}
    ok, reason = validate_step_with_reason(bad_one_of, tier1_catalog)
    assert not ok and "one-of" in reason.lower()

    good_one_of = {"step_calls": [{"name": "sms.send", "arguments": {"contact_id": "c_01", "content": "Alo"}}]}
    ok, _ = validate_step_with_reason(good_one_of, tier1_catalog)
    assert ok


def test_distill_think_extracts_observation_facts_and_balances():
    step = {
        "step_category": "act",
        "remaining_subgoals": ["reminders.create"],
        "step_calls": [{"name": "bill_payment.pay", "arguments": {"bill_id": "b1", "source": "bank"}}],
        "prompt_messages": [
            {"role": "user", "content": "Nếu ví đủ thì trả tiền điện, không đủ thì trả ngân hàng."},
            {
                "role": "tool",
                "content": json.dumps(
                    {
                        "bills": [{"id": "b1", "amount": 500000}],
                        "balance": 120000,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    think = synthesize_deterministic_think(step)
    assert "Đủ điều kiện thực thi" in think
    assert "500000" in think or "120000" in think
    assert "reminders.create" in think


def test_guardrail_config_and_oracle_minicpm5_config():
    g = GuardrailConfig()
    assert g.enable_boundary_verifier is True
    cfg = load_config("configs/oracle_minicpm5_2b.yaml")
    assert cfg.llm.max_model_len == 16384
    assert cfg.llm.max_new_tokens >= 768
    assert cfg.agent.max_steps >= 10
    assert cfg.agent.max_prompt_tokens >= 15000


def test_boundary_verifier_wired_into_react_loop(tmp_path):
    # Verify enable_boundary_verifier strips hallucinated repeat arg on alarm.create cleanly
    from src.core.dataset import load_catalog, load_split

    if not (pytest.importorskip("pathlib").Path("data/train/train.jsonl").is_file()):
        pytest.skip("competition data missing")
    sample = load_split("train", "data")[0]
    catalog = load_catalog("train", "data")
    sample_alarm = Sample(
        id=sample.id,
        audio=sample.audio,
        audio_text="Đặt báo thức 7 giờ sáng mai",
        context={"user_profile": {}, "installed_apps": {"alarm": {}}},
    )
    env = SafeEnv("train", sample.id, "data")
    fake = SequenceFakeBackend([
        '<function name="alarm.create"><param name="time">07:00</param><param name="repeat">[\'mon\', \'tue\']</param></function>',
        "Đã đặt báo thức.",
    ])
    res = run_react_loop(
        sample_alarm,
        env,
        catalog,
        fake,
        guardrails=GuardrailConfig(enable_boundary_verifier=True),
    )
    calls = env.get_tool_calls()
    assert len(calls) == 1
    assert calls[0]["name"] == "alarm.create"
    assert "repeat" not in calls[0]["arguments"]


def test_v3_dataset_builder_zero_leak_and_clean_prompts(tmp_path):
    import gzip
    from pathlib import Path
    from scripts.build_minicpm5_dataset_v3 import build_v3_dataset

    if not Path("data/train/train.jsonl").is_file():
        pytest.skip("competition data missing")
    hard_file = tmp_path / "hard.json"
    hard_file.write_text(json.dumps(["train_0001"]), encoding="utf-8")

    count, gz_path = build_v3_dataset(
        split_name="train",
        root="data",
        dev_split_path="configs/splits/dev_v1.json",
        out_dir=tmp_path,
        when2call_step0_count=20,
        when2call_post_count=30,
        anchor_count=20,
        apigen_count=20,
        recovery_count=20,
        hard_sample_ids_path=hard_file,
        seed=42,
    )
    assert count > 100
    assert gz_path.exists()
    dev_ids = set(json.loads(Path("configs/splits/dev_v1.json").read_text(encoding="utf-8"))["dev_ids"])
    sources = set()
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            assert row["sample_id"] not in dev_ids
            assert "Quy tắc bắt buộc" in row["messages"][0]["content"]
            assert "</think>" in row["messages"][-1]["content"]
            # Make sure recovery steps do not accidentally keep broken OpenAI <tool_call> tags if XML expected
            sources.add(row["source"])
    assert "when2call_post_lookup" in sources
    assert "looptool_edde" in sources
    assert any(s.startswith("recovery_") for s in sources)


def test_sft_chunked_resume_zero_overlap_full_coverage(monkeypatch):
    import random
    import sys
    from types import ModuleType
    from scripts.train_minicpm5_sft import (
        TRAIN_CHAT_TEMPLATE,
        build_epoch_schedule,
        slice_dataset_by_epoch_progress,
        token_len_from_template_output,
    )

    try:
        from datasets import Dataset
    except ImportError:
        class _MiniDataset:
            def __init__(self, rows):
                self._rows = list(rows)

            @classmethod
            def from_list(cls, rows):
                return cls(rows)

            def __len__(self):
                return len(self._rows)

            def __getitem__(self, key):
                if isinstance(key, str):
                    return [r[key] for r in self._rows]
                return self._rows[key]

            def shuffle(self, seed=42):
                idxs = list(range(len(self._rows)))
                random.Random(seed).shuffle(idxs)
                return _MiniDataset([self._rows[i] for i in idxs])

            def select(self, indices):
                return _MiniDataset([self._rows[i] for i in indices])

        fake_mod = ModuleType("datasets")
        fake_mod.Dataset = _MiniDataset
        fake_mod.concatenate_datasets = lambda parts: _MiniDataset([r for p in parts for r in p._rows])
        monkeypatch.setitem(sys.modules, "datasets", fake_mod)
        Dataset = _MiniDataset

    assert "+ '<|im_end|>\\n' }}{%- endgeneration %}" in TRAIN_CHAT_TEMPLATE
    assert token_len_from_template_output([1, 2, 3, 4]) == 4
    assert token_len_from_template_output({"input_ids": [1, 2, 3, 4, 5], "attention_mask": [1] * 5}) == 5
    assert token_len_from_template_output({"input_ids": [[10, 20, 30]]}) == 3

    ds = Dataset.from_list([{"id": i} for i in range(100)])
    full = build_epoch_schedule(ds, total_epochs=2.0, seed=42)
    assert len(full) == 200
    assert set(full["id"][:100]) == set(range(100))
    assert set(full["id"][100:]) == set(range(100))

    c1, s1, tot1 = slice_dataset_by_epoch_progress(full, total_epochs=2.0, epoch_offset=0.0)
    c2, s2, tot2 = slice_dataset_by_epoch_progress(full, total_epochs=2.0, epoch_offset=0.25)
    c3, s3, tot3 = slice_dataset_by_epoch_progress(full, total_epochs=2.0, epoch_offset=1.35)
    assert (s1, tot1) == (0, 200)
    assert (s2, tot2) == (25, 200)
    assert (s3, tot3) == (135, 200)

    seen_ids_order = list(c1["id"][:25]) + list(c2["id"][: 135 - 25]) + list(c3["id"])
    assert seen_ids_order == list(full["id"])
    assert len(seen_ids_order) == 200


def test_hf_checkpoint_callback_epoch_scale(tmp_path):
    from types import SimpleNamespace
    from src.core.hf_sync import make_checkpoint_callback

    pushed = []

    class DummyModel:
        def save_pretrained(self, d):
            from pathlib import Path
            Path(d).mkdir(parents=True, exist_ok=True)
            (Path(d) / "adapter_model.safetensors").write_bytes(b"ok")

    class DummyPusher:
        def submit(self, local_dir, path_in_repo, message, on_done=None):
            prog = json.loads((local_dir / "progress.json").read_text(encoding="utf-8"))
            pushed.append((path_in_repo, prog))
            if on_done:
                on_done(True)

        def flush(self, timeout=None):
            return True

    cb = make_checkpoint_callback(
        DummyPusher(),
        tmp_path / "snap",
        interval_s=0.0,
        save_steps=1,
        path_in_repo="exp_test/checkpoints",
        save_latest_pointer=True,
        step_offset=10,
        epoch_offset=0.5,
        epoch_scale=1.5,
    )
    state = SimpleNamespace(is_world_process_zero=True, global_step=2, max_steps=10, epoch=0.5, log_history=[{"loss": 0.25}])
    cb.on_step_end(None, state, SimpleNamespace(), model=DummyModel())
    assert len(pushed) == 2
    assert pushed[0][0] == "exp_test/checkpoints/checkpoint-12"
    assert pushed[0][1]["epoch"] == pytest.approx(1.25)
    assert pushed[1][0] == "exp_test/checkpoints/latest"
