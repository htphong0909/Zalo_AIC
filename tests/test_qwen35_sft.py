from __future__ import annotations

import glob
import json
from pathlib import Path

from scripts.train_qwen38_sft import resolve_resume_from_hf
from src.agent.qwen35_sft import (
    QWEN35_LORA_TARGET_MODULES,
    format_qwen35_tool_calls_xml,
    render_qwen35_prompt_completion,
    save_qwen35_text_only_checkpoint,
    step_to_qwen35_example,
)
from src.agent.tool_caller import extract_final_text, parse_tool_calls
from src.core.dataset import load_catalog
from src.core.schemas import ToolCall


def test_format_qwen35_tool_calls_xml_roundtrip():
    catalog = load_catalog("train", "data")
    calls = [
        ToolCall(name="smart_home.light_set", arguments={"device_id": "d_01", "power": "on", "brightness": 75}),
        ToolCall(name="phone.call", arguments={"phone_number": "0912345678"}),
        ToolCall(name="sys.ask_missing_info", arguments={"tool_name": "sms.send", "params": ["contact_id"]}),
    ]
    xml = format_qwen35_tool_calls_xml(calls)
    parsed = parse_tool_calls(xml, catalog)
    assert parsed == calls


def test_step_to_qwen35_example_and_render_with_cached_chat_template():
    catalog = load_catalog("train", "data")
    step = {
        "id": "mock_step",
        "step_category": "act",
        "prompt_messages": [
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý giọng nói LifeMate.\n"
                    "## Ngữ cảnh hiện tại\n"
                    '7. **Ngắn gọn tối đa:** ... `<function name="..."><param name="...">...</param></function>`'
                ),
            },
            {"role": "user", "content": "Đặt báo thức 6 giờ sáng mai"},
        ],
        "step_calls": [{"name": "alarm.create", "arguments": {"time": "2026-11-15T06:00:00+07:00"}}],
    }
    ctx = {
        "current_time": "2026-11-14T12:00:00+07:00",
        "timezone": "Asia/Ho_Chi_Minh",
        "location": {},
        "user_profile": {},
        "installed_apps": {"alarm": {"tools": ["alarm.create"]}},
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": "alarm.create",
                "description": "Tạo báo thức",
                "parameters": {"type": "object", "properties": {"time": {"type": "string"}}, "required": ["time"]},
            },
        }
    ]
    ex_no_think = step_to_qwen35_example(step, ctx, tools, catalog, enable_thinking=False)
    sys_msg = ex_no_think["prompt_messages"][0]["content"]
    assert "<tool_call><function=tên_tool>" in sys_msg
    assert '<function name="...">' not in sys_msg
    assert ex_no_think["completion"].endswith("<|im_end|>")
    assert "</think>" not in ex_no_think["completion"]
    parsed_no_think = parse_tool_calls(ex_no_think["completion"].removesuffix("<|im_end|>"), catalog)
    assert parsed_no_think == [ToolCall(name="alarm.create", arguments={"time": "2026-11-15T06:00:00+07:00"})]

    ex_think = step_to_qwen35_example(step, ctx, tools, catalog, enable_thinking=True)
    assert "</think>\n\n<tool_call>" in ex_think["completion"]
    assert ex_think["completion"].endswith("<|im_end|>")
    assert extract_final_text(ex_think["completion"].removesuffix("<|im_end|>")) == ""

    # Verify against the exact cached Qwen3.8-4B-Distill Jinja template
    import jinja2

    snaps = glob.glob(
        str(Path.home() / ".cache/huggingface/hub/models--empero-ai--Qwen3.8-4B-Distill/snapshots/*/chat_template.jinja")
    )
    if snaps:
        tpl_src = Path(snaps[0]).read_text(encoding="utf-8")

        class _JinjaTok:
            def apply_chat_template(
                self, messages, tools=None, tokenize=False, add_generation_prompt=True, enable_thinking=False
            ):
                env = jinja2.Environment()
                env.filters["tojson"] = lambda v: json.dumps(v, ensure_ascii=False)
                tpl = env.from_string(tpl_src)
                return tpl.render(
                    messages=messages,
                    tools=tools,
                    add_generation_prompt=add_generation_prompt,
                    enable_thinking=enable_thinking,
                )

        rendered = render_qwen35_prompt_completion(_JinjaTok(), ex_no_think)
        assert rendered["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        assert "<function=alarm.create>" in rendered["completion"]
        assert "If you choose to call a function ONLY reply" in rendered["prompt"]
    else:
        class _FallbackTok:
            def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=True, enable_thinking=False):
                return f"{messages[0]['content']}|n_tools={len(tools or [])}|think={enable_thinking}"

        rendered = render_qwen35_prompt_completion(_FallbackTok(), ex_no_think)
        assert "n_tools=1|think=False" in rendered["prompt"]
        assert "<function=alarm.create>" in rendered["completion"]


def test_lora_targets_cover_hybrid_gated_deltanet_and_attention():
    for expected in (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_a",
        "in_proj_b",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ):
        assert expected in QWEN35_LORA_TARGET_MODULES


def test_step_to_qwen35_strips_minicpm5_xml_in_assistant_history_and_preserves_hammer_system():
    catalog = load_catalog("train", "data")
    step = {
        "id": "leak_check",
        "step_category": "act",
        "prompt_messages": [
            {
                "role": "system",
                "content": "Bạn là trợ lý giọng nói...\n7. **Ngắn gọn tối đa:** <tool_call><function=tên_tool></function></tool_call> dùng fn_abcde",
            },
            {"role": "user", "content": "Bật đèn"},
            {
                "role": "assistant",
                "content": '<function name="smart_home.get_status"></function>',
                "tool_calls": [{"type": "function", "function": {"name": "fn_abcde", "arguments": {}}}],
            },
            {"role": "tool", "name": "fn_abcde", "content": '{"devices": [{"id": "d1"}]}'},
        ],
        "step_calls": [{"name": "fn_light", "arguments": {"device_id": "d1", "power": "on"}}],
    }
    ctx = {"installed_apps": {"smart_home": {"tools": ["smart_home.get_status", "smart_home.light_set"]}}}
    ex = step_to_qwen35_example(step, ctx, [], catalog, enable_thinking=False)
    assert "fn_abcde" in ex["prompt_messages"][0]["content"]
    assert "<function name=" not in ex["prompt_messages"][2]["content"]
    assert ex["prompt_messages"][2]["tool_calls"][0]["function"]["name"] == "fn_abcde"


def test_resolve_resume_from_hf_auto_downloads_latest(tmp_path):
    class _FakeSyncer:
        def __init__(self) -> None:
            self.downloaded: list[str] = []

        def exists(self, path_in_repo: str) -> bool:
            return path_in_repo == "exp_q38/checkpoints/latest/progress.json"

        def download_dir(self, path_in_repo: str, local_dir: Path) -> Path:
            self.downloaded.append(path_in_repo)
            dest = Path(local_dir) / path_in_repo
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "progress.json").write_text(json.dumps({"global_step": 120, "epoch": 0.75}), encoding="utf-8")
            (dest / "adapter_model.safetensors").write_bytes(b"mock")
            return dest

    syncer = _FakeSyncer()
    adapter, step_off, epoch_off = resolve_resume_from_hf(syncer, "exp_q38", tmp_path)
    assert adapter is not None and Path(adapter).exists()
    assert step_off == 120
    assert epoch_off == 0.75
    assert syncer.downloaded == ["exp_q38/checkpoints/latest"]


def test_save_qwen35_text_only_checkpoint_sets_use_cache_and_saves_tokenizer(tmp_path):
    class _MockCfg:
        use_cache = False

    class _MockModel:
        def __init__(self) -> None:
            self.config = _MockCfg()
            self.saved_dirs: list[str] = []

        def save_pretrained(self, out_dir: str, safe_serialization: bool = True) -> None:
            self.saved_dirs.append(out_dir)
            (Path(out_dir) / "config.json").write_text(
                json.dumps({"architectures": ["Qwen3_5ForCausalLM"], "use_cache": self.config.use_cache}),
                encoding="utf-8",
            )

    class _MockTok:
        def save_pretrained(self, out_dir: str) -> None:
            (Path(out_dir) / "tokenizer.json").write_text("{}", encoding="utf-8")

    out = save_qwen35_text_only_checkpoint(_MockModel(), _MockTok(), tmp_path / "merged_text")
    cfg_saved = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg_saved["architectures"] == ["Qwen3_5ForCausalLM"]
    assert cfg_saved["use_cache"] is True
    assert (out / "tokenizer.json").exists()


def test_qwen38_sft_yaml_valid():
    import yaml

    cfg = yaml.safe_load(Path("configs/qwen38_sft_v1.yaml").read_text(encoding="utf-8"))
    assert cfg["base_model"] == "empero-ai/Qwen3.8-4B-Distill"
    assert cfg["prompt_style"] == "qwen35"
    assert cfg["training"]["max_length"] == 16384
    for mod in ("in_proj_qkv", "in_proj_z", "out_proj", "q_proj"):
        assert mod in cfg["lora"]["target_modules"]


def test_qwen35_sft_v2_yaml_and_irrelevance_builder():
    import yaml
    from scripts.build_qwen38_sft_data import generate_irrelevance_oos_samples
    from src.core.schemas import Sample

    cfg = yaml.safe_load(Path("configs/qwen35_sft_v2.yaml").read_text(encoding="utf-8"))
    assert cfg["base_model"] == "Qwen/Qwen3.5-4B"
    assert cfg["prompt_style"] == "qwen35"
    assert cfg["training"]["max_length"] == 16384
    assert cfg["training"]["learning_rate"] == 5.0e-5
    assert cfg["training"]["weight_decay"] == 0.01
    for mod in ("in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj", "q_proj"):
        assert mod in cfg["lora"]["target_modules"]

    s = Sample(
        id="irr_1",
        audio_text="Chuyển 200k qua ví",
        context={
            "installed_apps": {
                "ewallet": {"tools": ["ewallet.transfer"]},
                "banking": {"tools": ["banking.transfer"]},
                "weather": {"tools": ["weather.get_current"]},
                "alarm": {"tools": ["alarm.create"]},
                "music": {"tools": ["music.play"]},
            }
        },
        label={"tool_calls": [{"name": "ewallet.transfer", "arguments": {"recipient_phone": "090", "amount": 200000}}]},
    )
    oos = generate_irrelevance_oos_samples([s], load_catalog("train", "data"), count=5, seed=42)
    assert len(oos) == 1
    assert oos[0]["step_category"] == "oos"
    assert oos[0]["step_calls"] == []
    assert "ewallet" not in oos[0]["tools"] and "banking" not in oos[0]["tools"]

