import pytest
from src.agent.lfm3_sft import (
    LFM3_TARGET_MODULES,
    build_lfm3_lora_config,
    build_lfm3_sft_config,
    load_yaml_config,
)


def test_lfm3_lora_target_modules():
    cfg = build_lfm3_lora_config(r=32, alpha=64, dropout=0.05)
    assert set(cfg.target_modules) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "in_proj",
        "out_proj",
        "w1",
        "w2",
        "w3",
    }
    assert cfg.r == 32
    assert cfg.lora_alpha == 64
    assert pytest.approx(cfg.lora_dropout, 0.001) == 0.05
    assert cfg.task_type == "CAUSAL_LM"


def test_load_sft_yaml_and_build_sft_config():
    raw = load_yaml_config("configs/lfm3_sft.yaml")
    assert raw["model_id"] == "LiquidAI/LFM3-1.7B"
    assert raw["lora"]["r"] == 32
    assert raw["lora"]["alpha"] == 64
    assert "gate_proj" in raw["lora"]["target_modules"]
    sft_cfg = build_lfm3_sft_config(raw, use_bf16=False)
    assert sft_cfg.max_steps == -1
    assert sft_cfg.num_train_epochs == 3.0
    assert sft_cfg.max_length == 10240
    assert sft_cfg.per_device_train_batch_size == 2
    assert sft_cfg.gradient_accumulation_steps == 8
    assert pytest.approx(sft_cfg.learning_rate, 1e-6) == 2.0e-4
