import pytest
from src.agent.lfm3_grpo import (
    build_lfm3_grpo_config,
    build_lfm3_grpo_lora_config,
)
from src.agent.lfm3_sft import load_yaml_config


def test_lfm3_grpo_lora_config():
    cfg = build_lfm3_grpo_lora_config(r=32, alpha=64, dropout=0.0)
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
    assert cfg.lora_dropout == 0.0


def test_load_grpo_yaml_and_build_grpo_config():
    raw = load_yaml_config("configs/lfm3_grpo.yaml")
    assert raw["lora"]["r"] == 32
    assert "gate_proj" in raw["lora"]["target_modules"]
    grpo_cfg = build_lfm3_grpo_config(raw, use_bf16=False)
    assert grpo_cfg.num_generations == 8
    assert grpo_cfg.generation_batch_size == 8
    assert grpo_cfg.max_completion_length == 640
    assert pytest.approx(grpo_cfg.beta, 0.001) == 0.04
    assert pytest.approx(grpo_cfg.learning_rate, 1e-7) == 1e-5
