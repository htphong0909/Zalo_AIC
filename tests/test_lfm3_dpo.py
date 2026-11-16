from __future__ import annotations

import pytest

from src.agent.lfm3_dpo import build_lfm3_dpo_config, build_lfm3_dpo_lora_config
from src.agent.lfm3_sft import load_yaml_config


def test_lfm3_dpo_lora_and_config():
    lora_cfg = build_lfm3_dpo_lora_config(r=32, alpha=64, dropout=0.05)
    assert "gate_proj" in set(lora_cfg.target_modules)
    assert lora_cfg.r == 32
    assert lora_cfg.lora_alpha == 64

    raw = load_yaml_config("configs/lfm3_dpo.yaml")
    assert "gate_proj" in raw["lora"]["target_modules"]
    dpo_cfg = build_lfm3_dpo_config(raw, use_bf16=False)
    assert list(dpo_cfg.loss_type) == ["sigmoid", "sft"]
    assert list(dpo_cfg.loss_weights) == [1.0, 1.0]
    assert pytest.approx(dpo_cfg.beta, 0.001) == 0.1
    assert dpo_cfg.max_length == 12288
