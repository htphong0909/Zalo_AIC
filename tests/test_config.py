from __future__ import annotations

from src.core.config import ExperimentConfig, load_config


def test_load_yaml_configs():
    cfg_oracle = load_config("configs/oracle_qwen35_4b.yaml")
    assert isinstance(cfg_oracle, ExperimentConfig)
    assert cfg_oracle.asr.engine == "oracle"
    assert cfg_oracle.llm.model_id == "Qwen/Qwen3.5-4B"
    assert cfg_oracle.agent.guardrails.enable_xref_guard is True
    assert cfg_oracle.agent.guardrails.enable_arg_sanitizer is True
    assert cfg_oracle.agent.guardrails.enable_ask_guard is True

    cfg_raw = load_config("configs/oracle_qwen35_4b_raw.yaml")
    assert cfg_raw.agent.guardrails.enable_xref_guard is False
    assert cfg_raw.agent.guardrails.enable_arg_sanitizer is False
    assert cfg_raw.agent.guardrails.enable_ask_guard is False

    cfg_e2e = load_config("configs/e2e_qwen3_asr_06b_qwen35_4b.yaml")
    assert cfg_e2e.asr.engine == "qwen3_asr"
    assert cfg_e2e.asr.model_id == "Qwen/Qwen3-ASR-0.6B-hf"


def test_load_phowhisper_and_lfm3_configs():
    cfg_pw_qwen = load_config("configs/e2e_phowhisper_qwen35_4b.yaml")
    assert cfg_pw_qwen.asr.engine == "whisper"
    assert cfg_pw_qwen.asr.model_id == "kzap201/phowhisper-large-zaloai"
    assert cfg_pw_qwen.asr.task == "transcribe"
    assert cfg_pw_qwen.asr.enable_itn is True

    cfg_pw_lfm3 = load_config("configs/e2e_phowhisper_lfm3_17b.yaml")
    assert cfg_pw_lfm3.asr.engine == "whisper"
    assert cfg_pw_lfm3.asr.model_id == "kzap201/phowhisper-large-zaloai"
    assert cfg_pw_lfm3.asr.dtype == "float16"
    assert cfg_pw_lfm3.asr.task == "transcribe"
    assert cfg_pw_lfm3.asr.enable_itn is True
    assert cfg_pw_lfm3.llm.model_id == "LiquidAI/LFM3-1.7B"
    assert cfg_pw_lfm3.llm.dtype == "bfloat16"
    assert cfg_pw_lfm3.llm.repetition_penalty == 1.1
    assert cfg_pw_lfm3.llm.enable_thinking is False


def test_load_qwen38_distill_configs():
    import yaml
    from pathlib import Path

    cfg_oracle = load_config("configs/oracle_qwen38_4b_distill.yaml")
    assert cfg_oracle.asr.engine == "oracle"
    assert cfg_oracle.llm.model_id == "empero-ai/Qwen3.8-4B-Distill"
    assert cfg_oracle.llm.enable_thinking is False
    assert cfg_oracle.llm.repetition_penalty == 1.05
    assert cfg_oracle.agent.prompt_style == "qwen35"

    cfg_e2e = load_config("configs/e2e_phowhisper_qwen38_4b_distill.yaml")
    assert cfg_e2e.asr.engine == "whisper"
    assert cfg_e2e.asr.model_id == "kzap201/phowhisper-large-zaloai"
    assert cfg_e2e.llm.model_id == "empero-ai/Qwen3.8-4B-Distill"
    assert cfg_e2e.agent.prompt_style == "qwen35"

    models_cfg = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))
    assert models_cfg["llm"]["repo_id"] == "empero-ai/Qwen3.8-4B-Distill"
    assert "model.visual." in models_cfg["llm"]["exclude_prefixes"]
    assert "mtp." in models_cfg["llm"]["exclude_prefixes"]


def test_llm_config_accepts_guided_decoding_flag():
    cfg = ExperimentConfig()
    assert cfg.llm.guided_decoding is False
    cfg_on = ExperimentConfig.model_validate({"llm": {"guided_decoding": True}})
    assert cfg_on.llm.guided_decoding is True


