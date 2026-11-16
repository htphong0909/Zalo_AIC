"""Declarative configuration schemas for ASR, LLM, Guardrails, and Agent pipelines."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ASRConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: Literal["oracle", "cached", "qwen3_asr", "whisper", "chunkformer"] = "oracle"
    model_id: str = "Qwen/Qwen3-ASR-0.6B-hf"
    adapter_path: str | None = None
    transcripts_path: str | None = None
    language: str = "vi"
    task: str = "transcribe"
    dtype: str = "bfloat16"
    enable_itn: bool = True
    batch_size: int = 8
    no_context_bias: bool = False
    nbest_beams: int = 1
    nbest_max_delta: float = 0.65


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["transformers", "vllm", "fake"] = "transformers"
    model_id: str = "Qwen/Qwen3.5-4B"
    adapter_path: str | None = None
    max_model_len: int = 16384
    max_new_tokens: int = 384
    enable_thinking: bool = False
    dtype: str = "bfloat16"
    repetition_penalty: float = 1.0
    gpu_memory_utilization: float = 0.85
    max_num_seqs: int | None = None
    guided_decoding: bool = False


class GuardrailConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enable_xref_guard: bool = True
    enable_arg_sanitizer: bool = True
    enable_ask_guard: bool = True
    enable_boundary_verifier: bool = True


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_style: str = "v1"
    tool_selector: Literal["installed_apps"] = "installed_apps"
    max_steps: int = 8
    max_repairs: int = 2
    timeout_s: float = 25.0
    max_prompt_tokens: int = 12000
    max_result_chars: int = 1500
    batch_size: int = 1
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "oracle_qwen35_4b"
    seed: int = 42
    asr: ASRConfig = Field(default_factory=ASRConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    p = Path(path)
    raw_text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data: dict[str, Any] = json.loads(raw_text)
    else:
        data = yaml.safe_load(raw_text) or {}
    return ExperimentConfig.model_validate(data)
