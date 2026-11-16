"""Qwen3-ASR engine using pure transformers AutoProcessor + AutoModelForSeq2SeqLM with Vietnamese domain context biasing."""
from __future__ import annotations

import time
from pathlib import Path

from src.asr.audio import load_wav_16k
from src.asr.base import ASRResult, strip_asr_tags

DEFAULT_VI_ASR_SYSTEM_PROMPT = (
    "Trợ lý ảo tiếng Việt hỗ trợ điều khiển nhà thông minh, đặt báo thức, nhắc nhở, lịch, "
    "danh bạ, gọi điện, nhắn tin SMS, Zalo, email, chuyển khoản ngân hàng, ví điện tử, "
    "thanh toán hóa đơn, đặt xe, giao đồ ăn, thời tiết, ghi chú, công việc. "
    "Các tên thường gặp: Minh, Lan, Tuấn, Hương, Sếp Dũng, Bác Ba, Cô giáo Mai, Bố, Mẹ, "
    "An, Thảo, Huy, Nam, Bình, Phượng, Sarah, Emily, David Miller, Sophie Martin."
)


def build_qwen_asr_prompt(
    system_prompt: str = DEFAULT_VI_ASR_SYSTEM_PROMPT,
    *,
    force_vi_prefix: bool = True,
) -> str:
    prefix = "language Vietnamese<asr_text>" if force_vi_prefix else ""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        f"<|im_start|>assistant\n{prefix}"
    )


class Qwen3ASREngine:
    """Deterministic Qwen3-ASR-0.6B-hf / 1.7B wrapper with Vietnamese context biasing."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-ASR-0.6B-hf",
        *,
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        max_new_tokens: int = 256,
        system_prompt: str = DEFAULT_VI_ASR_SYSTEM_PROMPT,
        force_vi_prefix: bool = True,
        enable_itn: bool = True,
    ):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoProcessor

        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.force_vi_prefix = force_vi_prefix
        self.enable_itn = enable_itn
        self.processor = AutoProcessor.from_pretrained(adapter_path or model_id)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            dtype=torch_dtype,
            device_map="auto",
        )
        if adapter_path:
            from peft import PeftModel

            base_model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model = base_model
        self.model.eval()

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        extra_context: str | None = None,
    ) -> ASRResult:
        import torch

        t0 = time.perf_counter()
        waveform, sr = load_wav_16k(audio_path, target_sr=16000)
        sys_text = self.system_prompt
        if extra_context:
            sys_text = f"{sys_text} {extra_context}".strip()
        prompt = build_qwen_asr_prompt(sys_text, force_vi_prefix=self.force_vi_prefix)
        inputs = self.processor(
            text=prompt,
            audio=[waveform],
            sampling_rate=sr,
            return_tensors="pt",
        )
        inputs = {
            k: (
                v.to(self.model.device, dtype=self.model.dtype)
                if v.is_floating_point()
                else v.to(self.model.device)
            )
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        gen_ids = out_ids[:, prompt_len:] if prompt_len and out_ids.shape[1] > prompt_len else out_ids
        raw_text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        clean_text = strip_asr_tags(raw_text)
        if self.enable_itn:
            from src.asr.itn import inverse_text_normalize

            clean_text = inverse_text_normalize(clean_text)
        return ASRResult(
            text=clean_text,
            raw_text=raw_text,
            latency_s=time.perf_counter() - t0,
        )
