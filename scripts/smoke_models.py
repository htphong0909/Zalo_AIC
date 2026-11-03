"""Smoke test: load ASR (Qwen/Qwen3-ASR-0.6B-hf) and LLM (Qwen/Qwen3.5-4B) natively in transformers>=5.2."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path


def run_asr(model_id: str, wav: str) -> dict:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoProcessor

    if not os.path.exists(wav) and os.path.exists("/sample.wav"):
        wav = "/sample.wav"
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0")
    load_s = round(time.time() - t0, 2)
    loaded_params = sum(p.numel() for p in model.parameters())

    t1 = time.time()
    msgs = [{"role": "user", "content": [{"type": "audio", "audio": wav}]}]
    inputs = processor.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    inputs = inputs.to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    gen_ids = out[:, inputs["input_ids"].shape[1] :]
    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    infer_s = round(time.time() - t1, 2)
    vram_gb = round(torch.cuda.max_memory_allocated() / (1024**3), 2)
    return {
        "model": model_id,
        "model_class": type(model).__name__,
        "loaded_params": loaded_params,
        "load_s": load_s,
        "infer_s": infer_s,
        "vram_gb": vram_gb,
        "text": text,
    }


def run_llm(model_id: str, max_len: int) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0")
    load_s = round(time.time() - t0, 2)
    loaded_params = sum(p.numel() for p in model.parameters())

    tools = [
        {
            "type": "function",
            "function": {
                "name": "smart_home.light_set",
                "description": "Bật/tắt đèn",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "device_id": {"type": "string"},
                        "power": {"type": "string", "enum": ["on", "off"]},
                    },
                    "required": ["device_id", "power"],
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "Tắt đèn bếp (device_id: light_kitchen)."}]
    prompt = tok.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tok([prompt], return_tensors="pt").to(model.device)
    assert inputs.input_ids.shape[1] <= max_len
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    gen_ids = out[0][inputs.input_ids.shape[1] :]
    text = tok.decode(gen_ids, skip_special_tokens=False)
    infer_s = round(time.time() - t1, 2)
    vram_gb = round(torch.cuda.max_memory_allocated() / (1024**3), 2)
    return {
        "model": model_id,
        "model_class": type(model).__name__,
        "loaded_params": loaded_params,
        "load_s": load_s,
        "infer_s": infer_s,
        "peak_vram_gb": vram_gb,
        "output": text,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="Qwen/Qwen3.5-4B")
    p.add_argument("--asr", default="Qwen/Qwen3-ASR-0.6B-hf")
    p.add_argument("--wav", default="/content/sample.wav")
    p.add_argument("--max-len", type=int, default=16384)
    p.add_argument("--out", default="/smoke_result.json")
    a, _ = p.parse_known_args()
    import torch
    import transformers

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    asr_info = run_asr(a.asr, a.wav)
    print("ASR_RESULT_JSON:", json.dumps(asr_info, ensure_ascii=True))
    llm_info = run_llm(a.llm, a.max_len)
    info = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "total_loaded_params": asr_info["loaded_params"] + llm_info["loaded_params"],
        "asr": asr_info,
        "llm": llm_info,
    }
    Path(a.out).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SMOKE_RESULT_JSON:", json.dumps(info, ensure_ascii=True, indent=2))
