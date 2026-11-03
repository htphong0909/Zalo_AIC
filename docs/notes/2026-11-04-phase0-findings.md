# Phase 0 Findings — Parameter Budget & Colab Smoke Test

- **Date:** 2026-11-04
- **GPU tested:** NVIDIA L4 (24 GB VRAM) via `colab` CLI (`phase0-smoke`)
- **Runtime versions:** `torch==2.11.0+cu130`, `transformers==5.22.1`

---

## 1. Exact Parameter Counts (Safetensors & Loaded Tensors)

| Component | Repo ID | Stored Tensors (Safetensors) | Loaded Active Params (`bfloat16`) | Verdict |
|---|---|---:|---:|---|
| **ASR (Primary)** | `Qwen/Qwen3-ASR-0.6B-hf` | 782,426,112 (tied `lm_head`) / 938,008,576 (`Qwen/Qwen3-ASR-0.6B`) | **782,426,112** (~0.78B) | ✅ |
| **LLM (Primary)** | `Qwen/Qwen3.5-4B` (`Qwen3_5ForCausalLM`) | 4,659,865,088 (full repo incl. `model.visual` 333.5M + `mtp` 120.6M) | **4,205,751,296** (~4.21B) | ✅ < 5B (both loaded & full repo) |
| **Combined Submission System** | `Qwen3-ASR-0.6B-hf` + `Qwen3.5-4B` | 5,442,291,200 (if counting unused vision/MTP shards) | **4,988,177,408** (~4.99B) | ✅ **< 6B under every counting method** |
| **Reference ASR (Dev only)** | `Qwen/Qwen3-ASR-1.7B` | 2,349,217,408 (~2.35B) | ~2.04B–2.35B | ❌ Combined > 6B (dev WER reference only) |

### Key Takeaway on Text-Only Export
- Even if an auditor sums **all** tensors in the raw Hugging Face repositories without excluding `model.visual` or `mtp`, `Qwen/Qwen3-ASR-0.6B-hf` (782.4M) + `Qwen/Qwen3.5-4B` (4,659.9M) = **5,442,291,200 parameters (< 6B)**, and `Qwen/Qwen3.5-4B` alone is **4,659,865,088 (< 5B)**.
- Furthermore, loading `Qwen/Qwen3.5-4B` via `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B")` in `transformers>=5.22.0` automatically instantiates `Qwen3_5ForCausalLM` with **4,205,751,296** parameters (skipping `model.visual` and `mtp`), giving a total active parameter count of **4,988,177,408 (~4.99B)**.
- For the final offline Docker image, saving `Qwen3_5ForCausalLM.save_pretrained(...)` drops the unused vision/MTP weights on disk as well, saving ~900 MB of image size.

---

## 2. Runtime & Format Discoveries (Critical for Phase 2 & Phase 3)

1. **Pure `transformers>=5.22.0` supports both models natively:**
   - `Qwen/Qwen3-ASR-0.6B-hf` loads via `AutoProcessor` + `AutoModelForSeq2SeqLM` (`Qwen3ASRForConditionalGeneration`) with no external `qwen-asr` package required (avoiding `qwen-asr`'s `transformers 4.x` version conflict).
   - ASR output format includes a language header:
     `language Vietnamese<asr_text>Cái còn lại nếu may mưa thì rời muộn nửa tiếng, không mưa giữ nguyên.`
     → The ASR wrapper in `src/asr/` must strip the `language ...<asr_text>` prefix.
2. **`Qwen/Qwen3.5-4B` native tool-calling syntax:**
   - `chat_template.jinja` in `Qwen/Qwen3.5-4B` emits XML-style tags inside `<tool_call>`:
     ```xml
     <tool_call>
     <function=smart_home.light_set>
     <parameter=device_id>
     light_kitchen
     </parameter>
     <parameter=power>
     on
     </parameter>
     </function>
     </tool_call>
     ```
   - The tool-call parser in `src/agent/` (Phase 2) must support both this native `<function=...><parameter=...>` format and JSON `<tool_call>{"name": ...}</tool_call>` fallback.
3. **VRAM & Latency on 24 GB GPU (L4):**
   - ASR (`Qwen3-ASR-0.6B-hf`, bf16): **1.51 GB VRAM**, first-call latency **2.03 s**.
   - Both ASR + LLM (`Qwen3.5-4B`, bf16) resident simultaneously: **7.99 GB peak VRAM** (leaving **16 GB VRAM headroom** on a 24 GB L4/A30 GPU).
   - Single tool-call generation latency (eager PyTorch): **3.30 s** (installing `flash-linear-attention` and `causal_conv1d` or running via vLLM reduces this further).
