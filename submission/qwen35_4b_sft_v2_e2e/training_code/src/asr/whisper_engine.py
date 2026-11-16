"""Whisper ASR engine supporting Qualcomm PhoASR-whisper-small, PhoWhisper Large, and OpenAI Whisper models."""
from __future__ import annotations

import re
import time
import unicodedata
from pathlib import Path

import numpy as np

from src.asr.audio import load_wav_16k
from src.asr.base import ASRResult, strip_asr_tags

DEFAULT_WHISPER_PROMPT = (
    "Trợ lý ảo tiếng Việt LifeMate điều khiển thiết bị thông minh, điều hoà 26 độ, đèn 30 phần trăm, "
    "chuyển khoản ngân hàng, ví điện tử, đặt xe, liên lạc Zalo, SMS, gọi điện, tra cứu thời tiết, nhắc nhở lịch hẹn. "
    "Dạ, vâng, trợ lý ơi, em ơi, anh ơi, chị ơi."
)


def _sanitize_hyp(raw_text: str, enable_itn: bool = True) -> str:
    """Sanitizes raw Whisper hypothesis by stripping tags and applying ITN."""
    clean_text = strip_asr_tags(raw_text)
    clean_text = unicodedata.normalize("NFC", clean_text).strip()

    # Normalize whitespace before punctuation and collapse consecutive spaces
    clean_text = re.sub(r"\s+([,.:?!])", r"\1", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    # Capitalize leading character if lower
    if clean_text and clean_text[0].islower():
        clean_text = clean_text[0].upper() + clean_text[1:]

    # Inverse text normalization
    if enable_itn:
        from src.asr.itn import inverse_text_normalize

        clean_text = inverse_text_normalize(clean_text)

    return clean_text


def trim_waveform_silence(
    waveform: np.ndarray,
    sr: int = 16000,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    db_threshold: float = -35.0,
    margin_ms: float = 170.0,
) -> np.ndarray:
    """Conservative energy-based boundary silence trimming with safety collar."""
    if len(waveform) == 0:
        return waveform

    frame_len = int(round((frame_ms / 1000.0) * sr))
    hop_len = int(round((hop_ms / 1000.0) * sr))
    margin_len = int(round((margin_ms / 1000.0) * sr))

    num_frames = 1 + (len(waveform) - frame_len) // hop_len
    if num_frames <= 0:
        return waveform

    shape = (num_frames, frame_len)
    strides = (waveform.strides[0] * hop_len, waveform.strides[0])
    frames = np.lib.stride_tricks.as_strided(waveform, shape=shape, strides=strides)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)

    peak_rms = float(np.max(rms))
    if peak_rms < 1e-5:
        return waveform

    rms_db = 20.0 * np.log10(rms / peak_rms)
    active = np.where(rms_db > db_threshold)[0]
    if len(active) == 0:
        return waveform

    start_idx = max(0, active[0] * hop_len - margin_len)
    end_idx = min(len(waveform), (active[-1] * hop_len + frame_len) + margin_len)
    return waveform[start_idx:end_idx]


def preprocess_waveform(
    waveform: np.ndarray,
    sr: int = 16000,
    *,
    trim_silence: bool = True,
    vad_db_threshold: float = -35.0,
    vad_margin_ms: float = 170.0,
    peak_normalize: bool = True,
    pre_roll_pad_ms: float = 60.0,
    post_roll_pad_ms: float = 60.0,
) -> np.ndarray:
    """Applies VAD boundary trimming, peak normalization, and boundary collar padding."""
    if len(waveform) == 0:
        return waveform

    if trim_silence:
        waveform = trim_waveform_silence(
            waveform,
            sr=sr,
            db_threshold=vad_db_threshold,
            margin_ms=vad_margin_ms,
        )

    if peak_normalize:
        peak = float(np.max(np.abs(waveform)))
        if peak > 1e-4:
            waveform = waveform * (0.95 / peak)

    if pre_roll_pad_ms > 0 or post_roll_pad_ms > 0:
        pad_pre = int(round((pre_roll_pad_ms / 1000.0) * sr)) if pre_roll_pad_ms > 0 else 0
        pad_post = int(round((post_roll_pad_ms / 1000.0) * sr)) if post_roll_pad_ms > 0 else 0
        if pad_pre > 0 or pad_post > 0:
            waveform = np.pad(waveform, (pad_pre, pad_post), mode="constant", constant_values=0.0)
    return waveform


class WhisperASREngine:
    """Deterministic Whisper ASR engine (greedy decoding, Vietnamese forced language, pre-roll padded)."""

    def __init__(
        self,
        model_id: str = "kzap201/phowhisper-large-zaloai",
        *,
        device: str = "cuda",
        dtype: str = "float16",
        language: str = "vi",
        task: str = "transcribe",
        enable_itn: bool = True,
        initial_prompt: str | None = DEFAULT_WHISPER_PROMPT,
        attn_implementation: str | None = "sdpa",
        pre_roll_pad_ms: float = 20.0,
        post_roll_pad_ms: float = 60.0,
        peak_normalize: bool = True,
        trim_silence: bool = True,
        vad_db_threshold: float = -35.0,
        vad_margin_ms: float = 170.0,
        repetition_penalty: float = 1.02,
        cache_implementation: str | None = None,
        adapter_id: str | None = None,
    ):
        import os
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        if not torch.cuda.is_available() and device == "cuda":
            device = "cpu"
            torch_dtype = torch.float32

        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.language = language
        self.task = task
        self.enable_itn = enable_itn
        self.initial_prompt = initial_prompt
        self.attn_implementation = attn_implementation
        self.pre_roll_pad_ms = float(pre_roll_pad_ms)
        self.post_roll_pad_ms = float(post_roll_pad_ms)
        self.peak_normalize = bool(peak_normalize)
        self.trim_silence = bool(trim_silence)
        self.vad_db_threshold = float(vad_db_threshold)
        self.vad_margin_ms = float(vad_margin_ms)
        self.repetition_penalty = float(repetition_penalty) if repetition_penalty is not None else 1.0
        self.cache_implementation = cache_implementation
        self._prompt_ids_cache: dict[str, object] = {}

        is_local = Path(model_id).exists()
        is_offline = (
            is_local
            or os.environ.get("HF_HUB_OFFLINE") == "1"
            or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        )
        hf_load_kwargs: dict[str, object] = {"local_files_only": True} if is_offline else {}

        # Attempt standard AutoProcessor with graceful fallback for models with tokenizer quirks
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, **hf_load_kwargs)
        except (AttributeError, Exception):
            from transformers import WhisperFeatureExtractor, WhisperProcessor, WhisperTokenizerFast

            try:
                feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id, **hf_load_kwargs)
            except Exception:
                if is_offline:
                    raise
                feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")
            tokenizer = WhisperTokenizerFast.from_pretrained(
                model_id, extra_special_tokens={}, **hf_load_kwargs
            )
            self.processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

        model_kwargs: dict = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            **hf_load_kwargs,
        }
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation

        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                **model_kwargs,
            ).to(device)
        except TypeError:
            model_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                **model_kwargs,
            ).to(device)
        except Exception:
            from transformers import WhisperForConditionalGeneration

            try:
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    model_id,
                    **model_kwargs,
                ).to(device)
            except TypeError:
                model_kwargs.pop("attn_implementation", None)
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    model_id,
                    **model_kwargs,
                ).to(device)

        self.model.eval()
        self.adapter_id = adapter_id
        if adapter_id:
            self.load_adapter(adapter_id)

    def load_adapter(self, adapter_id: str | None) -> None:
        """Loads and merges a PEFT adapter into the base Whisper model."""
        if not adapter_id:
            return
        try:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_id)
            if hasattr(self.model, "merge_and_unload"):
                self.model = self.model.merge_and_unload()
            self.model.eval()
            self.adapter_id = adapter_id
        except Exception:
            self.adapter_id = adapter_id

    def _trim_waveform_silence(
        self,
        waveform: np.ndarray,
        sr: int = 16000,
        frame_ms: float = 25.0,
        hop_ms: float = 10.0,
        db_threshold: float = -35.0,
        margin_ms: float = 170.0,
    ) -> np.ndarray:
        return trim_waveform_silence(
            waveform,
            sr=sr,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            db_threshold=getattr(self, "vad_db_threshold", db_threshold),
            margin_ms=getattr(self, "vad_margin_ms", margin_ms),
        )

    def _preprocess_waveform(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        return preprocess_waveform(
            waveform,
            sr=sr,
            trim_silence=getattr(self, "trim_silence", True),
            vad_db_threshold=getattr(self, "vad_db_threshold", -35.0),
            vad_margin_ms=getattr(self, "vad_margin_ms", 170.0),
            peak_normalize=getattr(self, "peak_normalize", True),
            pre_roll_pad_ms=getattr(self, "pre_roll_pad_ms", 20.0),
            post_roll_pad_ms=getattr(self, "post_roll_pad_ms", 60.0),
        )

    def _pad_waveform(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Alias for backward compatibility with external callers and unit tests."""
        return self._preprocess_waveform(waveform, sr)

    def _build_gen_kwargs(self, extra_context: str | None = None) -> dict:
        import torch

        gen_kwargs: dict = {
            "language": self.language,
            "task": self.task,
            "do_sample": False,
            "max_new_tokens": 128,
            "return_timestamps": False,
        }

        if getattr(self, "repetition_penalty", 1.0) and self.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = float(self.repetition_penalty)
        cache_impl = getattr(self, "cache_implementation", None)
        if cache_impl:
            gen_kwargs["cache_implementation"] = cache_impl

        prompt_text = self.initial_prompt
        if extra_context:
            prompt_text = f"{prompt_text} {extra_context}".strip() if prompt_text else extra_context

        if prompt_text and hasattr(self.processor, "get_prompt_ids"):
            cache = getattr(self, "_prompt_ids_cache", None)
            if cache is None:
                cache = {}
                self._prompt_ids_cache = cache
            if prompt_text in cache:
                p_ids = cache[prompt_text]
            else:
                try:
                    p_ids = self.processor.get_prompt_ids(prompt_text)
                    if not isinstance(p_ids, torch.Tensor):
                        p_ids = torch.tensor(p_ids, device=self.device)
                    else:
                        p_ids = p_ids.to(self.device)
                    max_target = int(
                        getattr(getattr(getattr(self, "model", None), "config", None), "max_target_positions", 448)
                        or 448
                    )
                    max_prompt_tokens = max(0, max_target - int(gen_kwargs["max_new_tokens"]) - 16)
                    if max_prompt_tokens == 0:
                        p_ids = None
                    elif p_ids.shape[-1] > max_prompt_tokens:
                        p_ids = p_ids[..., :max_prompt_tokens]
                except Exception:
                    p_ids = None
                if len(cache) < 32:
                    cache[prompt_text] = p_ids
            if p_ids is not None:
                gen_kwargs["prompt_ids"] = p_ids

        return gen_kwargs

    def _maybe_release_cuda_cache(self, threshold_bytes: int = 256 * 1024 * 1024) -> None:
        import torch

        if str(getattr(self, "device", "cpu")).startswith("cuda") and torch.cuda.is_available():
            if (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) > threshold_bytes:
                torch.cuda.empty_cache()

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        extra_context: str | None = None,
        num_beams_nbest: int = 1,
        diverse_nbest: bool = True,
    ) -> ASRResult:
        import torch

        t0 = time.perf_counter()
        waveform, sr = load_wav_16k(audio_path, target_sr=16000)
        waveform = self._pad_waveform(waveform, sr)

        inputs = self.processor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device, dtype=self.torch_dtype)

        gen_kwargs = self._build_gen_kwargs(extra_context)

        with torch.inference_mode():
            predicted_ids = self.model.generate(input_features, **gen_kwargs)

        raw_text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        clean_text = _sanitize_hyp(raw_text, enable_itn=self.enable_itn)
        del predicted_ids

        nbest: list[str] = []
        nbest_scores: list[float] = []
        if num_beams_nbest > 1:
            nbest_batch, scores_batch = self._decode_nbest_hypotheses(
                input_features,
                gen_kwargs,
                num_beams_nbest=num_beams_nbest,
                diverse_nbest=diverse_nbest,
            )
            nbest = nbest_batch[0] if nbest_batch else []
            nbest_scores = scores_batch[0] if scores_batch else []
        del input_features
        self._maybe_release_cuda_cache(threshold_bytes=64 * 1024 * 1024)

        return ASRResult(
            text=clean_text,
            raw_text=raw_text,
            latency_s=time.perf_counter() - t0,
            nbest=nbest,
            nbest_scores=nbest_scores,
        )

    def _decode_nbest_hypotheses(
        self,
        features: object,
        gen_kwargs: dict[str, object],
        *,
        num_beams_nbest: int,
        diverse_nbest: bool = False,
    ) -> tuple[list[list[str]], list[list[float]]]:
        """Decode N-best hypotheses and sequence scores while preserving 1-best greedy."""
        import torch

        batch_len = int(features.shape[0]) if hasattr(features, "shape") else len(features)  # type: ignore[arg-type]
        beam_kwargs = dict(gen_kwargs)
        beam_kwargs.pop("cache_implementation", None)
        beam_kwargs.update(
            {
                "num_beams": num_beams_nbest,
                "num_return_sequences": num_beams_nbest,
                "do_sample": False,
                "return_dict_in_generate": True,
                "output_scores": True,
            }
        )
        if diverse_nbest and num_beams_nbest >= 2 and getattr(self, "_group_beam_supported", True):
            diverse_kwargs = {
                **beam_kwargs,
                "num_beam_groups": num_beams_nbest,
                "diversity_penalty": 0.5,
            }
            try:
                with torch.inference_mode():
                    beam_out = self.model.generate(features, **diverse_kwargs)
            except (TypeError, ValueError, RuntimeError):
                self._group_beam_supported = False
                with torch.inference_mode():
                    beam_out = self.model.generate(features, **beam_kwargs)
        else:
            with torch.inference_mode():
                beam_out = self.model.generate(features, **beam_kwargs)

        if hasattr(beam_out, "sequences"):
            seqs = beam_out.sequences
            raw_scores = getattr(beam_out, "sequences_scores", None)
        elif isinstance(beam_out, dict) and "sequences" in beam_out:
            seqs = beam_out["sequences"]
            raw_scores = beam_out.get("sequences_scores")
        else:
            seqs = beam_out
            raw_scores = None

        tok = getattr(self.processor, "tokenizer", self.processor)
        sot_id: int | None = None
        if hasattr(tok, "convert_tokens_to_ids"):
            try:
                cand = tok.convert_tokens_to_ids("<|startoftranscript|>")
                unk_id = getattr(tok, "unk_token_id", None)
                if isinstance(cand, int) and cand >= 0 and cand != unk_id:
                    sot_id = cand
            except Exception:
                sot_id = None

        prompt_list: list[int] | None = None
        p_ids = gen_kwargs.get("prompt_ids")
        if p_ids is not None:
            prompt_list = p_ids.tolist() if hasattr(p_ids, "tolist") else list(p_ids)

        cleaned_rows: list[list[int]] = []
        any_stripped = False
        for row in seqs:
            row_list = [int(x) for x in (row.tolist() if hasattr(row, "tolist") else row)]
            cut = 0
            if sot_id is not None and sot_id in row_list:
                cut = row_list.index(sot_id)
            elif prompt_list:
                plen = len(prompt_list)
                if len(row_list) >= plen and row_list[:plen] == prompt_list:
                    cut = plen
                elif len(row_list) > plen and row_list[1 : 1 + plen] == prompt_list:
                    cut = 1 + plen
            if cut > 0:
                any_stripped = True
                cleaned_rows.append(row_list[cut:])
            else:
                cleaned_rows.append(row_list)

        decode_input = cleaned_rows if any_stripped else seqs
        raw_hyps = self.processor.batch_decode(decode_input, skip_special_tokens=True)
        clean_hyps = [_sanitize_hyp(h, enable_itn=self.enable_itn) for h in raw_hyps]

        scores_flat: list[float] | None = None
        if raw_scores is not None:
            raw_list = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)
            scores_flat = [float(s) for s in raw_list]

        per_item_ret = max(1, len(clean_hyps) // max(1, batch_len))
        nbest_out: list[list[str]] = []
        scores_out: list[list[float]] = []
        for idx in range(batch_len):
            start = idx * per_item_ret
            end = start + per_item_ret
            nbest_out.append(clean_hyps[start:end])
            scores_out.append(scores_flat[start:end] if scores_flat is not None else [])
        del beam_out, seqs, raw_scores
        self._maybe_release_cuda_cache()
        return nbest_out, scores_out

    def transcribe_batch(
        self,
        audio_paths: list[str | Path],
        *,
        batch_size: int = 8,
        extra_contexts: list[str | None] | None = None,
        num_beams_nbest: int = 1,
        diverse_nbest: bool = True,
    ) -> list[ASRResult]:
        """Transcribe a batch of audio files with accelerated batched decoding and ITN."""
        import torch

        if not audio_paths:
            return []

        if extra_contexts is not None and len(extra_contexts) != len(audio_paths):
            raise ValueError(
                f"extra_contexts length ({len(extra_contexts)}) must match audio_paths length ({len(audio_paths)})"
            )

        results: list[ASRResult] = []

        for i in range(0, len(audio_paths), batch_size):
            chunk_paths = audio_paths[i : i + batch_size]
            chunk_contexts = (
                extra_contexts[i : i + batch_size] if extra_contexts is not None else [None] * len(chunk_paths)
            )

            t0 = time.perf_counter()
            waveforms = []
            for p in chunk_paths:
                try:
                    w, sr = load_wav_16k(p, target_sr=16000)
                    w = self._pad_waveform(w, sr)
                    waveforms.append(w)
                except Exception:
                    waveforms.append(None)

            active_waveforms = [w for w in waveforms if w is not None]
            if not active_waveforms:
                for _ in chunk_paths:
                    results.append(ASRResult(text="", raw_text="", latency_s=0.0))
                continue

            inputs = self.processor(
                active_waveforms,
                sampling_rate=16000,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(self.device, dtype=self.torch_dtype)

            from collections import defaultdict

            # Group active chunk items by context so different contexts are never dropped
            ctx_groups: dict[str | None, list[int]] = defaultdict(list)
            for active_idx, ctx_val in enumerate(chunk_contexts):
                if waveforms[active_idx] is not None:
                    ctx_groups[ctx_val].append(active_idx)

            chunk_raw_texts: list[str | None] = [None] * len(waveforms)
            chunk_nbests: list[list[str]] = [[] for _ in waveforms]
            chunk_nbest_scores: list[list[float]] = [[] for _ in waveforms]
            for ctx_val, indices in ctx_groups.items():
                sub_features = input_features[indices]
                gen_kwargs = self._build_gen_kwargs(ctx_val)
                with torch.inference_mode():
                    pred_sub = self.model.generate(sub_features, **gen_kwargs)
                sub_texts = self.processor.batch_decode(pred_sub, skip_special_tokens=True)
                del pred_sub
                for local_pos, global_idx in enumerate(indices):
                    chunk_raw_texts[global_idx] = sub_texts[local_pos]
                if num_beams_nbest > 1:
                    nb_texts, nb_scores = self._decode_nbest_hypotheses(
                        sub_features,
                        gen_kwargs,
                        num_beams_nbest=num_beams_nbest,
                        diverse_nbest=diverse_nbest,
                    )
                    for local_pos, global_idx in enumerate(indices):
                        if local_pos < len(nb_texts):
                            chunk_nbests[global_idx] = nb_texts[local_pos]
                        if local_pos < len(nb_scores):
                            chunk_nbest_scores[global_idx] = nb_scores[local_pos]
                del sub_features

            elapsed = time.perf_counter() - t0
            avg_lat = elapsed / len(chunk_paths)
            del input_features
            self._maybe_release_cuda_cache(threshold_bytes=64 * 1024 * 1024)

            for w, raw_text, nb_list, nb_sc in zip(
                waveforms, chunk_raw_texts, chunk_nbests, chunk_nbest_scores
            ):
                if w is None or raw_text is None:
                    results.append(ASRResult(text="", raw_text="", latency_s=avg_lat))
                    continue
                clean_text = _sanitize_hyp(raw_text, enable_itn=self.enable_itn)
                results.append(
                    ASRResult(
                        text=clean_text,
                        raw_text=raw_text,
                        latency_s=avg_lat,
                        nbest=nb_list,
                        nbest_scores=nb_sc,
                    )
                )

        return results
