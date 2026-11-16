"""ChunkFormer CTC ASR engine supporting khanhld/chunkformer-ctc-large-vie."""
from __future__ import annotations

import time
import unicodedata
from pathlib import Path

from src.asr.base import ASRResult, strip_asr_tags


class ChunkFormerASREngine:
    """Fast, lightweight CTC ASR engine based on Masked Chunking Conformer."""

    def __init__(
        self,
        model_id: str = "khanhld/chunkformer-ctc-large-vie",
        *,
        device: str = "cuda",
        chunk_size: int = 64,
        left_context_size: int = 128,
        right_context_size: int = 128,
    ):
        from chunkformer import ChunkFormerModel

        self.model_id = model_id
        self.device = device
        self.chunk_size = chunk_size
        self.left_context_size = left_context_size
        self.right_context_size = right_context_size

        self.model = ChunkFormerModel.from_pretrained(model_id)

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        extra_context: str | None = None,
    ) -> ASRResult:
        t0 = time.perf_counter()
        raw_results = self.model.batch_decode(
            audio_paths=[str(audio_path)],
            chunk_size=self.chunk_size,
            left_context_size=self.left_context_size,
            right_context_size=self.right_context_size,
        )

        raw_res = raw_results[0] if raw_results else ""
        if isinstance(raw_res, str):
            raw_text = raw_res
        elif isinstance(raw_res, list):
            raw_text = " ".join(
                item if isinstance(item, str) else str(item.get("text", ""))
                for item in raw_res
            )
        elif isinstance(raw_res, dict):
            raw_text = str(raw_res.get("text", ""))
        else:
            raw_text = str(raw_res)
        clean_text = strip_asr_tags(raw_text)
        clean_text = unicodedata.normalize("NFC", clean_text).strip()

        return ASRResult(
            text=clean_text,
            raw_text=raw_text,
            latency_s=time.perf_counter() - t0,
        )

    def batch_transcribe(
        self,
        audio_paths: list[str | Path],
    ) -> list[ASRResult]:
        t0 = time.perf_counter()
        paths_str = [str(p) for p in audio_paths]
        raw_results = self.model.batch_decode(
            audio_paths=paths_str,
            chunk_size=self.chunk_size,
            left_context_size=self.left_context_size,
            right_context_size=self.right_context_size,
        )
        total_time = time.perf_counter() - t0
        avg_time = total_time / max(1, len(paths_str))

        results = []
        for raw_res in raw_results:
            if isinstance(raw_res, str):
                raw_text = raw_res
            elif isinstance(raw_res, list):
                raw_text = " ".join(
                    item if isinstance(item, str) else str(item.get("text", ""))
                    for item in raw_res
                )
            elif isinstance(raw_res, dict):
                raw_text = str(raw_res.get("text", ""))
            else:
                raw_text = str(raw_res)
            clean_text = strip_asr_tags(raw_text)
            clean_text = unicodedata.normalize("NFC", clean_text).strip()
            results.append(ASRResult(text=clean_text, raw_text=raw_text, latency_s=avg_time))
        return results
