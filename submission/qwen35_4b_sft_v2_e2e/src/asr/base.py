"""ASR engine protocol, tag cleaner, and deterministic fake/oracle engine."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

_ASR_PREFIX_RE = re.compile(r"^(?:language\s+[^<]*<asr_text>|<\|[^|]+\|>)+", re.IGNORECASE)
_SPECIAL_TAG_RE = re.compile(r"<\|[^|]+\|>|</?asr_text>", re.IGNORECASE)


@dataclass
class ASRResult:
    text: str
    raw_text: str = ""
    latency_s: float = 0.0
    nbest: list[str] = field(default_factory=list)
    nbest_scores: list[float] = field(default_factory=list)


class ASREngine(Protocol):
    def transcribe(self, audio_path: str | Path) -> ASRResult: ...


def strip_asr_tags(raw: str) -> str:
    """Strip Qwen3-ASR 'language Vietnamese<asr_text>' prefix and special tokens."""
    cleaned = _ASR_PREFIX_RE.sub("", raw.strip())
    cleaned = _SPECIAL_TAG_RE.sub("", cleaned)
    cleaned = unicodedata.normalize("NFC", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


class FakeASREngine:
    """Deterministic lookup-based ASR engine for unit testing."""

    def __init__(
        self,
        mapping: dict[str, str] | None = None,
        default_text: str = "",
        enable_itn: bool = False,
    ):
        self.mapping = dict(mapping or {})
        self.default_text = default_text
        self.enable_itn = enable_itn

    def transcribe(self, audio_path: str | Path) -> ASRResult:
        key = Path(audio_path).name
        raw = self.mapping.get(str(audio_path), self.mapping.get(key, self.default_text))
        text = strip_asr_tags(raw)
        if self.enable_itn:
            from src.asr.itn import inverse_text_normalize

            text = inverse_text_normalize(text)
        return ASRResult(text=text, raw_text=raw, latency_s=0.0)
