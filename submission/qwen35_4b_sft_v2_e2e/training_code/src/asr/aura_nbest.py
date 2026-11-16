"""AURA (IEEE ASRU 2025) Uncertainty-Aware Phonetic Diff extractor for N-best ASR hypotheses."""
from __future__ import annotations

import difflib
import unicodedata
from typing import Sequence


def _norm_token(w: str) -> str:
    cleaned = "".join(c for c in unicodedata.normalize("NFC", w).lower() if c.isalnum())
    return cleaned


def extract_phonetic_diff(
    beam_texts: Sequence[str],
    beam_scores: Sequence[float] | None = None,
    max_score_delta: float = 0.65,
) -> str | None:
    """Extract phonetically ambiguous content word differences between Beam 1 and Beam 2.

    Returns a compact string hint when uncertainty is high and words differ, or None.
    """
    if len(beam_texts) < 2:
        return None

    b1_raw = (beam_texts[0] or "").strip()
    b2_raw = (beam_texts[1] or "").strip()
    if not b1_raw or not b2_raw:
        return None

    if beam_scores is not None and len(beam_scores) >= 2:
        # If beam scores are available (e.g. logprobs where scores are negative or probabilities),
        # check score delta
        delta = abs(beam_scores[0] - beam_scores[1])
        if delta > max_score_delta:
            return None

    tokens1 = b1_raw.split()
    tokens2 = b2_raw.split()

    norm1 = [_norm_token(t) for t in tokens1]
    norm2 = [_norm_token(t) for t in tokens2]

    if norm1 == norm2:
        # Differences are only casing or punctuation
        return None

    matcher = difflib.SequenceMatcher(None, norm1, norm2)
    diff_spans1: list[str] = []
    diff_spans2: list[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete", "insert"):
            span1 = " ".join(tokens1[i1:i2])
            span2 = " ".join(tokens2[j1:j2])
            if span1:
                diff_spans1.append(span1)
            if span2:
                diff_spans2.append(span2)

    if not diff_spans1 and not diff_spans2:
        return None

    str1 = " ... ".join(diff_spans1)
    str2 = " ... ".join(diff_spans2)
    return f'[Gợi ý đồng âm ASR: "{str1}" <-> "{str2}"]'
