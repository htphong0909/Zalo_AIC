"""Vietnamese ASR evaluation metrics: WER, CER, slot/number/loanword recall, and dialect breakdown."""
from __future__ import annotations

import re
import unicodedata
from statistics import mean
from typing import Any

from src.core.schemas import Sample

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_NUM_TIME_WORDS = {
    "không",
    "một",
    "mốt",
    "hai",
    "ba",
    "bốn",
    "tư",
    "năm",
    "lăm",
    "sáu",
    "bảy",
    "tám",
    "chín",
    "mười",
    "mươi",
    "chục",
    "trăm",
    "nghìn",
    "ngàn",
    "triệu",
    "tỷ",
    "rưỡi",
    "giờ",
    "phút",
    "ngày",
    "tháng",
    "thứ",
}
_DIALECT_WORDS = {
    "rứa",
    "mô",
    "ni",
    "tê",
    "chừ",
    "mạ",
    "nghen",
    "dùm",
    "giùm",
    "tui",
    "ảnh",
    "chỉ",
    "ổng",
    "bả",
    "hổm",
    "bữa",
    "coi",
    "nghe",
}


def normalize_vi_text(text: str) -> str:
    s = unicodedata.normalize("NFC", str(text or "")).lower()
    s = _PUNCT_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def _edit_distance(seq_a: list[Any] | str, seq_b: list[Any] | str) -> int:
    n, m = len(seq_a), len(seq_b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ai = seq_a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == seq_b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


def compute_wer(pred: str, ref: str) -> tuple[int, int, float]:
    p_words = normalize_vi_text(pred).split()
    r_words = normalize_vi_text(ref).split()
    if not r_words:
        err = len(p_words)
        return err, 0, float(bool(p_words))
    dist = _edit_distance(p_words, r_words)
    return dist, len(r_words), dist / len(r_words)


def compute_cer(pred: str, ref: str) -> tuple[int, int, float]:
    p_chars = normalize_vi_text(pred).replace(" ", "")
    r_chars = normalize_vi_text(ref).replace(" ", "")
    if not r_chars:
        err = len(p_chars)
        return err, 0, float(bool(p_chars))
    dist = _edit_distance(p_chars, r_chars)
    return dist, len(r_chars), dist / len(r_chars)


def _is_ascii_loanword(word: str) -> bool:
    return word.isascii() and word.isalpha() and len(word) >= 3


def _extract_number_tokens(words: list[str]) -> list[str]:
    return [w for w in words if any(ch.isdigit() for ch in w) or w in _NUM_TIME_WORDS]


def _extract_gold_arg_tokens(sample: Sample) -> set[str]:
    tokens: set[str] = set()
    if not sample.label:
        return tokens
    ref_words = set(normalize_vi_text(sample.audio_text or "").split())
    for c in sample.label.tool_calls:
        for v in c.arguments.values():
            if isinstance(v, str):
                for w in normalize_vi_text(v).split():
                    if w in ref_words and len(w) >= 2:
                        tokens.add(w)
    return tokens


def evaluate_transcripts(
    transcripts: dict[str, dict[str, Any]],
    samples: list[Sample],
    *,
    top_k_errors: int = 15,
) -> dict[str, Any]:
    total_w_err = 0
    total_w_ref = 0
    total_c_err = 0
    total_c_ref = 0
    hist_w_err = 0
    hist_w_ref = 0
    sample_wers: list[float] = []
    sample_cers: list[float] = []
    dialect_wers: list[float] = []
    standard_wers: list[float] = []
    latencies: list[float] = []
    exact_matches = 0
    n_evaluated = 0

    num_hits = 0
    num_total = 0
    loan_hits = 0
    loan_total = 0
    slot_hits = 0
    slot_total = 0

    sample_errors: list[dict[str, Any]] = []

    for s in samples:
        if s.audio_text is None:
            continue
        row = transcripts.get(s.id, {})
        pred_text = str(row.get("audio_text", ""))
        w_err, w_ref, wer = compute_wer(pred_text, s.audio_text)
        c_err, c_ref, cer = compute_cer(pred_text, s.audio_text)
        total_w_err += w_err
        total_w_ref += w_ref
        total_c_err += c_err
        total_c_ref += c_ref
        sample_wers.append(wer)
        sample_cers.append(cer)

        p_words = set(normalize_vi_text(pred_text).split())
        r_words_list = normalize_vi_text(s.audio_text).split()

        if any(w in _DIALECT_WORDS for w in r_words_list):
            dialect_wers.append(wer)
        else:
            standard_wers.append(wer)

        for nw in _extract_number_tokens(r_words_list):
            num_total += 1
            if nw in p_words:
                num_hits += 1

        for lw in set(w for w in r_words_list if _is_ascii_loanword(w)):
            loan_total += 1
            if lw in p_words:
                loan_hits += 1

        for sw in _extract_gold_arg_tokens(s):
            slot_total += 1
            if sw in p_words:
                slot_hits += 1

        # History audio turns evaluation
        pred_hist = row.get("history_texts") or {}
        for i, t in enumerate(s.history):
            if t.role == "user" and t.audio_text:
                hp = str(pred_hist.get(str(i), pred_hist.get(i, "")))
                hw_err, hw_ref, _ = compute_wer(hp, t.audio_text)
                hist_w_err += hw_err
                hist_w_ref += hw_ref

        if normalize_vi_text(pred_text) == normalize_vi_text(s.audio_text):
            exact_matches += 1
        elif wer > 0:
            sample_errors.append(
                {
                    "id": s.id,
                    "wer": round(wer, 4),
                    "pred": pred_text,
                    "ref": s.audio_text,
                }
            )
        if "latency_s" in row:
            latencies.append(float(row["latency_s"]))
        n_evaluated += 1

    sample_errors.sort(key=lambda x: x["wer"], reverse=True)

    return {
        "n": n_evaluated,
        "corpus_wer": total_w_err / max(1, total_w_ref),
        "corpus_cer": total_c_err / max(1, total_c_ref),
        "macro_wer": mean(sample_wers) if sample_wers else 0.0,
        "macro_cer": mean(sample_cers) if sample_cers else 0.0,
        "history_corpus_wer": hist_w_err / max(1, hist_w_ref) if hist_w_ref else 0.0,
        "normalized_exact_match": exact_matches / max(1, n_evaluated),
        "slot_token_recall": slot_hits / max(1, slot_total) if slot_total else 1.0,
        "number_time_token_recall": num_hits / max(1, num_total) if num_total else 1.0,
        "loanword_recall": loan_hits / max(1, loan_total) if loan_total else 1.0,
        "by_accent_subset": {
            "dialect_marked": {
                "n": len(dialect_wers),
                "macro_wer": mean(dialect_wers) if dialect_wers else 0.0,
            },
            "standard": {
                "n": len(standard_wers),
                "macro_wer": mean(standard_wers) if standard_wers else 0.0,
            },
        },
        "avg_latency_s": mean(latencies) if latencies else 0.0,
        "worst_samples": sample_errors[:top_k_errors],
    }
