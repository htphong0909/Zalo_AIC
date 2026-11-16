"""Tests for AURA phonetic diff extraction."""
from src.asr.aura_nbest import extract_phonetic_diff


def test_extract_phonetic_diff_detects_real_homophone_difference():
    beam_texts = [
        "Nếu đủ tiền trả cả hai môn tiếng Nhật lẫn hàng thì đóng luôn",
        "Nếu đủ tiền trả cả hai môn tiếng Nhật lẫn Hàn thì đóng luôn",
    ]
    diff = extract_phonetic_diff(beam_texts)
    assert diff is not None
    assert "hàng" in diff and "Hàn" in diff
    assert "Gợi ý đồng âm ASR" in diff


def test_extract_phonetic_diff_ignores_punctuation_and_casing():
    beam_texts = [
        "bật đèn phòng khách nhé",
        "Bật đèn phòng khách nhé.",
    ]
    diff = extract_phonetic_diff(beam_texts)
    assert diff is None


def test_extract_phonetic_diff_respects_score_threshold():
    beam_texts = [
        "tắt bếp giùm anh",
        "bắt bếp giùm anh",
    ]
    # Score delta is very large (0.9 > max_delta 0.65) -> confidence is high on beam 1, suppress diff
    diff_suppressed = extract_phonetic_diff(beam_texts, beam_scores=[-0.1, -1.0], max_score_delta=0.65)
    assert diff_suppressed is None

    # Score delta is small (0.2 <= 0.65) -> uncertainty is high, emit diff
    diff_active = extract_phonetic_diff(beam_texts, beam_scores=[-0.4, -0.6], max_score_delta=0.65)
    assert diff_active is not None
    assert "tắt" in diff_active and "bắt" in diff_active
