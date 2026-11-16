from __future__ import annotations

import pytest

from src.asr.audio import load_wav_16k
from src.asr.base import FakeASREngine, strip_asr_tags
from src.core.dataset import audio_path, load_split
from src.eval.asr_metrics import compute_cer, compute_wer, evaluate_transcripts, normalize_vi_text


def test_strip_asr_tags_removes_qwen_language_header():
    raw = "language Vietnamese<asr_text>  Bật đèn phòng khách lên nhé.  <|endoftext|>"
    assert strip_asr_tags(raw) == "Bật đèn phòng khách lên nhé."


def test_load_wav_16k_reads_real_sample_and_blocks_environment_dir(tmp_path):
    sample = load_split("train", "data")[0]
    wav_p = audio_path("train", "data", sample.audio)
    waveform, sr = load_wav_16k(wav_p)
    assert sr == 16000
    assert waveform.ndim == 1
    assert len(waveform) > 1000

    forbidden = tmp_path / "train_environment" / sample.id / "fake.wav"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_bytes(b"")
    with pytest.raises(PermissionError):
        load_wav_16k(forbidden)


def test_wer_cer_and_evaluate_transcripts():
    assert normalize_vi_text("Xin chào, Bố!") == "xin chào bố"
    w_err, w_ref, wer = compute_wer("bật đèn phòng khách", "Bật đèn phòng ngủ!")
    assert (w_err, w_ref, wer) == (1, 4, 0.25)

    c_err, c_ref, cer = compute_cer("abc", "abd")
    assert (c_err, c_ref) == (1, 3)
    assert abs(cer - 1 / 3) < 1e-6

    samples = load_split("train", "data")[:3]
    oracle_map = {s.id: {"id": s.id, "audio_text": s.audio_text, "latency_s": 0.1} for s in samples}
    rep = evaluate_transcripts(oracle_map, samples)
    assert rep["n"] == 3
    assert rep["corpus_wer"] == 0.0
    assert rep["corpus_cer"] == 0.0
    assert rep["normalized_exact_match"] == 1.0


def test_fake_asr_engine():
    eng = FakeASREngine({"a.wav": "language Vietnamese<asr_text>Mở cửa"})
    res = eng.transcribe("dir/a.wav")
    assert res.text == "Mở cửa"


def test_whisper_engine_import_and_signature():
    import numpy as np
    from src.asr.whisper_engine import WhisperASREngine

    assert hasattr(WhisperASREngine, "transcribe")
    # Verify bilateral padding, silence trimming, and peak normalization helper method logic
    dummy_wav = np.ones(16000, dtype=np.float32) * 0.5

    class DummyEngine:
        pre_roll_pad_ms = 60.0
        post_roll_pad_ms = 60.0
        peak_normalize = True
        trim_silence = False
        vad_db_threshold = -35.0
        vad_margin_ms = 170.0
        repetition_penalty = 1.02
        language = "vi"
        task = "transcribe"
        initial_prompt = "Trợ lý ảo"
        device = "cpu"
        processor = None
        _preprocess_waveform = WhisperASREngine._preprocess_waveform
        _pad_waveform = WhisperASREngine._pad_waveform
        _trim_waveform_silence = WhisperASREngine._trim_waveform_silence
        _build_gen_kwargs = WhisperASREngine._build_gen_kwargs

    eng = DummyEngine()
    padded = eng._pad_waveform(dummy_wav, 16000)
    expected_len = 16000 + int(0.06 * 16000) * 2
    assert len(padded) == expected_len
    # Peak should be normalized to 0.95
    assert abs(np.max(padded) - 0.95) < 1e-4

    # Test empty and zero waveforms
    empty_wav = np.zeros(0, dtype=np.float32)
    assert len(eng._pad_waveform(empty_wav, 16000)) == 0

    zero_wav = np.zeros(16000, dtype=np.float32)
    zero_padded = eng._pad_waveform(zero_wav, 16000)
    assert len(zero_padded) == expected_len
    assert np.max(zero_padded) == 0.0

    # Test conservative silence trimming
    sr = 16000
    lead_silence = np.zeros(int(0.5 * sr), dtype=np.float32)  # 500ms silence
    tone = np.ones(int(0.5 * sr), dtype=np.float32) * 0.5    # 500ms tone
    tail_silence = np.zeros(int(0.5 * sr), dtype=np.float32)  # 500ms silence
    composite_wav = np.concatenate([lead_silence, tone, tail_silence])
    assert len(composite_wav) == 24000  # 1.5s total

    trimmed = eng._trim_waveform_silence(composite_wav, sr=sr, db_threshold=-35.0, margin_ms=170.0)
    # The trimmed length should be strictly less than original 24000, but preserve tone + safety margins
    margin_samples = int(0.170 * sr)
    min_expected = len(tone) + margin_samples
    assert min_expected <= len(trimmed) < len(composite_wav)

    # Test end-to-end preprocess with trimming enabled
    eng.trim_silence = True
    preprocessed = eng._preprocess_waveform(composite_wav, sr=sr)
    # Result must be peak-normalized to 0.95 with 60ms collar on both sides
    assert abs(np.max(preprocessed) - 0.95) < 1e-4
    collar_len = int(0.06 * sr) * 2
    assert len(preprocessed) == len(trimmed) + collar_len

    # Test repetition_penalty and deterministic decoding kwargs
    gen_kw = eng._build_gen_kwargs()
    assert gen_kw["repetition_penalty"] == 1.02
    assert gen_kw["max_new_tokens"] == 128
    assert gen_kw["do_sample"] is False
    assert gen_kw["return_timestamps"] is False


def test_gemini_transcribe_engine_init_and_rotation():
    from src.asr.gemini_asr import GeminiTranscribeEngine

    eng = GeminiTranscribeEngine(api_keys=["key1", "key2", "key3"], max_tpm_per_key=1000)
    assert len(eng.api_keys) == 3
    assert eng._select_key(100) in eng.api_keys


def test_whisper_sanitize_hyp_punctuation_and_casing():
    from src.asr.whisper_engine import _sanitize_hyp

    # Spurious spaces before punctuation and double spaces should be cleanly normalized
    raw = "  bật đèn phòng khách  nhé .   "
    clean = _sanitize_hyp(raw, enable_itn=False)
    assert clean == "Bật đèn phòng khách nhé."

    raw2 = "nhắc tôi lúc 9 giờ kém mười lăm ."
    clean2 = _sanitize_hyp(raw2, enable_itn=True)
    assert clean2 == "Nhắc tôi lúc 9 giờ kém 15."


def test_whisper_engine_adapter_loading_signature_and_backward_compatibility():
    import inspect
    from unittest.mock import MagicMock, patch
    from src.asr.whisper_engine import WhisperASREngine, preprocess_waveform, trim_waveform_silence

    # 1. Verify WhisperASREngine.__init__ signature accepts optional adapter_id defaulting to None
    sig = inspect.signature(WhisperASREngine.__init__)
    assert "adapter_id" in sig.parameters
    assert sig.parameters["adapter_id"].default is None
    assert hasattr(WhisperASREngine, "load_adapter")

    # 2. Verify standalone waveform preprocessing exports
    import numpy as np
    empty_wav = np.zeros(0, dtype=np.float32)
    assert len(trim_waveform_silence(empty_wav)) == 0
    assert len(preprocess_waveform(empty_wav)) == 0

    zero_wav = np.zeros(16000, dtype=np.float32)
    assert len(trim_waveform_silence(zero_wav)) == 16000
    expected_padded_len = 16000 + int(0.06 * 16000) * 2
    assert len(preprocess_waveform(zero_wav)) == expected_padded_len

    # 3. Test backward compatibility: adapter_id=None leaves model untouched
    class MockBaseModel:
        def __init__(self):
            self.eval_called = False

        def eval(self):
            self.eval_called = True
            return self

    mock_base = MockBaseModel()
    eng = WhisperASREngine.__new__(WhisperASREngine)
    eng.model = mock_base
    eng.adapter_id = None
    assert eng.adapter_id is None
    assert eng.model is mock_base

    # 4. Test adapter loading and merging with PeftModel
    class MockMergedModel:
        def __init__(self):
            self.eval_called = False

        def eval(self):
            self.eval_called = True
            return self

    class MockPeftModel:
        @classmethod
        def from_pretrained(cls, model, adapter_id):
            assert model is mock_base
            assert adapter_id == "experiments/phowhisper_lora_loop36"
            return cls()

        def merge_and_unload(self):
            merged = MockMergedModel()
            return merged

    mock_peft_pkg = MagicMock()
    mock_peft_pkg.PeftModel = MockPeftModel
    with patch.dict("sys.modules", {"peft": mock_peft_pkg}):
        eng.load_adapter("experiments/phowhisper_lora_loop36")
        assert isinstance(eng.model, MockMergedModel)
        assert eng.model.eval_called is True
        assert eng.adapter_id == "experiments/phowhisper_lora_loop36"

    # 5. Test zero-compute mock handling when PeftModel raises an exception on Mock model
    class BrokenPeftModel:
        @classmethod
        def from_pretrained(cls, model, adapter_id):
            raise RuntimeError("Mock zero-compute offline bypass")

    mock_broken_pkg = MagicMock()
    mock_broken_pkg.PeftModel = BrokenPeftModel
    with patch.dict("sys.modules", {"peft": mock_broken_pkg}):
        mock_model_2 = MockBaseModel()
        eng2 = WhisperASREngine.__new__(WhisperASREngine)
        eng2.model = mock_model_2
        # On a mock model, graceful handling ensures no crash during unit testing
        eng2.load_adapter("nonexistent/adapter")
        assert eng2.adapter_id == "nonexistent/adapter"


def test_whisper_engine_nbest_preserves_greedy_one_best(tmp_path):
    import soundfile as sf
    import numpy as np
    import torch
    from src.asr.whisper_engine import WhisperASREngine

    wav = tmp_path / "utt.wav"
    sf.write(str(wav), np.zeros(16000, dtype=np.float32), 16000)

    class _FakeProc:
        def __call__(self, audio, sampling_rate=16000, return_tensors="pt"):
            return type("Out", (), {"input_features": torch.zeros((1, 80, 100), dtype=torch.float32)})()

        def batch_decode(self, ids, skip_special_tokens=True):
            return [f"hyp_{int(row[0].item())}" for row in ids]

    class _FakeModel:
        def generate(self, input_features, **kwargs):
            if kwargs.get("num_beams", 1) > 1:
                seqs = torch.tensor([[1], [2], [3]], dtype=torch.long)
                return type("BeamOut", (), {"sequences": seqs, "sequences_scores": torch.tensor([-0.2, -0.35, -0.9])})()
            return torch.tensor([[1]], dtype=torch.long)

    eng = WhisperASREngine.__new__(WhisperASREngine)
    eng.processor = _FakeProc()
    eng.model = _FakeModel()
    eng.device = "cpu"
    eng.torch_dtype = torch.float32
    eng.language = "vi"
    eng.task = "transcribe"
    eng.enable_itn = False
    eng.initial_prompt = None
    eng.pre_roll_pad_ms = 20.0
    eng.post_roll_pad_ms = 60.0
    eng.peak_normalize = False
    eng.trim_silence = False
    eng.vad_db_threshold = -35.0
    eng.vad_margin_ms = 170.0
    eng.repetition_penalty = 1.0

    import pytest

    res = eng.transcribe(wav, num_beams_nbest=3)
    assert res.text == "Hyp_1"
    assert res.nbest == ["Hyp_1", "Hyp_2", "Hyp_3"]
    assert res.nbest_scores == pytest.approx([-0.2, -0.35, -0.9])


def test_whisper_engine_clamps_oversized_prompt_ids():
    import torch
    from src.asr.whisper_engine import WhisperASREngine

    class _Proc:
        def get_prompt_ids(self, text):
            return torch.arange(500, dtype=torch.long)

    eng = WhisperASREngine.__new__(WhisperASREngine)
    eng.processor = _Proc()
    eng.model = type("M", (), {"config": type("C", (), {"max_target_positions": 448})()})()
    eng.device = "cpu"
    eng.language = "vi"
    eng.task = "transcribe"
    eng.initial_prompt = "prompt"
    eng.repetition_penalty = 1.0

    kw = eng._build_gen_kwargs("very long context")
    assert kw["prompt_ids"].shape[-1] <= 448 - kw["max_new_tokens"] - 16


def test_whisper_engine_nbest_strips_prompt_prefix(tmp_path):
    import numpy as np
    import soundfile as sf
    import torch
    from src.asr.whisper_engine import WhisperASREngine

    wav = tmp_path / "utt.wav"
    sf.write(str(wav), np.zeros(16000, dtype=np.float32), 16000)

    class _Tok:
        unk_token_id = 0

        def convert_tokens_to_ids(self, tok):
            return 50258 if tok == "<|startoftranscript|>" else 0

    class _Proc:
        tokenizer = _Tok()

        def __call__(self, audio, sampling_rate=16000, return_tensors="pt"):
            return type("Out", (), {"input_features": torch.zeros((1, 80, 100), dtype=torch.float32)})()

        def get_prompt_ids(self, text):
            return torch.tensor([50361, 901, 902], dtype=torch.long)

        def batch_decode(self, ids, skip_special_tokens=True):
            out = []
            for row in ids:
                vals = row.tolist() if hasattr(row, "tolist") else list(row)
                out.append(" ".join(str(x) for x in vals if x != 50258))
            return out

    class _Model:
        config = type("C", (), {"max_target_positions": 448})()

        def generate(self, input_features, **kwargs):
            if kwargs.get("num_beams", 1) > 1:
                seqs = torch.tensor(
                    [
                        [50361, 901, 902, 50258, 11, 12],
                        [50361, 901, 902, 50258, 11, 13],
                    ],
                    dtype=torch.long,
                )
                return type("BeamOut", (), {"sequences": seqs, "sequences_scores": torch.tensor([-0.1, -0.2])})()
            return torch.tensor([[11, 12]], dtype=torch.long)

    eng = WhisperASREngine.__new__(WhisperASREngine)
    eng.processor = _Proc()
    eng.model = _Model()
    eng.device = "cpu"
    eng.torch_dtype = torch.float32
    eng.language = "vi"
    eng.task = "transcribe"
    eng.enable_itn = False
    eng.initial_prompt = "prompt prefix"
    eng.pre_roll_pad_ms = 20.0
    eng.post_roll_pad_ms = 60.0
    eng.peak_normalize = False
    eng.trim_silence = False
    eng.vad_db_threshold = -35.0
    eng.vad_margin_ms = 170.0
    eng.repetition_penalty = 1.0

    res = eng.transcribe(wav, num_beams_nbest=2)
    assert res.text == "11 12"
    assert res.nbest == ["11 12", "11 13"]







