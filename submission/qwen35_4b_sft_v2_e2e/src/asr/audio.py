"""Safe 16kHz mono WAV loader with environment-directory access guard."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def _guard_path(path: Path) -> None:
    if any(part.endswith("_environment") for part in path.parts):
        raise PermissionError(f"Refusing to read inside environment directory: {path}")


def load_wav_16k(path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load a WAV file as 1D float32 mono waveform in [-1.0, 1.0] at 16 kHz."""
    p = Path(path)
    _guard_path(p)

    with wave.open(str(p), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    if sampwidth == 2:
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        audio = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes in {p}")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    if sr != target_sr and len(audio) > 0:
        duration = len(audio) / float(sr)
        target_len = max(1, int(round(duration * target_sr)))
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
        sr = target_sr

    return audio, sr
