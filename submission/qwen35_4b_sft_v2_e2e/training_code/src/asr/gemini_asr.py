"""Gemini 3.5 Transcribe engine with multi-key round-robin, strict TPM + RPM rate limiting, and backoff."""
from __future__ import annotations

import base64
import collections
import time
import unicodedata
from pathlib import Path
from typing import Sequence

import requests

from src.asr.base import ASRResult, strip_asr_tags

DEFAULT_GEMINI_KEYS: tuple[str, ...] = tuple(
    k.strip() for k in os.environ.get("GEMINI_API_KEYS", "").split(",") if k.strip()
)


class GeminiTranscribeEngine:
    """ASR engine calling models/gemini-3.5-transcribe with multi-key pooling and strict TPM/RPM rate limiting."""

    def __init__(
        self,
        api_keys: Sequence[str] = DEFAULT_GEMINI_KEYS,
        *,
        model_id: str = "gemini-3.5-transcribe",
        max_tpm_per_key: int = 18000,
        max_rpm_per_key: int = 14,
        timeout_s: float = 35.0,
        max_retries: int = 6,
    ):
        self.api_keys = list(api_keys)
        if not self.api_keys:
            raise ValueError("At least one Gemini API key is required")
        self.model_id = model_id
        self.max_tpm_per_key = max_tpm_per_key
        self.max_rpm_per_key = max_rpm_per_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries

        # Deque of (timestamp, token_count) per key
        self._token_usage: dict[str, collections.deque[tuple[float, int]]] = {
            k: collections.deque() for k in self.api_keys
        }
        # Deque of request timestamps per key
        self._request_history: dict[str, collections.deque[float]] = {
            k: collections.deque() for k in self.api_keys
        }
        # Penalty expiration timestamp per key (e.g. if 429 received)
        self._penalties: dict[str, float] = {k: 0.0 for k in self.api_keys}
        self._key_index = 0

    def _clean_history(self, key: str, now: float) -> tuple[int, int]:
        cutoff = now - 60.0
        # Clean tokens
        tdq = self._token_usage[key]
        while tdq and tdq[0][0] < cutoff:
            tdq.popleft()
        total_tokens = sum(tok for _, tok in tdq)

        # Clean requests
        rdq = self._request_history[key]
        while rdq and rdq[0] < cutoff:
            rdq.popleft()
        total_requests = len(rdq)

        return total_tokens, total_requests

    def _select_key(self, estimated_tokens: int = 250) -> str:
        now = time.monotonic()
        # 1. First pass: try round-robin for an immediately available key
        for _ in range(len(self.api_keys)):
            k = self.api_keys[self._key_index]
            self._key_index = (self._key_index + 1) % len(self.api_keys)
            if now < self._penalties[k]:
                continue
            tokens, reqs = self._clean_history(k, now)
            if tokens + estimated_tokens <= self.max_tpm_per_key and reqs + 1 <= self.max_rpm_per_key:
                return k

        # 2. All keys are constrained: determine minimum wait time
        min_wait = 60.0
        best_key = self.api_keys[0]
        for k in self.api_keys:
            pen_wait = max(0.0, self._penalties[k] - now)
            tokens, reqs = self._clean_history(k, now)
            rdq = self._request_history[k]
            tdq = self._token_usage[k]

            req_wait = max(0.0, (rdq[0] + 60.0) - now + 0.3) if (reqs >= self.max_rpm_per_key and rdq) else 0.0
            tok_wait = max(0.0, (tdq[0][0] + 60.0) - now + 0.3) if (tokens + estimated_tokens > self.max_tpm_per_key and tdq) else 0.0
            wait_for_k = max(pen_wait, req_wait, tok_wait)

            if wait_for_k < min_wait:
                min_wait = wait_for_k
                best_key = k

        sleep_s = max(0.5, min_wait)
        time.sleep(sleep_s)
        return best_key

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        extra_context: str | None = None,
    ) -> ASRResult:
        p = Path(audio_path)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {p}")

        audio_bytes = p.read_bytes()
        b64_audio = base64.b64encode(audio_bytes).decode("ascii")

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "audio/wav",
                                "data": b64_audio,
                            }
                        }
                    ]
                }
            ]
        }

        t0 = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            key = self._select_key(estimated_tokens=250)
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent?key={key}"

            now_mono = time.monotonic()
            self._request_history[key].append(now_mono)

            try:
                resp = requests.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout_s,
                )
                now_done = time.monotonic()

                if resp.status_code == 200:
                    data = resp.json()
                    tokens = data.get("usageMetadata", {}).get("totalTokenCount", 200)
                    self._token_usage[key].append((now_done, tokens))

                    raw_text = ""
                    candidates = data.get("candidates") or []
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts") or []
                        if parts:
                            raw_text = (
                                parts[0].get("audioTranscription", {}).get("text")
                                or parts[0].get("text")
                                or ""
                            )

                    clean = strip_asr_tags(raw_text)
                    clean = unicodedata.normalize("NFC", clean).strip()

                    return ASRResult(
                        text=clean,
                        raw_text=raw_text,
                        latency_s=time.perf_counter() - t0,
                    )

                if resp.status_code == 429:
                    # Rate limit hit: put this key on cooldown for 20s and record error
                    self._penalties[key] = now_done + 20.0
                    last_error = RuntimeError(f"HTTP 429 RateLimit on key {key[:8]}... (attempt {attempt+1}/{self.max_retries})")
                    time.sleep(2.0)
                    continue

                if resp.status_code >= 500:
                    last_error = RuntimeError(f"HTTP {resp.status_code} on key {key[:8]}... (attempt {attempt+1}/{self.max_retries})")
                    time.sleep(2.0)
                    continue

                last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()

            except (requests.RequestException, OSError) as e:
                last_error = e
                time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(
            f"Failed to transcribe {audio_path} after {self.max_retries} attempts: {last_error}"
        )
