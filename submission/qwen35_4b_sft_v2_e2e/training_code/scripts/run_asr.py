"""Run ASR transcription over a split (including history audio turns) and cache transcripts.jsonl."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.asr.aura_nbest import extract_phonetic_diff
from src.asr.base import FakeASREngine
from src.asr.chunkformer_engine import ChunkFormerASREngine
from src.asr.gemini_asr import GeminiTranscribeEngine
from src.asr.qwen_asr import Qwen3ASREngine
from src.asr.whisper_engine import WhisperASREngine
from src.core.dataset import audio_path, load_split
from src.core.schemas import Sample
from src.eval.asr_metrics import evaluate_transcripts


def _load_ids(path: str | Path) -> set[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return set(data)
    return set(data.get("dev_ids") or data.get("ids") or [])


def _collect_entity_strings(obj: Any, out: list[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and k in ("name", "title", "location", "station", "author", "provider"):
                if 2 <= len(v) <= 60 and v not in out:
                    out.append(v)
            else:
                _collect_entity_strings(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_entity_strings(item, out)


_APP_DOMAIN_KEYWORDS: dict[str, str] = {
    "learning": "khóa học, tiếng Hàn, tiếng Nhật, học phí, lịch học",
    "banking": "chuyển khoản, số dư, tiền, người thụ hưởng, tài khoản",
    "ewallet": "ví điện tử, chuyển tiền, số dư",
    "fuel_price": "giá xăng, dầu diesel, xăng ron 95, e5",
    "gold_price": "giá vàng, sjc, lượng vàng, chỉ vàng",
    "lottery": "kết quả xổ số, giải đặc biệt, đài hà nội, đài miền bắc",
    "exchange_rate": "tỷ giá ngoại tệ, đô la mỹ, usd, eur, jpy",
    "smart_home": "đèn, quạt, rèm cửa, điều hòa, máy sưởi, công tắc",
    "robot_vacuum": "robot hút bụi, lau nhà, dọn dẹp",
    "ride_hailing": "đặt xe, điểm đón, điểm đến, chuyến đi",
    "food_delivery": "đặt món, giao hàng, đồ ăn, quán",
    "weather": "thời tiết, dự báo thời tiết, nhiệt độ, mưa",
    "calendar": "lịch hẹn, sự kiện, cuộc họp, nhắc lịch",
    "alarm": "báo thức, hẹn giờ",
    "reminders": "nhắc nhở, việc cần làm, ghi chú",
    "contacts": "danh bạ, liên hệ, số điện thoại",
    "bill_payment": "thanh toán hóa đơn, tiền điện, tiền nước, internet",
    "home_security": "báo động, camera an ninh, cảnh báo",
    "tv": "ti vi, kênh truyền hình, âm lượng",
}


def build_sample_asr_context(sample: Sample) -> str:
    """Build legal inference-time context hints from user_profile, installed_apps, and prior history."""
    hints: list[str] = []
    profile = (sample.context or {}).get("user_profile") or {}
    for k in ("city", "home_address", "work_address"):
        val = profile.get(k)
        if isinstance(val, str) and val.strip():
            hints.append(val.strip())

    # Add domain keywords from installed apps to bias ASR decoder
    installed = list((sample.context or {}).get("installed_apps", {}).keys())
    for app in installed:
        kw = _APP_DOMAIN_KEYWORDS.get(app)
        if kw:
            hints.append(kw)

    for turn in getattr(sample, "history", []) or []:
        _collect_entity_strings(turn.tool_results, hints)

    if not hints:
        return ""
    return "Ngữ cảnh mẫu: " + ", ".join(hints[:30]) + "."


def _build_asr_row(
    sample: Sample,
    *,
    curr_text: str,
    nbest_alts: list[str],
    nbest_scores: list[float],
    hist_texts: dict[str, str],
    latency_s: float,
    nbest_beams: int,
    nbest_max_delta: float,
) -> dict[str, Any]:
    phonetic_diff: str | None = None
    if nbest_alts:
        top_score = nbest_scores[0] if nbest_scores else None
        filtered_alts: list[str] = []
        seen_alts: set[str] = {curr_text}
        for j, alt in enumerate(nbest_alts):
            if not alt or alt in seen_alts:
                continue
            if top_score is not None and j < len(nbest_scores):
                if (top_score - nbest_scores[j]) > nbest_max_delta:
                    continue
            seen_alts.add(alt)
            filtered_alts.append(alt)
        if filtered_alts:
            phonetic_diff = extract_phonetic_diff(
                [curr_text, *filtered_alts],
                max_score_delta=nbest_max_delta,
            )

    row: dict[str, Any] = {
        "id": sample.id,
        "audio_text": curr_text,
        "history_texts": hist_texts,
        "latency_s": round(latency_s, 4),
    }
    if nbest_beams > 1 or nbest_alts or phonetic_diff:
        row["nbest"] = nbest_alts
        row["asr_phonetic_diff"] = phonetic_diff
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--root", default="data")
    parser.add_argument("--ids", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--engine",
        choices=("qwen3_asr", "whisper", "gemini", "chunkformer", "oracle", "fake"),
        default="qwen3_asr",
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--api-keys", nargs="*", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--no-context-bias", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for WhisperASR batched inference")
    parser.add_argument(
        "--nbest-beams",
        type=int,
        default=1,
        help="Optional Whisper beam count (> 1) to attach N-best and phonetic diff hints",
    )
    parser.add_argument(
        "--nbest-max-delta",
        type=float,
        default=0.65,
        help="Maximum sequence score delta vs top beam for phonetic diff hints",
    )
    parser.add_argument("--out", required=True, help="Output transcripts.jsonl path")
    parser.add_argument("--metrics-out", default=None, help="Optional asr_metrics.json path")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    samples = load_split(args.split, args.root)
    if args.ids:
        keep = _load_ids(args.ids)
        samples = [s for s in samples if s.id in keep]
    if args.limit is not None:
        samples = samples[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_rows: dict[str, dict] = {}
    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    done_rows[row["id"]] = row

    engine = None
    if args.engine == "qwen3_asr":
        m_id = args.model_id or "Qwen/Qwen3-ASR-0.6B-hf"
        engine = Qwen3ASREngine(
            model_id=m_id,
            adapter_path=args.adapter_path,
            dtype=args.dtype,
        )
    elif args.engine == "whisper":
        m_id = args.model_id or "kzap201/phowhisper-large-zaloai"
        w_dtype = "float16" if args.dtype in ("float16", "bfloat16") else args.dtype
        engine = WhisperASREngine(
            model_id=m_id,
            dtype=w_dtype,
            adapter_id=args.adapter_path,
        )
    elif args.engine == "gemini":
        m_id = args.model_id or "gemini-3.5-transcribe"
        engine = (
            GeminiTranscribeEngine(api_keys=args.api_keys, model_id=m_id)
            if args.api_keys
            else GeminiTranscribeEngine(model_id=m_id)
        )
    elif args.engine == "chunkformer":
        m_id = args.model_id or "khanhld/chunkformer-ctc-large-vie"
        engine = ChunkFormerASREngine(model_id=m_id)
    elif args.engine == "fake":
        engine = FakeASREngine(default_text="xin chào")

    mode = "a" if args.resume and out_path.exists() else "w"
    t_start = time.perf_counter()
    processed = 0

    with open(out_path, mode, encoding="utf-8") as fout:
        pending = [s for s in samples if s.id not in done_rows]
        use_batched_whisper = isinstance(engine, WhisperASREngine) and int(args.batch_size) > 1
        if use_batched_whisper:
            bs = max(1, int(args.batch_size))
            for start in range(0, len(pending), bs):
                chunk = pending[start : start + bs]
                t0 = time.perf_counter()
                cur_wavs = [audio_path(args.split, args.root, s.audio) for s in chunk]
                cur_ctxs = None if args.no_context_bias else [build_sample_asr_context(s) for s in chunk]
                cur_results = engine.transcribe_batch(
                    cur_wavs,
                    batch_size=bs,
                    extra_contexts=cur_ctxs,
                    num_beams_nbest=args.nbest_beams if args.nbest_beams > 1 else 1,
                    diverse_nbest=True,
                )

                hist_jobs: list[tuple[int, str, Path, str | None]] = []
                hist_texts_list: list[dict[str, str]] = [{} for _ in chunk]
                for pos, s in enumerate(chunk):
                    s_ctx = None if args.no_context_bias or cur_ctxs is None else cur_ctxs[pos]
                    for i, t in enumerate(s.history):
                        if t.role != "user":
                            continue
                        if t.audio:
                            hist_jobs.append((pos, str(i), audio_path(args.split, args.root, t.audio), s_ctx))
                        elif t.audio_text or t.text:
                            hist_texts_list[pos][str(i)] = t.audio_text or t.text or ""

                if hist_jobs:
                    h_wavs = [job[2] for job in hist_jobs]
                    h_ctxs = None if args.no_context_bias else [job[3] for job in hist_jobs]
                    h_results = engine.transcribe_batch(
                        h_wavs,
                        batch_size=bs,
                        extra_contexts=h_ctxs,
                        num_beams_nbest=1,
                        diverse_nbest=False,
                    )
                    for (pos, turn_idx_str, _, _), h_res in zip(hist_jobs, h_results):
                        hist_texts_list[pos][turn_idx_str] = h_res.text

                batch_avg_lat = (time.perf_counter() - t0) / max(1, len(chunk))
                for pos, (s, curr_res) in enumerate(zip(chunk, cur_results)):
                    row = _build_asr_row(
                        s,
                        curr_text=curr_res.text,
                        nbest_alts=list(getattr(curr_res, "nbest", []) or []),
                        nbest_scores=list(getattr(curr_res, "nbest_scores", []) or []),
                        hist_texts=hist_texts_list[pos],
                        latency_s=batch_avg_lat,
                        nbest_beams=args.nbest_beams,
                        nbest_max_delta=args.nbest_max_delta,
                    )
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    done_rows[s.id] = row
                    processed += 1
                fout.flush()
                elapsed = time.perf_counter() - t_start
                done_total = len(done_rows)
                print(
                    f"[{done_total}/{len(samples)}] processed={processed} elapsed={elapsed:.1f}s "
                    f"avg={elapsed / max(1, processed):.2f}s/sample last_id={chunk[-1].id}",
                    flush=True,
                )
        else:
            for idx, s in enumerate(samples, 1):
                if s.id in done_rows:
                    continue

                t0 = time.perf_counter()
                nbest_alts: list[str] = []
                nbest_scores: list[float] = []
                if args.engine == "oracle":
                    curr_text = s.audio_text or ""
                    hist_texts = {
                        str(i): (t.audio_text or t.text or "")
                        for i, t in enumerate(s.history)
                        if t.role == "user"
                    }
                else:
                    assert engine is not None
                    extra_ctx = None if args.no_context_bias else build_sample_asr_context(s)
                    wav_p = audio_path(args.split, args.root, s.audio)
                    if isinstance(engine, WhisperASREngine) and args.nbest_beams > 1:
                        curr_res = engine.transcribe(
                            wav_p,
                            extra_context=extra_ctx,
                            num_beams_nbest=args.nbest_beams,
                            diverse_nbest=True,
                        )
                    elif isinstance(engine, (Qwen3ASREngine, WhisperASREngine, GeminiTranscribeEngine, ChunkFormerASREngine)):
                        curr_res = engine.transcribe(wav_p, extra_context=extra_ctx)
                    else:
                        curr_res = engine.transcribe(wav_p)
                    curr_text = curr_res.text
                    nbest_alts = list(getattr(curr_res, "nbest", []) or [])
                    nbest_scores = list(getattr(curr_res, "nbest_scores", []) or [])
                    hist_texts = {}
                    for i, t in enumerate(s.history):
                        if t.role == "user":
                            if t.audio:
                                h_wav = audio_path(args.split, args.root, t.audio)
                                if isinstance(engine, (Qwen3ASREngine, WhisperASREngine, GeminiTranscribeEngine, ChunkFormerASREngine)):
                                    hist_texts[str(i)] = engine.transcribe(h_wav, extra_context=extra_ctx).text
                                else:
                                    hist_texts[str(i)] = engine.transcribe(h_wav).text
                            elif t.audio_text or t.text:
                                hist_texts[str(i)] = t.audio_text or t.text or ""

                row = _build_asr_row(
                    s,
                    curr_text=curr_text,
                    nbest_alts=nbest_alts,
                    nbest_scores=nbest_scores,
                    hist_texts=hist_texts,
                    latency_s=time.perf_counter() - t0,
                    nbest_beams=args.nbest_beams,
                    nbest_max_delta=args.nbest_max_delta,
                )
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                done_rows[s.id] = row
                processed += 1

                if processed % 20 == 0 or idx == len(samples):
                    elapsed = time.perf_counter() - t_start
                    print(
                        f"[{idx}/{len(samples)}] processed={processed} elapsed={elapsed:.1f}s "
                        f"avg={elapsed / max(1, processed):.2f}s/sample last_id={s.id}",
                        flush=True,
                    )

    if args.metrics_out or any(s.audio_text for s in samples):
        metrics = evaluate_transcripts(done_rows, samples)
        if args.metrics_out:
            m_path = Path(args.metrics_out)
            m_path.parent.mkdir(parents=True, exist_ok=True)
            m_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        summary_metrics = {k: v for k, v in metrics.items() if k != "worst_samples"}
        try:
            print(json.dumps(summary_metrics, indent=2, ensure_ascii=False))
        except Exception:
            print(json.dumps(summary_metrics, indent=2))


if __name__ == "__main__":
    main()
