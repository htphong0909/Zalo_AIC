"""Referee 2: Sealed Blind Box Dev Evaluator (STRICT ZERO-LEAKAGE INVARIANT).

Rule: Absolutely no sample IDs, transcripts, or per-sample error logs from
configs/splits/dev_v1.json may ever be revealed or logged.
Outputs strictly macro-level metrics (WER, CER, Sentence EM, Latency).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.asr_metrics import compute_wer, compute_cer, normalize_vi_text

ALLOWED_DEV_MACRO_KEYS = {
    "benchmark",
    "total_samples",
    "wer",
    "cer",
    "sentence_em",
    "latency",
    "timestamp",
}


def run_sealed_blind_dev_evaluation(
    engine,
    output_json: Path | None = None,
    batch_size: int = 8,
) -> dict:
    splits_file = ROOT / "configs" / "splits" / "dev_v1.json"
    splits = json.load(open(splits_file, encoding="utf-8"))
    dev_ids = set(splits.get("dev_ids", []))

    train_jsonl = ROOT / "data" / "train" / "train.jsonl"
    data = []
    with open(train_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item["id"] in dev_ids:
                data.append(item)

    valid_data = []
    valid_paths = []
    for item in data:
        audio_rel = item.get("audio", "")
        audio_path = ROOT / "data" / "train" / audio_rel
        if audio_path.exists():
            valid_data.append(item)
            valid_paths.append(str(audio_path))

    if hasattr(engine, "transcribe_batch") and batch_size > 1:
        res_list = engine.transcribe_batch(valid_paths, batch_size=batch_size)
    else:
        res_list = [engine.transcribe(p) for p in valid_paths]

    wers, cers, exact_count, latencies = [], [], 0, []

    for item, res in zip(valid_data, res_list):
        gt = item.get("audio_text", "")
        pred = res.text
        lat = res.latency_s

        w = compute_wer(pred, gt)[2]
        c = compute_cer(pred, gt)[2]
        wers.append(w)
        cers.append(c)
        latencies.append(lat)

        norm_p = normalize_vi_text(pred)
        norm_g = normalize_vi_text(gt)
        if norm_p == norm_g:
            exact_count += 1

    n = len(wers) or 1
    macro_wer = round(sum(wers) / n * 100, 2)
    macro_cer = round(sum(cers) / n * 100, 2)
    macro_em = round(exact_count / n * 100, 2)
    macro_lat = round(sum(latencies) / n, 4)

    # Strictly sanitize results - ZERO leak of individual sample data
    sanitized_results = {
        "benchmark": "sealed_blind_dev_299",
        "total_samples": len(wers),
        "wer": macro_wer,
        "cer": macro_cer,
        "sentence_em": macro_em,
        "latency": macro_lat,
        "timestamp": time.time(),
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(sanitized_results, f, ensure_ascii=False, indent=2)

    return sanitized_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sealed Blind Box Dev ASR Evaluation")
    parser.add_argument("--model-id", type=str, default="kzap201/phowhisper-large-zaloai")
    parser.add_argument("--engine", type=str, default="whisper", choices=["whisper", "chunkformer"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=str, default="scripts/scratch/temp_golden_res.json")
    parser.add_argument("--adapter-path", type=str, default=None)
    args = parser.parse_args()

    if args.engine == "whisper":
        from src.asr.whisper_engine import WhisperASREngine
        engine = WhisperASREngine(model_id=args.model_id, adapter_id=args.adapter_path)
    else:
        from src.asr.chunkformer_engine import ChunkFormerASREngine
        engine = ChunkFormerASREngine()

    out_p = Path(args.output) if args.output else None
    res = run_sealed_blind_dev_evaluation(engine, output_json=out_p, batch_size=args.batch_size)
    # Print ONLY macro metrics - no individual sample identities
    print(f"[REFEREE 2: SEALED BLIND DEV] Samples={res['total_samples']} | Dev WER={res['wer']}% | Dev CER={res['cer']}% | Dev EM={res['sentence_em']}% | Latency={res['latency']}s")
