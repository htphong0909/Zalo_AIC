"""Referee 1: Glass-Box Train Evaluator for ASR forensic diagnosis.

Runs on train_ids (from configs/splits/dev_v1.json).
Outputs full error clusters, phoneme confusions, and per-sample diagnostics
so the loop-researcher can analyze root causes.
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


def run_train_evaluation(
    engine,
    max_samples: int = 150,
    output_json: Path | None = None,
    batch_size: int = 8,
) -> dict:
    splits_file = ROOT / "configs" / "splits" / "dev_v1.json"
    splits = json.load(open(splits_file, encoding="utf-8"))
    train_ids = set(splits.get("train_ids", []))

    train_jsonl = ROOT / "data" / "train" / "train.jsonl"
    data = []
    with open(train_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item["id"] in train_ids:
                data.append(item)
            if len(data) >= max_samples:
                break

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
    error_clusters = []

    for item, res in zip(valid_data, res_list):
        sid = item["id"]
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
        is_em = (norm_p == norm_g)
        if is_em:
            exact_count += 1
        else:
            if len(error_clusters) < 50:
                error_clusters.append({
                    "id": sid,
                    "gt": gt,
                    "pred": pred,
                    "wer": round(w * 100, 2),
                    "cer": round(c * 100, 2),
                })

    n = len(wers) or 1
    macro_wer = round(sum(wers) / n * 100, 2)
    macro_cer = round(sum(cers) / n * 100, 2)
    macro_em = round(exact_count / n * 100, 2)
    macro_lat = round(sum(latencies) / n, 4)

    results = {
        "benchmark": "train_glass_box",
        "total_samples": len(wers),
        "wer": macro_wer,
        "cer": macro_cer,
        "sentence_em": macro_em,
        "latency": macro_lat,
        "error_samples": error_clusters,
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Glass-Box Train ASR Evaluation")
    parser.add_argument("--model-id", type=str, default="kzap201/phowhisper-large-zaloai")
    parser.add_argument("--engine", type=str, default="whisper", choices=["whisper", "chunkformer"])
    parser.add_argument("--max-samples", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=str, default="scripts/scratch/temp_train_res.json")
    parser.add_argument("--adapter-path", type=str, default=None)
    args = parser.parse_args()

    if args.engine == "whisper":
        from src.asr.whisper_engine import WhisperASREngine
        engine = WhisperASREngine(model_id=args.model_id, adapter_id=args.adapter_path)
    else:
        from src.asr.chunkformer_engine import ChunkFormerASREngine
        engine = ChunkFormerASREngine()

    out_p = Path(args.output) if args.output else None
    res = run_train_evaluation(engine, max_samples=args.max_samples, output_json=out_p, batch_size=args.batch_size)
    print(f"[REFEREE 1: TRAIN GLASS-BOX] Samples={res['total_samples']} | WER={res['wer']}% | CER={res['cer']}% | EM={res['sentence_em']}% | Latency={res['latency']}s")
