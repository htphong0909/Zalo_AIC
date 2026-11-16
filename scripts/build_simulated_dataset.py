#!/usr/bin/env python3
"""Build a simulated test dataset matching the organizer's exact LifeMate split structure.

Creates data/simulated_eval/simulation_test/ with:
- simulation_test.jsonl (renamed sample IDs: sim_0001, sim_0002, ...)
- simulation_test_tools_catalog.json
- simulation_test_environment/ (with per-sample state.json)
- zaic_tools/ (copied clean zaic_tools package)
- audio/ (real .wav files copied under audio/<sim_id>/<sim_id>.wav)
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_SPLIT = ROOT / "data" / "public_test"
DEST_ROOT = ROOT / "data" / "simulated_eval"
DEST_SPLIT = DEST_ROOT / "simulation_test"

# Diverse 6-sample test suite covering:
# 1. Out of scope / no action (public_0002 -> sim_0001)
# 2. Ask missing info (public_0003 -> sim_0002)
# 3. Multi-turn dialogue history (public_0006 -> sim_0003)
# 4. Smart home single/multi action (public_0008 -> sim_0004)
# 5. Calendar / reminder query & action (public_0010 -> sim_0005)
# 6. Alarm creation (public_0039 -> sim_0006)
SAMPLE_MAP = [
    ("public_0002", "sim_0001", "Out of Scope / No Installed App Tool"),
    ("public_0003", "sim_0002", "Ask Missing Info (music.add_to_playlist)"),
    ("public_0006", "sim_0003", "Multi-Turn Dialogue Context (4 history turns)"),
    ("public_0008", "sim_0004", "Smart Home Device Control"),
    ("public_0010", "sim_0005", "Calendar / Schedule Query & Event"),
    ("public_0039", "sim_0006", "Alarm List & Create Sequential"),
]


def main() -> None:
    print(f"Creating simulated test dataset at {DEST_SPLIT} ...")
    if DEST_SPLIT.exists():
        shutil.rmtree(DEST_SPLIT)
    DEST_SPLIT.mkdir(parents=True, exist_ok=True)

    # 1. Copy tools catalog
    src_cat = SRC_SPLIT / "public_test_tools_catalog.json"
    dst_cat = DEST_SPLIT / "simulation_test_tools_catalog.json"
    shutil.copy(src_cat, dst_cat)
    print(f"[1/5] Copied catalog -> {dst_cat.name} ({dst_cat.stat().st_size:,} bytes)")

    # 2. Copy zaic_tools package
    src_zt = SRC_SPLIT / "zaic_tools"
    dst_zt = DEST_SPLIT / "zaic_tools"
    shutil.copytree(src_zt, dst_zt, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"[2/5] Copied zaic_tools package -> {dst_zt}")

    # 3. Load public test samples
    pub_samples = {}
    with open(SRC_SPLIT / "public_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                pub_samples[item["id"]] = item

    # 4. Build simulation_test.jsonl, audio, and environment
    sim_samples = []
    dst_env = DEST_SPLIT / "simulation_test_environment"
    dst_env.mkdir(parents=True, exist_ok=True)
    dst_audio_root = DEST_SPLIT / "audio"
    dst_audio_root.mkdir(parents=True, exist_ok=True)

    for src_id, sim_id, desc in SAMPLE_MAP:
        assert src_id in pub_samples, f"Source sample {src_id} not found!"
        orig = pub_samples[src_id]

        # New audio path relative to simulation_test split
        new_audio_rel = f"audio/{sim_id}/{sim_id}.wav"
        src_wav = SRC_SPLIT / orig["audio"]
        dst_wav = DEST_SPLIT / new_audio_rel
        dst_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_wav, dst_wav)

        # Copy environment state.json
        src_state = SRC_SPLIT / "public_test_environment" / src_id / "state.json"
        dst_state = dst_env / sim_id / "state.json"
        dst_state.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_state, dst_state)

        # Construct simulated sample item
        sim_item = {
            "id": sim_id,
            "audio": new_audio_rel,
            "context": orig["context"],
            "history": orig.get("history", []),
        }
        sim_samples.append(sim_item)
        print(f"  Sample {sim_id} ({desc}): audio={dst_wav.stat().st_size:,}B, state={dst_state.stat().st_size:,}B")

    # Write simulation_test.jsonl
    sim_jsonl_path = DEST_SPLIT / "simulation_test.jsonl"
    with open(sim_jsonl_path, "w", encoding="utf-8", newline="\n") as f:
        for s in sim_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[3/5] Wrote {len(sim_samples)} samples -> {sim_jsonl_path.name}")

    # 5. Verify simulator loads seamlessly with zaic_tools
    print("[4/5] Verifying simulated environment loading via zaic_tools...")
    import sys
    sys.path.insert(0, str(DEST_SPLIT))
    from zaic_tools import ToolEnv, catalog_path

    cat_p = catalog_path("simulation_test", str(DEST_ROOT))
    assert Path(cat_p).exists(), f"Catalog path not found: {cat_p}"

    for _, sim_id, _ in SAMPLE_MAP:
        env = ToolEnv("simulation_test", sim_id, str(DEST_ROOT))
        h = env.get_state_hash()
        calls = env.get_tool_calls()
        assert len(h) == 64, f"Invalid state hash length for {sim_id}: {h}"
        print(f"  ToolEnv OK: {sim_id} -> initial state_hash={h[:16]}... allowed={len(env.allowed)} tools")

    print(f"[5/5] SUCCESS: Simulated test environment is 100% compliant at {DEST_SPLIT}")


if __name__ == "__main__":
    main()
