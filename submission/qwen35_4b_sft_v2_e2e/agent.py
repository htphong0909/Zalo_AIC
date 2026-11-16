"""Unified offline E2E inference runtime for Qwen3.5-4B SFT V2 + PhoWhisper-Large-ZaloAI."""
from __future__ import annotations

import os


def configure_offline_env() -> None:
    """Enforce strictly offline HuggingFace/Transformers and safe vLLM spawn multiprocessing."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


configure_offline_env()

import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.agent.llm_client import VLLMBackend, create_llm_backend  # noqa: E402
from src.agent.react_loop import BatchReactItem, run_react_batch, run_react_loop  # noqa: E402
from src.asr.aura_nbest import extract_phonetic_diff  # noqa: E402
from src.asr.whisper_engine import WhisperASREngine  # noqa: E402
from src.core.config import ExperimentConfig, load_config  # noqa: E402
from src.core.dataset import audio_path, load_catalog, load_split  # noqa: E402
from src.core.schemas import Sample  # noqa: E402
from src.env.safe_env import SafeEnv  # noqa: E402

DEFAULT_CONFIG_PATH = CODE_ROOT / "configs" / "e2e_qwen35_4b_sft_v2.yaml"


def seed_everything(seed: int = 42) -> None:
    """Fix all RNG seeds and cuDNN determinism flags for reproducible greedy inference."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_code_path(path_str: str) -> str:
    """Resolve /code/... paths seamlessly inside Docker or local workspace checkouts."""
    p = Path(path_str)
    if p.exists():
        return str(p)
    s = path_str.replace("\\", "/")
    if s.startswith("/code/"):
        rel = CODE_ROOT / s[len("/code/") :]
        if rel.exists():
            return str(rel)
    if not p.is_absolute():
        rel = CODE_ROOT / path_str
        if rel.exists():
            return str(rel)
    return path_str


def discover_split(data_root: str | Path = "/data", split_override: str | None = None) -> str:
    """Dynamically discover the mounted evaluation split under /data without inspecting *_environment."""
    if split_override:
        return split_override
    env_split = os.environ.get("DATA_SPLIT")
    if env_split:
        return env_split
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {root}")
    candidates = [
        p.name
        for p in sorted(root.iterdir())
        if p.is_dir()
        and (p / f"{p.name}_tools_catalog.json").is_file()
        and (p / f"{p.name}.jsonl").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No valid split directory with <split>_tools_catalog.json found under {root}")
    if len(candidates) == 1:
        return candidates[0]
    for preferred in ("private_test", "public_test"):
        if preferred in candidates:
            return preferred
    raise RuntimeError(f"Ambiguous splits under {root}: {candidates}. Pass --split or set DATA_SPLIT.")


def import_zaic_tools(data_root: str | Path, split: str) -> Any:
    """Import zaic_tools strictly from the mounted <data_root>/<split>/ directory."""
    split_dir = (Path(data_root) / split).resolve()
    if str(split_dir) not in sys.path:
        sys.path.insert(0, str(split_dir))
    cached = sys.modules.get("zaic_tools")
    if cached is not None and getattr(cached, "__file__", None):
        cached_path = Path(cached.__file__).resolve()
        if not cached_path.is_relative_to(split_dir):
            for mod_name in [m for m in list(sys.modules) if m == "zaic_tools" or m.startswith("zaic_tools.")]:
                del sys.modules[mod_name]
    import zaic_tools

    zt_file = Path(zaic_tools.__file__).resolve()
    if not zt_file.is_relative_to(split_dir):
        raise RuntimeError(f"zaic_tools imported from unexpected location {zt_file}, expected inside {split_dir}")
    return zaic_tools


def load_raw_yaml(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_models(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    asr_engine: Any = None,
    llm_backend: Any = None,
    device: str = "cuda",
) -> tuple[ExperimentConfig, dict[str, Any], Any, Any]:
    """Load offline Qwen3.5-4B SFT V2 (vLLM) and PhoWhisper-Large-ZaloAI models deterministically."""
    configure_offline_env()
    cfg = load_config(config_path)
    raw_cfg = load_raw_yaml(config_path)
    seed_everything(cfg.seed)

    resolved_llm_id = resolve_code_path(cfg.llm.model_id)
    resolved_asr_id = resolve_code_path(cfg.asr.model_id)
    cfg = cfg.model_copy(
        update={
            "llm": cfg.llm.model_copy(
                update={
                    "model_id": resolved_llm_id,
                    "adapter_path": resolve_code_path(cfg.llm.adapter_path) if cfg.llm.adapter_path else None,
                }
            ),
            "asr": cfg.asr.model_copy(
                update={
                    "model_id": resolved_asr_id,
                    "adapter_path": resolve_code_path(cfg.asr.adapter_path) if cfg.asr.adapter_path else None,
                }
            ),
        }
    )

    if asr_engine is None:
        if not Path(cfg.asr.model_id).is_dir():
            raise FileNotFoundError(
                f"Offline ASR checkpoint missing at {cfg.asr.model_id}. Run prepare_models.py first."
            )
        asr_dtype = "float16" if cfg.asr.dtype in ("float16", "bfloat16") else cfg.asr.dtype
        asr_engine = WhisperASREngine(
            model_id=cfg.asr.model_id,
            device=device,
            dtype=asr_dtype,
            language=cfg.asr.language,
            task=cfg.asr.task,
            enable_itn=cfg.asr.enable_itn,
            adapter_id=cfg.asr.adapter_path,
        )

    if llm_backend is None:
        if not Path(cfg.llm.model_id).is_dir():
            raise FileNotFoundError(
                f"Offline LLM checkpoint missing at {cfg.llm.model_id}. Run prepare_models.py first."
            )
        if cfg.llm.backend == "vllm":
            llm_backend = VLLMBackend(
                model_id=cfg.llm.model_id,
                adapter_path=cfg.llm.adapter_path,
                max_model_len=cfg.llm.max_model_len,
                max_new_tokens=cfg.llm.max_new_tokens,
                enable_thinking=cfg.llm.enable_thinking,
                dtype=cfg.llm.dtype,
                repetition_penalty=cfg.llm.repetition_penalty,
                gpu_memory_utilization=cfg.llm.gpu_memory_utilization,
                max_num_seqs=getattr(cfg.llm, "max_num_seqs", None),
                guided_decoding=cfg.llm.guided_decoding,
                seed=cfg.seed,
            )
        else:
            llm_backend = create_llm_backend(cfg.llm)
        if hasattr(llm_backend, "warmup"):
            llm_backend.warmup()

    return cfg, raw_cfg, asr_engine, llm_backend


def build_phonetic_diff(
    curr_text: str,
    nbest_alts: list[str],
    nbest_scores: list[float],
    *,
    nbest_max_delta: float = 0.65,
) -> str | None:
    """Filter N-best beam alternatives by score margin and extract compact phonetic ambiguity diff."""
    if not nbest_alts:
        return None
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
    if not filtered_alts:
        return None
    return extract_phonetic_diff([curr_text, *filtered_alts], max_score_delta=nbest_max_delta)


def transcribe_sample_audio(
    sample: Sample,
    *,
    split: str,
    data_root: str | Path,
    asr_engine: Any,
    nbest_beams: int = 3,
    nbest_max_delta: float = 0.65,
) -> tuple[str, dict[int, str], str | None]:
    """Transcribe current request audio and sample's prior user audio turns."""
    wav_p = audio_path(split, data_root, sample.audio)
    history_texts: dict[int, str] = {}
    h_jobs: list[tuple[int, Path]] = []
    for i, turn in enumerate(sample.history):
        if turn.role != "user":
            continue
        if turn.audio:
            h_jobs.append((i, audio_path(split, data_root, turn.audio)))
        elif turn.audio_text or turn.text:
            history_texts[i] = turn.audio_text or turn.text or ""

    if nbest_beams <= 1 and h_jobs and hasattr(asr_engine, "transcribe_batch"):
        all_wavs = [wav_p, *(h_wav for _, h_wav in h_jobs)]
        all_res = asr_engine.transcribe_batch(
            all_wavs,
            batch_size=min(16, len(all_wavs)),
            extra_contexts=None,
            num_beams_nbest=1,
            diverse_nbest=False,
        )
        curr_text = all_res[0].text
        for (turn_idx, _), h_res in zip(h_jobs, all_res[1:]):
            history_texts[turn_idx] = h_res.text
        return curr_text, history_texts, None

    if nbest_beams > 1:
        try:
            curr_res = asr_engine.transcribe(
                wav_p,
                extra_context=None,
                num_beams_nbest=nbest_beams,
                diverse_nbest=True,
            )
        except TypeError:
            curr_res = asr_engine.transcribe(wav_p, extra_context=None)
    else:
        curr_res = asr_engine.transcribe(wav_p, extra_context=None)

    curr_text = curr_res.text
    phonetic_diff = (
        build_phonetic_diff(
            curr_text,
            list(getattr(curr_res, "nbest", []) or []),
            list(getattr(curr_res, "nbest_scores", []) or []),
            nbest_max_delta=nbest_max_delta,
        )
        if nbest_beams > 1
        else None
    )

    if len(h_jobs) == 1:
        turn_idx, h_wav = h_jobs[0]
        history_texts[turn_idx] = asr_engine.transcribe(h_wav, extra_context=None).text
    elif len(h_jobs) > 1:
        h_wavs = [job[1] for job in h_jobs]
        if hasattr(asr_engine, "transcribe_batch"):
            h_results = asr_engine.transcribe_batch(
                h_wavs,
                batch_size=min(16, len(h_wavs)),
                extra_contexts=None,
                num_beams_nbest=1,
                diverse_nbest=False,
            )
            for (turn_idx, _), h_res in zip(h_jobs, h_results):
                history_texts[turn_idx] = h_res.text
        else:
            for turn_idx, h_wav in h_jobs:
                history_texts[turn_idx] = asr_engine.transcribe(h_wav, extra_context=None).text

    return curr_text, history_texts, phonetic_diff


def _collect_batch_audio(
    chunk: list[Sample],
    *,
    split: str,
    data_root: str | Path,
    asr_engine: Any,
    asr_batch_size: int,
    nbest_beams: int,
    nbest_max_delta: float,
) -> list[tuple[str, dict[int, str], str | None]]:
    """Transcribe all current + prior user audio files across a sample chunk via batched ASR."""
    if not hasattr(asr_engine, "transcribe_batch"):
        return [
            transcribe_sample_audio(
                s,
                split=split,
                data_root=data_root,
                asr_engine=asr_engine,
                nbest_beams=nbest_beams,
                nbest_max_delta=nbest_max_delta,
            )
            for s in chunk
        ]

    curr_wavs: list[Path] = []
    hist_jobs: list[tuple[int, int, Path]] = []
    hist_texts_list: list[dict[int, str]] = [{} for _ in chunk]

    for s_idx, sample in enumerate(chunk):
        curr_wavs.append(audio_path(split, data_root, sample.audio))
        for t_idx, turn in enumerate(sample.history):
            if turn.role != "user":
                continue
            if turn.audio:
                hist_jobs.append((s_idx, t_idx, audio_path(split, data_root, turn.audio)))
            elif turn.audio_text or turn.text:
                hist_texts_list[s_idx][t_idx] = turn.audio_text or turn.text or ""

    if nbest_beams <= 1:
        all_wavs = [*curr_wavs, *(j[2] for j in hist_jobs)]
        all_res = asr_engine.transcribe_batch(
            all_wavs,
            batch_size=max(1, asr_batch_size),
            extra_contexts=None,
            num_beams_nbest=1,
            diverse_nbest=False,
        )
        curr_res_list = all_res[: len(curr_wavs)]
        hist_res_list = all_res[len(curr_wavs) :]
    else:
        curr_res_list = asr_engine.transcribe_batch(
            curr_wavs,
            batch_size=max(1, min(8, asr_batch_size)),
            extra_contexts=None,
            num_beams_nbest=nbest_beams,
            diverse_nbest=True,
        )
        hist_res_list = (
            asr_engine.transcribe_batch(
                [j[2] for j in hist_jobs],
                batch_size=max(1, asr_batch_size),
                extra_contexts=None,
                num_beams_nbest=1,
                diverse_nbest=False,
            )
            if hist_jobs
            else []
        )

    for (s_idx, t_idx, _), h_res in zip(hist_jobs, hist_res_list):
        hist_texts_list[s_idx][t_idx] = h_res.text

    out: list[tuple[str, dict[int, str], str | None]] = []
    for c_res, h_map in zip(curr_res_list, hist_texts_list):
        pdiff = (
            build_phonetic_diff(
                c_res.text,
                list(getattr(c_res, "nbest", []) or []),
                list(getattr(c_res, "nbest_scores", []) or []),
                nbest_max_delta=nbest_max_delta,
            )
            if nbest_beams > 1
            else None
        )
        out.append((c_res.text, h_map, pdiff))
    return out


def infer_sample(
    sample: Sample,
    *,
    split: str,
    data_root: str | Path,
    catalog: dict[str, dict[str, Any]],
    asr_engine: Any,
    llm_backend: Any,
    cfg: ExperimentConfig,
    raw_cfg: dict[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Run strictly isolated per-sample E2E inference and return submission JSON row."""
    t1 = time.perf_counter()
    env = SafeEnv(split, sample.id, root=data_root)
    asr_raw = (raw_cfg or {}).get("asr") or {}
    nbest_beams = int(asr_raw.get("nbest_beams", getattr(cfg.asr, "nbest_beams", 1)))
    nbest_max_delta = float(asr_raw.get("nbest_max_delta", getattr(cfg.asr, "nbest_max_delta", 0.65)))

    try:
        curr_text, history_texts, phonetic_diff = transcribe_sample_audio(
            sample,
            split=split,
            data_root=data_root,
            asr_engine=asr_engine,
            nbest_beams=nbest_beams,
            nbest_max_delta=nbest_max_delta,
        )
        res = run_react_loop(
            sample,
            env,
            catalog,
            llm_backend,
            current_text=curr_text,
            history_texts=history_texts,
            asr_phonetic_diff=phonetic_diff,
            prompt_style=cfg.agent.prompt_style,
            max_steps=cfg.agent.max_steps,
            max_repairs=cfg.agent.max_repairs,
            timeout_s=cfg.agent.timeout_s,
            max_prompt_tokens=cfg.agent.max_prompt_tokens,
            max_result_chars=cfg.agent.max_result_chars,
            guardrails=cfg.agent.guardrails,
        )
        tool_calls = res.tool_calls
        state_hash = res.state_hash
    except Exception:
        if strict:
            raise
        traceback.print_exc(file=sys.stderr)
        tool_calls = env.get_tool_calls()
        state_hash = env.get_state_hash()

    elapsed_ms = int(round((time.perf_counter() - t1) * 1000))
    return {
        "id": sample.id,
        "tool_calls": tool_calls,
        "state_hash": state_hash,
        "time": elapsed_ms,
    }


def infer_batch(
    chunk: list[Sample],
    *,
    split: str,
    data_root: str | Path,
    catalog: dict[str, dict[str, Any]],
    asr_engine: Any,
    llm_backend: Any,
    cfg: ExperimentConfig,
    raw_cfg: dict[str, Any] | None = None,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Run strictly isolated batched E2E inference starting timing before chunk audio decode."""
    if not chunk:
        return []
    t_batch_start = time.perf_counter()
    asr_raw = (raw_cfg or {}).get("asr") or {}
    nbest_beams = int(asr_raw.get("nbest_beams", getattr(cfg.asr, "nbest_beams", 1)))
    nbest_max_delta = float(asr_raw.get("nbest_max_delta", getattr(cfg.asr, "nbest_max_delta", 0.65)))
    asr_batch_size = max(1, int(getattr(cfg.asr, "batch_size", 8)))

    envs = [SafeEnv(split, sample.id, root=data_root) for sample in chunk]
    try:
        audio_results = _collect_batch_audio(
            chunk,
            split=split,
            data_root=data_root,
            asr_engine=asr_engine,
            asr_batch_size=asr_batch_size,
            nbest_beams=nbest_beams,
            nbest_max_delta=nbest_max_delta,
        )
        items = [
            BatchReactItem(
                sample=sample,
                env=env,
                catalog=catalog,
                current_text=curr_text,
                history_texts=history_texts,
                asr_phonetic_diff=phonetic_diff,
                t_start=t_batch_start,
            )
            for sample, env, (curr_text, history_texts, phonetic_diff) in zip(chunk, envs, audio_results)
        ]
        react_results = run_react_batch(
            items,
            llm_backend,
            prompt_style=cfg.agent.prompt_style,
            max_steps=cfg.agent.max_steps,
            max_repairs=cfg.agent.max_repairs,
            timeout_s=cfg.agent.timeout_s,
            max_prompt_tokens=cfg.agent.max_prompt_tokens,
            max_result_chars=cfg.agent.max_result_chars,
            guardrails=cfg.agent.guardrails,
        )
        return [
            {
                "id": res.id,
                "tool_calls": res.tool_calls,
                "state_hash": res.state_hash,
                "time": int(round(res.latency_s * 1000)),
            }
            for res in react_results
        ]
    except Exception:
        if strict:
            raise
        traceback.print_exc(file=sys.stderr)
        return [
            infer_sample(
                sample,
                split=split,
                data_root=data_root,
                catalog=catalog,
                asr_engine=asr_engine,
                llm_backend=llm_backend,
                cfg=cfg,
                raw_cfg=raw_cfg,
                strict=False,
            )
            for sample in chunk
        ]


def _count_sample_audios(sample: Sample) -> int:
    """Count current + history user audio turns that will be transcribed for a sample."""
    return 1 + sum(1 for t in sample.history if t.role == "user" and t.audio)


def build_audio_capped_chunks(
    samples: list[Sample],
    max_batch_size: int,
    max_chunk_audios: int = 8,
) -> list[list[Sample]]:
    """Group contiguous samples into order-preserving batches capped by audio budget."""
    if max_batch_size <= 1:
        return [[s] for s in samples]
    chunks: list[list[Sample]] = []
    cur: list[Sample] = []
    cur_audios = 0
    for s in samples:
        n_aud = _count_sample_audios(s)
        if cur and (len(cur) >= max_batch_size or (cur_audios + n_aud) > max_chunk_audios):
            chunks.append(cur)
            cur = [s]
            cur_audios = n_aud
        else:
            cur.append(s)
            cur_audios += n_aud
    if cur:
        chunks.append(cur)
    return chunks


def run_inference(
    *,
    data_root: str | Path = "/data",
    output_path: str | Path = "/result/submission.jsonl",
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    split_override: str | None = None,
    limit: int | None = None,
    asr_engine: Any = None,
    llm_backend: Any = None,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Run full CLI submission pipeline mirroring predict_notebook.ipynb step-by-step."""
    t_load_start = time.perf_counter()
    cfg, raw_cfg, asr_engine, llm_backend = load_models(
        config_path,
        asr_engine=asr_engine,
        llm_backend=llm_backend,
    )
    load_ms = (time.perf_counter() - t_load_start) * 1000.0
    print(f"load model: {load_ms:.2f} ms", flush=True)

    split = discover_split(data_root, split_override=split_override)
    import_zaic_tools(data_root, split)
    catalog = load_catalog(split, data_root)
    samples = load_split(split, data_root)
    if limit is not None:
        samples = samples[:limit]

    seed_everything(cfg.seed)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    batch_size = max(1, int(getattr(cfg.agent, "batch_size", 1)))

    t_pred_start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with open(out_file, "w", encoding="utf-8", newline="\n") as fout:
        if batch_size == 1:
            for idx, sample in enumerate(samples, 1):
                row = infer_sample(
                    sample,
                    split=split,
                    data_root=data_root,
                    catalog=catalog,
                    asr_engine=asr_engine,
                    llm_backend=llm_backend,
                    cfg=cfg,
                    raw_cfg=raw_cfg,
                    strict=strict,
                )
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                rows.append(row)
                if idx % 50 == 0 or idx == len(samples):
                    so_far_ms = (time.perf_counter() - t_pred_start) * 1000.0
                    print(
                        f"[{idx}/{len(samples)}] elapsed={so_far_ms / 1000.0:.1f}s "
                        f"avg={so_far_ms / idx:.1f} ms/mẫu last_id={sample.id}",
                        flush=True,
                    )
        else:
            for chunk in build_audio_capped_chunks(samples, max_batch_size=batch_size, max_chunk_audios=8):
                chunk_rows = infer_batch(
                    chunk,
                    split=split,
                    data_root=data_root,
                    catalog=catalog,
                    asr_engine=asr_engine,
                    llm_backend=llm_backend,
                    cfg=cfg,
                    raw_cfg=raw_cfg,
                    strict=strict,
                )
                for row in chunk_rows:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                rows.extend(chunk_rows)
                done_n = len(rows)
                if done_n % 40 < batch_size or done_n == len(samples):
                    so_far_ms = (time.perf_counter() - t_pred_start) * 1000.0
                    print(
                        f"[{done_n}/{len(samples)}] elapsed={so_far_ms / 1000.0:.1f}s "
                        f"avg={so_far_ms / done_n:.1f} ms/mẫu last_id={chunk[-1].id}",
                        flush=True,
                    )

    pred_ms = (time.perf_counter() - t_pred_start) * 1000.0
    avg_ms = pred_ms / max(1, len(rows))
    print(f"predict: {len(rows)} mẫu, {pred_ms:.2f} ms, trung bình {avg_ms:.2f} ms/mẫu", flush=True)
    return rows
