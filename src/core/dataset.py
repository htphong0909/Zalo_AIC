"""Loaders for split jsonl, tool catalog and audio paths. Never touches <split>_environment/."""
from __future__ import annotations

import json
from pathlib import Path

from src.core.schemas import Sample

FORBIDDEN_DIR_SUFFIX = "_environment"


def _split_dir(split: str, root: str | Path) -> Path:
    return Path(root) / split


def load_split(split: str, root: str | Path) -> list[Sample]:
    path = _split_dir(split, root) / f"{split}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [Sample.model_validate_json(line) for line in f if line.strip()]


def load_catalog(split: str, root: str | Path) -> dict[str, dict]:
    path = _split_dir(split, root) / f"{split}_tools_catalog.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {tool["name"]: tool for app in data["apps"] for tool in app["tools"]}


def audio_path(split: str, root: str | Path, rel: str) -> Path:
    path = _split_dir(split, root) / rel
    if any(part.endswith(FORBIDDEN_DIR_SUFFIX) for part in path.parts):
        raise PermissionError(f"access to {FORBIDDEN_DIR_SUFFIX}/ is forbidden: {path}")
    return path
