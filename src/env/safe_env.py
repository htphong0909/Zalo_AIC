"""Inference-time facade over zaic_tools.ToolEnv exposing only the four permitted methods."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable


def load_toolenv_factory(split: str, root: str | Path) -> Callable[..., Any]:
    """Import the split's own zaic_tools package (<root>/<split>/zaic_tools). One split per process."""
    split_dir = (Path(root) / split).resolve()
    loaded = sys.modules.get("zaic_tools")
    if loaded is not None:
        if Path(loaded.__file__).resolve().parent.parent != split_dir:
            raise RuntimeError(f"zaic_tools already loaded from {loaded.__file__}; use one split per process")
        return loaded.ToolEnv
    sys.path.insert(0, str(split_dir))
    return importlib.import_module("zaic_tools").ToolEnv


class SafeEnv:
    __slots__ = ("_env",)

    def __init__(self, split: str, sample_id: str, root: str | Path):
        self._env = load_toolenv_factory(split, root)(split, sample_id, root=str(root))

    def validate(self, name: str, arguments: dict | None = None) -> dict:
        return self._env.validate(name, arguments or {})

    def call(self, name: str, arguments: dict | None = None) -> dict:
        return self._env.call(name, arguments or {})

    def get_tool_calls(self) -> list[dict]:
        return self._env.get_tool_calls()

    def get_state_hash(self) -> str:
        return self._env.get_state_hash()
