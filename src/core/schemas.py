"""Typed views of the competition jsonl records."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

ASK_MISSING_INFO = "sys.ask_missing_info"


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Turn(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    audio: str | None = None
    audio_text: str | None = None
    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[Any] = Field(default_factory=list)


class Label(BaseModel):
    model_config = ConfigDict(extra="allow")
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[Any] = Field(default_factory=list)
    final_state: dict[str, Any] = Field(default_factory=dict)


class Sample(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    audio: str
    context: dict[str, Any]
    history: list[Turn] = Field(default_factory=list)
    audio_text: str | None = None
    label: Label | None = None

    @property
    def allowed_tools(self) -> set[str]:
        apps = self.context.get("installed_apps", {})
        return {t for app in apps.values() for t in app.get("tools", [])} | {ASK_MISSING_INFO}
