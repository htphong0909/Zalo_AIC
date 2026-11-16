"""Lightweight Pre-Flight Boundary & Optional-Arg Verifier for safe tool execution."""
from __future__ import annotations

import re
from typing import Any

from src.core.schemas import ASK_MISSING_INFO, ToolCall

_PREREQ_READ_BY_TARGET: dict[tuple[str, str], str] = {
    ("notes.append", "note_id"): "notes.list",
    ("notes.update", "note_id"): "notes.list",
    ("notes.delete", "note_id"): "notes.list",
    ("smart_home.scene_run", "scene_id"): "smart_home.get_status",
    ("smart_home.light_set", "device_id"): "smart_home.get_status",
    ("smart_home.ac_set", "device_id"): "smart_home.get_status",
    ("smart_home.switch_set", "device_id"): "smart_home.get_status",
    ("alarm.update", "alarm_id"): "alarm.list",
    ("alarm.delete", "alarm_id"): "alarm.list",
    ("tasks.complete", "task_id"): "tasks.list",
    ("tasks.delete", "task_id"): "tasks.list",
    ("food_delivery.cancel_order", "order_id"): "food_delivery.order_history",
    ("flights.cancel", "booking_id"): "flights.my_bookings",
    ("banking.transfer", "beneficiary_id"): "banking.list_beneficiaries",
}

_SKIP_AUTO_PREREQ_APPS = {"contacts", "chat_ott", "phone"}

_EXPLICIT_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}|hôm\s+qua|hôm\s+kia|hôm\s+trước|"
    r"ngày\s+(?:\d+|mốt|kia|qua|mai)|mùng\s+\d+|mồng\s+\d+|tuần\s+(?:trước|sau))\b",
    re.IGNORECASE,
)


class BoundaryVerifier:
    """Pre-flight verification gate to prevent hallucinated optional parameters and bad boundaries."""

    def __init__(self, catalog: dict[str, dict[str, Any]] | None = None) -> None:
        self.catalog = catalog or {}

    def verify_optional_args(
        self,
        call: ToolCall,
        user_utterance: str,
        user_profile: dict[str, Any] | None = None,
        *,
        current_date: str | None = None,
    ) -> ToolCall:
        """Verify optional parameters against user utterance grounding.

        Removes sensitive optional parameters ONLY if there is clear evidence of model over-completion
        and zero grounding in the user utterance or user profile.
        """
        utt_low = (user_utterance or "").lower()
        cleaned_args = dict(call.arguments)

        # 1. Verification for alarm.create
        if call.name == "alarm.create":
            # If repeat was generated but user didn't mention repeating, remove it
            if "repeat" in cleaned_args:
                if not any(w in utt_low for w in ("lặp", "hàng ngày", "mỗi ngày", "hàng tuần")):
                    cleaned_args.pop("repeat", None)

        # 2. Verification for smart_home.light_set
        if call.name == "smart_home.light_set":
            if "color" in cleaned_args:
                if not any(w in utt_low for w in ("màu", "vàng", "trắng", "đỏ", "xanh", "cam")):
                    cleaned_args.pop("color", None)

        # 3. Verification for tasks.add
        if call.name == "tasks.add":
            if "priority" in cleaned_args:
                if not any(w in utt_low for w in ("ưu tiên", "khẩn", "gấp", "quan trọng")):
                    cleaned_args.pop("priority", None)

        # 4. Verification for robot_vacuum.start
        if call.name == "robot_vacuum.start":
            if "mode" in cleaned_args:
                mode_val = str(cleaned_args["mode"]).lower()
                # If mode is 'both', user must mention both vacuum and mop (or hút và lau)
                if mode_val == "both" and not any(w in utt_low for w in ("vừa hút vừa lau", "hút và lau", "cả hút cả lau", "hút với lau")):
                    cleaned_args.pop("mode", None)

        # 5. Strip redundant default today's date when user gave no explicit date (e.g. lottery.get_results)
        if call.name == "lottery.get_results" and "date" in cleaned_args and current_date:
            arg_date = str(cleaned_args["date"]).strip()[:10]
            if arg_date == str(current_date).strip()[:10] and not _EXPLICIT_DATE_RE.search(utt_low):
                cleaned_args.pop("date", None)

        return ToolCall(name=call.name, arguments=cleaned_args)

    def _infer_schema_prereq(
        self,
        target_tool: str,
        param: str,
        available_tools: set[str],
    ) -> str | None:
        """Dynamically infer a zero-required-arg read tool from x-ref metadata in catalog."""
        explicit = _PREREQ_READ_BY_TARGET.get((target_tool, param))
        if explicit and explicit in available_tools:
            return explicit
        props = ((self.catalog.get(target_tool) or {}).get("parameters") or {}).get("properties") or {}
        pspec = props.get(param)
        if not isinstance(pspec, dict):
            return None
        xref = str(pspec.get("x-ref") or "").strip()
        if "." not in xref:
            return None
        ref_app, coll = xref.split(".", 1)
        if ref_app in _SKIP_AUTO_PREREQ_APPS:
            return None
        coll_stem = coll.lower().rstrip("s")
        param_stem = param.lower().replace("_id", "")
        best_score = -1
        best_tool: str | None = None
        for tname in sorted(available_tools):
            if not tname.startswith(f"{ref_app}."):
                continue
            spec = self.catalog.get(tname) or {}
            if spec.get("kind") != "read":
                continue
            req = ((spec.get("parameters") or {}).get("required")) or []
            if req:
                continue
            action = tname.split(".", 1)[1].lower()
            score = 0
            if coll.lower() in action or (coll_stem and coll_stem in action):
                score += 4
            if param_stem and param_stem in action:
                score += 2
            if any(tok in action for tok in ("list", "get_status", "my_", "order_status", "history", "search")):
                score += 1
            if score > best_score:
                best_score = score
                best_tool = tname
        return best_tool

    def find_missing_prereq_read(
        self,
        call: ToolCall,
        *,
        seen_reads: set[tuple[str, str]],
        available_tools: set[str],
        known_ids_by_app: dict[str, set[str]] | None = None,
    ) -> ToolCall | None:
        """Return a prerequisite state-listing read call before asking for a state object ID."""
        if call.name != ASK_MISSING_INFO:
            return None
        target_tool = str(call.arguments.get("tool_name") or "").strip()
        params = call.arguments.get("params") or []
        if not isinstance(params, list):
            params = [params]
        seen_names = {name for name, _ in seen_reads}
        for param in params:
            prereq = self._infer_schema_prereq(target_tool, str(param).strip(), available_tools)
            if not prereq:
                continue
            if prereq not in seen_names:
                return ToolCall(name=prereq, arguments={})
            app = prereq.split(".", 1)[0]
            if (
                known_ids_by_app is not None
                and not known_ids_by_app.get(app)
                and (prereq, "{}") not in seen_reads
            ):
                return ToolCall(name=prereq, arguments={})
        return None

    def should_ask_clarification(
        self,
        user_utterance: str,
        retrieved_entities: list[dict[str, Any]],
        target_tool: str,
        required_id_param: str,
    ) -> bool:
        """Determine if assistant should ask clarification rather than picking an arbitrary entity."""
        if len(retrieved_entities) < 2:
            return False
        utt_low = (user_utterance or "").lower()
        labels = [
            str(e.get("name") or e.get("title") or "").strip().lower()
            for e in retrieved_entities
            if isinstance(e, dict) and str(e.get("name") or e.get("title") or "").strip()
        ]
        if len(labels) < 2 or len(set(labels)) >= len(labels):
            return False

        # If user utterance uniquely identifies one entity by ID, room, phone, or email, no clarification needed
        unique_hits = 0
        for e in retrieved_entities:
            if not isinstance(e, dict):
                continue
            attrs = [
                str(e.get(k) or "").strip().lower()
                for k in ("id", "phone", "email", "room", "floor", "relation")
                if e.get(k)
            ]
            if any(len(a) >= 2 and a in utt_low for a in attrs):
                unique_hits += 1
        if unique_hits == 1:
            return False
        return True
