"""APIGen-MT (NeurIPS 2025) Trajectory Critic for validating multi-turn synthetic data."""
from __future__ import annotations

from typing import Any

from src.core.schemas import ASK_MISSING_INFO
from src.env.safe_env import SafeEnv


def critique_trajectory(
    sample_id: str,
    split: str,
    data_root: str,
    tool_calls: list[dict[str, Any]],
    utterance: str,
    catalog: dict[str, dict[str, Any]],
    user_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Critique a proposed tool sequence against SafeEnv execution and semantic rules.

    Returns:
        {"passed": bool, "score": float, "state_hash": str | None, "issues": list[str]}
    """
    issues: list[str] = []

    # 1. SafeEnv validation & execution check
    try:
        env = SafeEnv(split, sample_id, data_root)
    except Exception as e:
        return {"passed": False, "score": 0.0, "state_hash": None, "issues": [f"SafeEnv init failed: {e}"]}

    for idx, tc in enumerate(tool_calls):
        name = tc.get("name", "")
        args = tc.get("arguments", {})

        # Validation check
        val = env.validate(name, args)
        if not val.get("ok", False):
            issues.append(f"Step {idx} '{name}' failed validation: {val.get('error')}")
            break

        # Execution check
        try:
            env.call(name, args)
        except Exception as e:
            issues.append(f"Step {idx} '{name}' raised execution exception: {e}")
            break

    final_hash = None
    if not issues:
        try:
            final_hash = env.get_state_hash()
        except Exception as e:
            issues.append(f"Failed to get final state hash: {e}")

    # 2. Semantic grounding check
    utt_low = (utterance or "").lower()
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("arguments", {})
        props = ((catalog.get(name) or {}).get("parameters") or {}).get("properties") or {}

        # Check ungrounded optional arguments
        for k, v in args.items():
            spec = props.get(k) or {}
            if k in ("mode", "color", "priority") and str(v).lower() not in utt_low:
                issues.append(f"Potential ungrounded parameter {k}='{v}' for {name} not mentioned in utterance")

    passed = len(issues) == 0
    score = 1.0 if passed else max(0.0, 1.0 - 0.25 * len(issues))

    return {
        "passed": passed,
        "score": score,
        "state_hash": final_hash,
        "issues": issues,
    }
