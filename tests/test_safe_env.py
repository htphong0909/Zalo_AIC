from pathlib import Path

from src.core.dataset import load_split
from src.env.safe_env import SafeEnv


def test_safe_env_exposes_only_permitted_methods():
    public = {n for n in dir(SafeEnv) if not n.startswith("_")}
    assert public == {"validate", "call", "get_tool_calls", "get_state_hash"}


def test_safe_env_validate_call_log_and_hash(data_root: Path):
    sample = load_split("train", data_root)[0]
    env = SafeEnv("train", sample.id, data_root)
    assert env.validate("no_such_app.tool", {})["ok"] is False
    assert env.get_tool_calls() == []  # validate does not log
    env.call("sys.ask_missing_info", {"tool_name": "sms.send", "params": ["contact_id"]})
    assert env.get_tool_calls() == [
        {
            "name": "sys.ask_missing_info",
            "arguments": {"tool_name": "sms.send", "params": ["contact_id"]},
        }
    ]
    h = env.get_state_hash()
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)
