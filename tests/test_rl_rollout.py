"""Test multi-turn SafeEnv rollout collection, assistant-only loss_mask, GiGPO grouping, and ARPO prefix fork."""
from types import SimpleNamespace
from src.core.dataset import load_catalog, load_split
from src.agent.rl_rollout import (
    RolloutRequest,
    build_sample_with_asr,
    compute_assistant_loss_mask_positions,
    execute_batched_rollouts,
    execute_single_rollout,
    filter_active_groups,
)


class _DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 200 + 1 for c in text]

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        parts = [f"<{m['role']}>{m.get('content', '')}</{m['role']}>" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        txt = "\n".join(parts)
        return self.encode(txt) if tokenize else txt


def test_assistant_only_mask_zeroes_tool_responses():
    tok = _DummyTokenizer()
    segs = [
        ("prompt", "sys+user"),
        ("assistant", "call_1<|im_end|>\n"),
        ("tool", "tool_resp_1"),
        ("assistant", "done<|im_end|>\n"),
    ]
    ids, mask, step_ids = compute_assistant_loss_mask_positions(tok, segs)
    assert len(ids) == len(mask) == len(step_ids)
    # tool segment tokens must strictly have mask == 0
    tool_start = len(tok.encode("sys+user")) + len(tok.encode("call_1<|im_end|>\n"))
    tool_end = tool_start + len(tok.encode("tool_resp_1"))
    assert all(m == 0 for m in mask[tool_start:tool_end])
    assert all(m == 1 for m in mask[tool_end:])
    assert step_ids[tool_end] == 1


def test_execute_single_rollout_real_safe_env_and_forced_prefix():
    catalog = load_catalog("train", "data")
    samples = load_split("train", "data")
    # Find an episode with contacts.search or simple act without mutating unknown state outside SafeEnv
    target = next(s for s in samples if s.id == "train_0001")
    gold = [c.model_dump() for c in (target.label.tool_calls if target.label else [])]
    from src.env.safe_env import SafeEnv
    env_g = SafeEnv("train", target.id, "data")
    for gc in gold:
        env_g.call(gc["name"], gc["arguments"])
    gold_hash = env_g.get_state_hash()

    scripted = [
        '<tool_call>\n<function=contacts.search>\n<parameter=query>\nLan\n</parameter>\n</function>\n</tool_call>',
        "Da xong.",
    ]
    def gen_fn(messages, step_idx, tools):
        assert tools is not None
        txt = scripted[step_idx] if step_idx < len(scripted) else "Da xong."
        return txt, [10, 11], -0.1, 0.20

    ep = execute_single_rollout(
        target,
        catalog,
        "train",
        "data",
        gen_fn,
        _DummyTokenizer(),
        gold_calls=gold,
        gold_hash=gold_hash,
        clean_sample=target,
        max_steps=4,
    )
    assert ep.steps[0].step_idx == 0
    assert ep.prompt_token_len > 0
    assert all(m == 0 for m in ep.loss_mask[: ep.prompt_token_len])
    assert ep.reward_breakdown.episode_reward <= 1.0


def test_filter_active_groups_drops_zero_variance():
    g0 = [SimpleNamespace(episode_reward=1.0), SimpleNamespace(episode_reward=1.0)]
    g1 = [SimpleNamespace(episode_reward=1.0), SimpleNamespace(episode_reward=0.2)]
    kept = filter_active_groups([g0, g1])
    assert len(kept) == 1
    assert kept[0] is g1


def test_build_sample_with_asr_injects_phonetic_diff():
    samples = load_split("train", "data")
    s = samples[0]
    noisy, diff = build_sample_with_asr(
        s,
        {"audio_text": "chuyen 500 ngay", "history_texts": {}, "asr_phonetic_diff": "ngay / ngan / nghin"},
        inject_asr=True,
    )
    assert noisy.audio_text == "chuyen 500 ngay"
    assert diff == "ngay / ngan / nghin"


def test_build_sample_with_asr_overwrites_history_audio_text_for_build_messages():
    from src.agent.history import build_messages

    samples = load_split("train", "data")
    hist_sample = next(s for s in samples if s.history and any(t.role == "user" and t.audio_text for t in s.history))
    user_idx = next(i for i, t in enumerate(hist_sample.history or []) if t.role == "user")
    noisy, _ = build_sample_with_asr(
        hist_sample,
        {
            "audio_text": "yeu cau asr moi",
            "history_texts": {str(user_idx): "noi dung lich su asr"},
            "asr_phonetic_diff": None,
        },
        inject_asr=True,
    )
    assert noisy.history is not None
    assert noisy.history[user_idx].audio_text == "noi dung lich su asr"
    assert noisy.history[user_idx].text == "noi dung lich su asr"
    msgs = build_messages(noisy, current_text=noisy.audio_text, prompt_style="qwen35")
    assert any("noi dung lich su asr" in (m.get("content") or "") for m in msgs)


def test_execute_single_rollout_uses_select_tools_and_validates_invalid_ask():
    from src.env.safe_env import SafeEnv

    catalog = load_catalog("train", "data")
    samples = load_split("train", "data")
    target = next(s for s in samples if s.id == "train_0001")
    gold = [c.model_dump() for c in (target.label.tool_calls if target.label else [])]
    env_g = SafeEnv("train", target.id, "data")
    for gc in gold:
        env_g.call(gc["name"], gc["arguments"])
    gold_hash = env_g.get_state_hash()

    seen_tool_names: list[str] = []
    # Step 0 emits invalid sys.ask_missing_info (missing 'params'); step 1 stops gracefully
    scripted = [
        '<tool_call>\n<function=sys.ask_missing_info>\n<parameter=tool_name>\nbanking.transfer\n</parameter>\n</function>\n</tool_call>',
        "Da xong.",
    ]

    def gen_fn(messages, step_idx, tools):
        nonlocal seen_tool_names
        if step_idx == 0:
            assert all(t.get("type") == "function" and "function" in t for t in tools)
            seen_tool_names = [t["function"]["name"] for t in tools]
        txt = scripted[step_idx] if step_idx < len(scripted) else "Da xong."
        return txt, [10, 11], -0.1, 0.20

    ep = execute_single_rollout(
        target,
        catalog,
        "train",
        "data",
        gen_fn,
        _DummyTokenizer(),
        gold_calls=gold,
        gold_hash=gold_hash,
        clean_sample=target,
        max_steps=3,
    )
    assert "sys.ask_missing_info" in seen_tool_names
    assert ep.reward_breakdown.invalid_steps >= 1
    assert ep.reward_breakdown.episode_reward < 1.0


def test_execute_batched_rollouts_parity_and_prefill():
    from src.env.safe_env import SafeEnv

    catalog = load_catalog("train", "data")
    samples = load_split("train", "data")
    target = next(s for s in samples if s.id == "train_0001")
    gold = [c.model_dump() for c in (target.label.tool_calls if target.label else [])]
    env_g = SafeEnv("train", target.id, "data")
    for gc in gold:
        env_g.call(gc["name"], gc["arguments"])
    gold_hash = env_g.get_state_hash()

    scripted_call = '<tool_call>\n<function=contacts.search>\n<parameter=query>\nLan\n</parameter>\n</function>\n</tool_call>'
    calls_per_step: list[int] = []

    def batch_gen(reqs):
        calls_per_step.append(len(reqs))
        out = []
        for _msgs, sidx, _tools in reqs:
            txt = scripted_call if sidx == 0 else "Da xong."
            out.append((txt, [10, 11], -0.1, 0.25))
        return out

    reqs = [
        RolloutRequest(
            sample=target,
            gold_calls=gold,
            gold_hash=gold_hash,
            clean_sample=target,
            prefilled_step0=(scripted_call, [10, 11], -0.1, 0.25),
        ),
        RolloutRequest(
            sample=target,
            gold_calls=gold,
            gold_hash=gold_hash,
            clean_sample=target,
        ),
    ]
    eps = execute_batched_rollouts(reqs, catalog, "train", "data", batch_gen, _DummyTokenizer(), max_steps=3)
    assert len(eps) == 2
    assert calls_per_step == [1, 2]
    assert eps[0].input_ids == eps[1].input_ids
    assert eps[0].loss_mask == eps[1].loss_mask
