"""Verify strict monotonic tiering, fuzzy densification, and anti-hacking of RL rewards."""
from src.core.dataset import load_catalog
from src.agent.rl_reward import (
    canonical_state_anchor,
    compute_bipartite_call_similarity,
    compute_episode_reward,
    compute_step_format_penalty,
)


def test_reward_hierarchy_strictly_monotonic():
    cat = load_catalog("train", "data")
    gold_calls = [
        {"name": "contacts.search", "arguments": {"query": "Lan"}},
        {"name": "sms.send", "arguments": {"contact_id": "c_01", "content": "Anh den tre 10 phut"}},
    ]
    gold_hash = "hash_ok"

    # 1. Perfect official score = 1.0
    r_exact = compute_episode_reward(gold_calls, gold_hash, gold_calls, gold_hash, cat)
    assert r_exact.episode_reward == 1.0
    assert r_exact.tier == "exact"

    # 2. State match = True, F1 < 1 (extra redundant read or fuzzy mismatch) strictly in [0.44, 0.94]
    redundant_calls = [
        {"name": "chat_ott.list_chats", "arguments": {}},
        *gold_calls,
    ]
    r_sm_extra = compute_episode_reward(
        redundant_calls, gold_hash, gold_calls, gold_hash, cat, extra_reads=1
    )
    r_sm_clean = compute_episode_reward(
        [{"name": "sms.send", "arguments": {"contact_id": "c_01", "content": "Anh den tre 10 phut"}}],
        gold_hash,
        gold_calls,
        gold_hash,
        cat,
        extra_reads=0,
    )
    assert 0.44 <= r_sm_extra.episode_reward < r_exact.episode_reward
    assert r_sm_extra.episode_reward <= 0.94
    assert r_sm_clean.episode_reward > r_sm_extra.episode_reward

    # 3. State match = False with good partial tools strictly below sm=True tier and above wrong tools
    r_partial = compute_episode_reward(
        [{"name": "contacts.search", "arguments": {"query": "Lan"}}, {"name": "sms.send", "arguments": {"contact_id": "c_01", "content": "Sai noi dung hoan toan xyz"}}],
        "hash_bad",
        gold_calls,
        gold_hash,
        cat,
    )
    r_wrong = compute_episode_reward(
        [{"name": "alarm.list", "arguments": {}}], "hash_bad", gold_calls, gold_hash, cat
    )
    assert -0.15 <= r_wrong.episode_reward < r_partial.episode_reward <= 0.35
    assert r_partial.episode_reward < r_sm_extra.episode_reward

    # 4. Premature false ask on act episode penalized below honest partial attempt
    r_false_ask = compute_episode_reward(
        [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "sms.send", "params": ["contact_id"]}}],
        "hash_bad",
        gold_calls,
        gold_hash,
        cat,
    )
    assert r_false_ask.episode_reward <= -0.15


def test_oos_and_ask_exact_and_partial_tiers():
    cat = load_catalog("train", "data")

    # OOS exact [] gets +1.0 (fixes ToolRL zero reward for text turns)
    r_oos_ok = compute_episode_reward([], "h0", [], "h0", cat)
    assert r_oos_ok.episode_reward == 1.0
    # OOS false ask penalized harshly
    r_oos_ask = compute_episode_reward(
        [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "alarm.create", "params": ["time"]}}],
        "h1",
        [],
        "h0",
        cat,
    )
    assert r_oos_ask.episode_reward == -0.25

    # Ask partial shaping beats missed ask
    gold_ask = [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "banking.transfer", "params": ["amount", "beneficiary_id"]}}]
    r_ask_exact = compute_episode_reward(gold_ask, "ha", gold_ask, "ha", cat)
    r_ask_part = compute_episode_reward(
        [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "banking.transfer", "params": ["amount"]}}],
        "hb",
        gold_ask,
        "ha",
        cat,
    )
    r_ask_missed = compute_episode_reward(
        [{"name": "banking.list_accounts", "arguments": {}}], "hc", gold_ask, "ha", cat
    )
    assert r_ask_exact.episode_reward == 1.0
    assert 0.10 <= r_ask_part.episode_reward <= 0.55
    assert r_ask_missed.episode_reward < 0.0


def test_fuzzy_densifier_gives_gradient_below_threshold():
    cat = load_catalog("train", "data")
    gold = [{"name": "notes.create", "arguments": {"title": "Hop nhom", "content": "Chuan bi slide quy 4 va bao cao doanh thu"}}]
    close = [{"name": "notes.create", "arguments": {"title": "Hop", "content": "Chuan bi slide quy 4"}}]
    far = [{"name": "notes.create", "arguments": {"title": "Mua sam", "content": "Di cho nau com toi"}}]
    s_close = compute_bipartite_call_similarity(close, gold, cat)
    s_far = compute_bipartite_call_similarity(far, gold, cat)
    assert s_close > s_far


def test_step_format_penalty_and_canonical_anchor():
    cat = load_catalog("train", "data")
    pen_ok, valid_ok = compute_step_format_penalty(
        '<tool_call>\n<function=alarm.list>\n</function>\n</tool_call>',
        [{"name": "alarm.list", "arguments": {}}],
        cat,
        is_last_turn=False,
    )
    assert valid_ok and pen_ok == 0.0

    pen_bad, valid_bad = compute_step_format_penalty(
        '<tool_call>broken garbage</tool_call>',
        [],
        cat,
        is_last_turn=False,
    )
    assert not valid_bad and pen_bad == -0.08

    a1 = canonical_state_anchor("train_0100", 1, [{"name": "contacts.search", "arguments": {"query": "Lan"}}], [{"b": 2, "a": 1}], False)
    a2 = canonical_state_anchor("train_0100", 1, [{"name": "contacts.search", "arguments": {"query": "Lan"}}], [{"a": 1, "b": 2}], False)
    assert a1 == a2


def test_act_with_invalid_step_cannot_tie_exact_one():
    cat = load_catalog("train", "data")
    gold = [{"name": "alarm.list", "arguments": {}}]
    clean = compute_episode_reward(gold, "h_same", gold, "h_same", cat, invalid_steps=0)
    dirty = compute_episode_reward(gold, "h_same", gold, "h_same", cat, invalid_steps=1)
    assert clean.episode_reward == 1.0
    assert clean.tier == "exact"
    assert dirty.episode_reward < 1.0
    assert dirty.tier == "sm_partial"
