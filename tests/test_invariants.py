"""Spec §13 test 8: invariants over 1,000 random-policy episodes.

Most episodes use a small, crowded arena (drone_value != 0, base_hp = 2) so that
kills, base hits, eliminations and the elimination bonus happen often; a few use
the default config.
"""

import numpy as np
import pytest

from conftest import DENSE
from drone_battle.config import Config
from drone_battle.env import reset, step
from drone_battle.geometry import PI
from drone_battle.match import joint_action
from drone_battle.policies import random_policy


def implied_reward(cfg, prev, nxt, info):
    """Reward recomputed from the events reported in info (spec §6.7)."""
    n, B = prev.n, prev.B
    val = prev.base_value
    destroyed, killed = info["bases_destroyed"], info["drones_killed"]
    r = val[B:][destroyed[B:]].sum() - val[:B][destroyed[:B]].sum()
    r += cfg.drone_value * (killed[n:].sum() - killed[:n].sum())
    blue_up, red_up = nxt.alive[:n].any(), nxt.alive[n:].any()
    if cfg.elimination_bonus and blue_up != red_up:
        alive_b = nxt.base_hp > 0
        r += val[B:][alive_b[B:]].sum() if blue_up else -val[:B][alive_b[:B]].sum()
    return r


def check_transition(cfg, prev, nxt, r, term, trunc, info, counters):
    n, B, N = prev.n, prev.B, prev.pos.shape[0]
    # alive never flips False -> True; killed drones were alive.
    assert not np.any(~prev.alive & nxt.alive)
    np.testing.assert_array_equal(nxt.alive, prev.alive & ~info["drones_killed"])
    assert not np.any(info["drones_killed"] & ~prev.alive)
    # positions in bounds, headings wrapped, motion bounded.
    assert np.all(nxt.pos >= 0)
    assert np.all(nxt.pos[:, 0] <= cfg.width) and np.all(nxt.pos[:, 1] <= cfg.height)
    assert np.all((nxt.heading >= -PI) & (nxt.heading < PI))
    disp_len = np.hypot(*info["disp"].T)
    assert np.all(disp_len <= cfg.step_len + 1e-12)
    np.testing.assert_array_equal(info["moved"], disp_len > 1e-9)
    np.testing.assert_allclose(nxt.pos - prev.pos, info["disp"], atol=1e-12)
    assert not np.any(info["moved"] & ~prev.alive)
    # HP never increases and matches the reported hits.
    assert np.all(nxt.base_hp <= prev.base_hp) and np.all(nxt.base_hp >= 0)
    np.testing.assert_array_equal(nxt.base_hp, np.maximum(prev.base_hp - info["bases_hit"], 0))
    np.testing.assert_array_equal(info["bases_destroyed"], (prev.base_hp > 0) & (nxt.base_hp == 0))
    # Static arrays never change.
    for f in ("weapon_range", "weapon_cone", "base_pos", "base_value"):
        np.testing.assert_array_equal(getattr(nxt, f), getattr(prev, f))
    # Shots: one per shooter, from alive shooters, at alive enemies, with 0 < P <= 1.
    shooters = [s[0] for s in info["shots"]]
    assert len(shooters) == len(set(shooters))
    hits_on = np.zeros(N, int)
    base_hits = np.zeros(2 * B, int)
    for i, kind, j, p, h in info["shots"]:
        assert prev.alive[i] and 0 < p <= 1
        if kind == "drone":
            assert j // n != i // n and prev.alive[j]
            hits_on[j] += h
        else:
            assert j // B != i // n and prev.base_hp[j] > 0
            base_hits[j] += h
    np.testing.assert_array_equal(hits_on > 0, info["drones_killed"])
    np.testing.assert_array_equal(base_hits, info["bases_hit"])
    # Reward equals the sum implied by the events.
    assert r == pytest.approx(implied_reward(cfg, prev, nxt, info), abs=1e-12)
    # Termination / truncation bookkeeping.
    knocked_out = [not (nxt.alive[t * n:(t + 1) * n].any() and (nxt.base_hp[t * B:(t + 1) * B] > 0).any())
                   for t in (0, 1)]
    assert term == any(knocked_out) == nxt.terminated
    assert nxt.t == prev.t + 1
    assert trunc == ((not term) and nxt.t >= cfg.max_steps)
    counters["shots"] += len(info["shots"])
    counters["kills"] += int(info["drones_killed"].sum())
    counters["base_hits"] += int(info["bases_hit"].sum())
    counters["bonus"] += int(term and (nxt.alive[:n].any() != nxt.alive[n:].any()))


def run_episodes(cfg, episodes, seed):
    root = np.random.SeedSequence(seed)
    counters = dict(shots=0, kills=0, base_hits=0, bonus=0, steps=0, terminated=0)
    for ss in root.spawn(episodes):
        env_rng, b_rng, r_rng = (np.random.default_rng(s) for s in ss.spawn(3))
        s = reset(cfg, env_rng)
        total, implied_total = 0.0, 0.0
        while True:
            a = joint_action(cfg, s, random_policy, random_policy, b_rng, r_rng)
            nxt, r, term, trunc, info = step(cfg, s, a, env_rng)
            check_transition(cfg, s, nxt, r, term, trunc, info, counters)
            total += r
            implied_total += implied_reward(cfg, s, nxt, info)
            counters["steps"] += 1
            s = nxt
            if term or trunc:
                counters["terminated"] += term
                break
        assert total == pytest.approx(implied_total, abs=1e-9)
    return counters


@pytest.mark.slow
def test_invariants_dense_random_play():
    cfg = Config(n_drones=3, drone_value=0.1, max_steps=120, **DENSE)
    c = run_episodes(cfg, 900, seed=11)
    # The dense arena must actually exercise the mechanics being checked.
    assert c["kills"] > 1000 and c["base_hits"] > 200 and c["bonus"] > 100 and c["terminated"] > 300


@pytest.mark.slow
def test_invariants_default_config_random_play():
    cfg = Config(max_steps=150)
    run_episodes(cfg, 100, seed=12)
