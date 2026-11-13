"""Spec §13 tests 11-12: step() purity and determinism."""

import numpy as np
import pytest

from conftest import DENSE
from drone_battle.config import Config
from drone_battle.env import _step_with_uniforms, step
from drone_battle.match import game_seeds, play_episode
from drone_battle.policies import POLICIES
from drone_battle.state import states_equal
from test_mirror import random_actions, random_state


@pytest.mark.parametrize("cfg", [Config(n_drones=3, **DENSE), Config()], ids=["dense", "default"])
def test_step_does_not_mutate_or_alias_input(cfg):
    rng = np.random.default_rng(0)
    for _ in range(300):
        s = random_state(cfg, rng)
        a = random_actions(cfg, rng, s.alive)
        a_before, s_before = a.copy(), s.copy()
        u = rng.random(cfg.n_total)
        s1, _, _, _, info = _step_with_uniforms(cfg, s, a, u)
        assert states_equal(s, s_before)
        np.testing.assert_array_equal(a, a_before)
        for f in s.ARRAY_FIELDS:
            assert not np.shares_memory(getattr(s1, f), getattr(s, f)), f
            for v in info.values():
                if isinstance(v, np.ndarray):
                    assert not np.shares_memory(v, getattr(s, f))


def test_copy_is_deep():
    cfg = Config()
    s = random_state(cfg, np.random.default_rng(1))
    c = s.copy()
    assert states_equal(s, c)
    for f in s.ARRAY_FIELDS:
        assert not np.shares_memory(getattr(s, f), getattr(c, f))


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_policies_do_not_mutate_observation(name):
    cfg = Config(n_drones=3, **DENSE)
    rng = np.random.default_rng(2)
    for _ in range(100):
        s = random_state(cfg, rng)
        before = s.copy()
        POLICIES[name](s, cfg, rng)
        assert states_equal(s, before)


@pytest.mark.parametrize("pair", [("rusher", "hunter"), ("random", "defender")])
def test_same_seed_same_trajectory(pair):
    cfg = Config(n_drones=3, max_steps=120, **DENSE)
    blue, red = (POLICIES[p] for p in pair)
    for seed in game_seeds(5, 10):
        a = play_episode(cfg, blue, red, seed, record=True)
        b = play_episode(cfg, blue, red, seed, record=True)
        assert a.blue_return == b.blue_return and a.length == b.length and a.cause == b.cause
        assert len(a.trajectory) == len(b.trajectory)
        for (sa, ia), (sb, ib) in zip(a.trajectory, b.trajectory):
            assert states_equal(sa, sb)  # exact
            if ia is not None:
                assert ia["shots"] == ib["shots"]


def test_different_seeds_differ():
    cfg = Config()
    a = play_episode(cfg, POLICIES["random"], POLICIES["random"], 1, record=True)
    b = play_episode(cfg, POLICIES["random"], POLICIES["random"], 2, record=True)
    assert not states_equal(a.trajectory[0][0], b.trajectory[0][0])


def test_step_matches_injected_uniforms():
    cfg = Config(n_drones=3, **DENSE)
    rng = np.random.default_rng(3)
    s = random_state(cfg, rng)
    a = random_actions(cfg, rng, s.alive)
    r1, r2 = np.random.default_rng(99), np.random.default_rng(99)
    s1, rew1, *_ = step(cfg, s, a, r1)
    s2, rew2, *_ = _step_with_uniforms(cfg, s, a, r2.random(cfg.n_total))
    assert states_equal(s1, s2) and rew1 == rew2
