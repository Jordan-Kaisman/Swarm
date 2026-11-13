"""Spec §13 test 5: two hovering drones facing each other at distance d."""

import math

import numpy as np
import pytest

from conftest import hover, one_v_one, p_static, scenario
from drone_battle.env import step
from drone_battle.geometry import PI

TRIALS = 20_000


@pytest.mark.slow
@pytest.mark.parametrize("d", [1.0, 2.0, 3.5])
def test_duel_kill_rates(d):
    cfg = one_v_one()
    s0 = scenario(cfg, [[15 - d / 2, 10], [15 + d / 2, 10]], [0.0, PI])
    acts = hover(cfg)
    rng = np.random.default_rng(int(d * 1000))
    blue_dead = red_dead = both = 0
    p = p_static(cfg, d)
    for _ in range(TRIALS):
        s1, r, term, trunc, info = step(cfg, s0, acts, rng)
        b, rd = not s1.alive[0], not s1.alive[1]
        blue_dead += b
        red_dead += rd
        both += b and rd
        # Each drone fires exactly one shot per step, at the other drone, with P(d).
        assert len(info["shots"]) == 2
        assert all(kind == "drone" and P == pytest.approx(p) for _, kind, _, P, _ in info["shots"])
        # Rewards: one-sided elimination pays the survivor the loser's alive base value (1)
        # plus drone_value for the kill.
        win = 1.0 + cfg.drone_value
        assert r == pytest.approx(0.0 if b == rd else (win if rd else -win), abs=1e-12)
        assert term == (b or rd)

    for rate in (blue_dead / TRIALS, red_dead / TRIALS):
        assert abs(rate - p) <= 3 * math.sqrt(p * (1 - p) / TRIALS)
    p2 = p * p
    assert abs(both / TRIALS - p2) <= 3 * math.sqrt(p2 * (1 - p2) / TRIALS)


def test_duel_out_of_range_never_fires():
    cfg = one_v_one()
    d = cfg.weapon_range + 0.01
    s0 = scenario(cfg, [[15 - d / 2, 10], [15 + d / 2, 10]], [0.0, PI])
    rng = np.random.default_rng(0)
    for _ in range(200):
        s1, r, term, _, info = step(cfg, s0, hover(cfg), rng)
        assert info["shots"] == [] and s1.alive.all() and r == 0.0 and not term
