"""Spec §13 tests 6-7: bases, rewards, termination and truncation."""

import numpy as np
import pytest

from conftest import DENSE, hover, one_v_one, scenario
from drone_battle.config import Config
from drone_battle.env import _step_with_uniforms, step
from drone_battle.geometry import PI
from drone_battle.match import game_seeds, play_episode
from drone_battle.policies import POLICIES

HIT, MISS = 0.0, 0.999999


def two_bases(**kw):
    """1 drone per team; blue bases at (3, 5) value 1 and (3, 15) value 2; red mirrored at x = 27."""
    return one_v_one(bases_per_team=2, base_values=(1.0, 2.0), base_layout=((3.0, 5.0), (3.0, 15.0)), **kw)


# ------------------------------------------------------------------ 6. base
@pytest.mark.parametrize("base_hp", [1, 3, 5])
def test_base_destroyed_on_step_base_hp(base_hp):
    cfg = one_v_one(base_hp=base_hp, base_values=(2.5,))
    # Blue drone 3.5 west of red's base (28, 10), facing it. Red drone parked far away, facing north.
    s = scenario(cfg, [[24.5, 10], [16, 18]], [0.0, PI / 2])
    rng = np.random.default_rng(0)
    for k in range(1, base_hp + 1):
        s, r, term, trunc, info = step(cfg, s, hover(cfg), rng)
        assert info["shots"] == [(0, "base", 1, 1.0, True)]
        assert s.base_hp[1] == base_hp - k and s.base_hp[0] == base_hp
        if k < base_hp:
            assert r == 0.0 and not term
        else:
            assert r == 2.5 and term and info["bases_destroyed"].tolist() == [False, True]
    with pytest.raises(RuntimeError):
        step(cfg, s, hover(cfg), rng)


def test_drones_first_targeting():
    cfg = one_v_one()
    # Blue drone with both the red base (d=3.5) and the red drone (d=3) in its cone: it shoots the drone.
    s = scenario(cfg, [[24.5, 10], [27.5, 10.0]], [0.0, PI / 2])
    _, _, _, _, info = _step_with_uniforms(cfg, s, hover(cfg), np.array([MISS, MISS]))
    assert info["shots"][0][:3] == (0, "drone", 1)


def test_ties_break_by_distance_then_index():
    cfg = one_v_one(n_drones=3, hit_pmax=0.9)
    # Two red drones very close to blue drone 0 (both capped at pmax): the nearer one is chosen.
    pos = [[10, 10], [10, 2], [10, 18], [10.6, 10.0], [10.4, 10.0], [28, 1]]
    s = scenario(cfg, pos, [0.0, PI / 2, -PI / 2, PI / 2, PI / 2, PI])
    _, _, _, _, info = _step_with_uniforms(cfg, s, hover(cfg), np.full(6, MISS))
    shot = [sh for sh in info["shots"] if sh[0] == 0][0]
    assert shot[1:3] == ("drone", 4) and shot[3] == pytest.approx(0.9)
    # Equal distance: the lower index wins.
    pos[3], pos[4] = [10.5, 10.2], [10.5, 9.8]
    s = scenario(cfg, pos, [0.0, PI / 2, -PI / 2, PI / 2, PI / 2, PI])
    _, _, _, _, info = _step_with_uniforms(cfg, s, hover(cfg), np.full(6, MISS))
    assert [sh for sh in info["shots"] if sh[0] == 0][0][1:3] == ("drone", 3)


def test_base_ties_break_by_distance():
    cfg = two_bases(weapon_range=12.0, weapon_cone=2 * PI)
    # Blue drone between red bases (27, 5) and (27, 15), slightly closer to the second.
    # The red drone sits 12.4 away, just out of range of everything.
    s = scenario(cfg, [[24, 10.5], [16, 1]], [0.0, PI / 2])
    _, _, _, _, info = _step_with_uniforms(cfg, s, hover(cfg), np.array([MISS, MISS]))
    assert info["shots"][0][1:3] == ("base", 3)


# ----------------------------------------------- 7. rewards and termination
def test_elimination_bonus_blue_wins():
    cfg = two_bases()
    # Blue faces red at d=2; red faces north (blue is out of its cone).
    s = scenario(cfg, [[14, 10], [16, 10]], [0.0, PI / 2])
    s1, r, term, trunc, info = _step_with_uniforms(cfg, s, hover(cfg), np.array([HIT, HIT]))
    assert info["drones_killed"].tolist() == [False, True]
    # Bonus: red's alive bases (1 + 2), plus drone_value for the kill.
    assert r == pytest.approx(3.0 + cfg.drone_value) and term and not trunc and s1.terminated
    # With one red base already destroyed, the bonus covers only the alive one.
    s = scenario(cfg, [[14, 10], [16, 10]], [0.0, PI / 2], base_hp=[5, 5, 0, 5])
    _, r, term, _, _ = _step_with_uniforms(cfg, s, hover(cfg), np.array([HIT, HIT]))
    assert r == pytest.approx(2.0 + cfg.drone_value) and term


def test_elimination_bonus_red_wins():
    cfg = two_bases()
    s = scenario(cfg, [[14, 10], [16, 10]], [PI / 2, PI])
    _, r, term, _, info = _step_with_uniforms(cfg, s, hover(cfg), np.array([HIT, HIT]))
    assert info["drones_killed"].tolist() == [True, False]
    assert r == pytest.approx(-(3.0 + cfg.drone_value)) and term


def test_mutual_elimination_has_no_bonus():
    cfg = two_bases()
    s = scenario(cfg, [[14, 10], [16, 10]], [0.0, PI])
    s1, r, term, trunc, info = _step_with_uniforms(cfg, s, hover(cfg), np.array([HIT, HIT]))
    assert not s1.alive.any() and r == 0.0 and term and not trunc   # the drone terms cancel too
    assert len(info["shots"]) == 2  # a drone killed this step still fires this step


def test_no_hits_no_change():
    cfg = two_bases()
    s = scenario(cfg, [[14, 10], [16, 10]], [0.0, PI])
    s1, r, term, _, info = _step_with_uniforms(cfg, s, hover(cfg), np.array([MISS, MISS]))
    assert s1.alive.all() and r == 0.0 and not term and len(info["shots"]) == 2


def test_termination_when_all_bases_destroyed():
    cfg = two_bases()
    # Red's first base is already gone; blue finishes the second (value 2) from 3 units away.
    s = scenario(cfg, [[24, 15], [16, 2]], [0.0, -PI / 2], base_hp=[5, 5, 0, 1])
    s1, r, term, trunc, info = _step_with_uniforms(cfg, s, hover(cfg), np.array([MISS, MISS]))
    assert info["bases_destroyed"].tolist() == [False, False, False, True]
    assert r == 2.0 and term and not trunc and s1.base_hp[3] == 0
    assert s1.alive.all()  # no elimination bonus involved


def test_simultaneous_base_kills():
    cfg = two_bases()
    # Both teams' last bases fall in the same step: rewards net out; the game terminates.
    s = scenario(cfg, [[24, 15], [6, 15]], [0.0, PI], base_hp=[0, 1, 0, 1])
    _, r, term, _, info = _step_with_uniforms(cfg, s, hover(cfg), np.array([MISS, MISS]))
    assert info["bases_destroyed"].tolist() == [False, True, False, True]
    assert r == 0.0 and term


def test_drone_value_reward():
    cfg = two_bases(n_drones=2, drone_value=0.25)
    # Blue drone 0 kills red drone 2; everyone else is far apart and facing away.
    pos = [[14, 10], [2, 19], [16, 10], [28, 1]]
    s = scenario(cfg, pos, [0.0, PI / 2, PI / 2, -PI / 2])
    _, r, term, _, info = _step_with_uniforms(cfg, s, hover(cfg), np.array([HIT, HIT, HIT, HIT]))
    assert info["drones_killed"].tolist() == [False, False, True, False]
    assert r == 0.25 and not term


def test_truncation():
    cfg = two_bases(max_steps=3)
    s = scenario(cfg, [[8, 10], [22, 10]], [0.0, PI])
    rng = np.random.default_rng(0)
    for k in range(1, 4):
        s, r, term, trunc, _ = step(cfg, s, hover(cfg), rng)
        assert not term and not s.terminated and s.t == k
        assert trunc == (k == 3)


def test_step_on_terminated_state_raises():
    cfg = two_bases()
    s = scenario(cfg, [[14, 10], [16, 10]], [0.0, PI])
    s1, *_ = _step_with_uniforms(cfg, s, hover(cfg), np.array([HIT, HIT]))
    assert s1.terminated
    with pytest.raises(RuntimeError, match="terminated"):
        step(cfg, s1, hover(cfg), np.random.default_rng(0))
    with pytest.raises(RuntimeError, match="terminated"):
        _step_with_uniforms(cfg, s1, hover(cfg), np.array([MISS, MISS]))


def test_step_draws_exactly_2n_uniforms():
    cfg = two_bases(n_drones=3)
    s = scenario(cfg, [[8, 4], [8, 10], [8, 16], [22, 4], [22, 10], [22, 16]], [0.0] * 3 + [PI] * 3)
    a, b = np.random.default_rng(5), np.random.default_rng(5)
    step(cfg, s, hover(cfg), a)            # nobody fires
    b.random(6)
    assert a.random() == b.random()


# ------------------------------------------------- episode return identity
def implied_return(cfg, final):
    """Episode return from the final state alone: the base-value term (bases destroyed, plus the
    elimination bonus) + drone_value * (blue survivors - red survivors). Undiscounted, a reward
    per kill adds up to a terminal reward of drone_value per surviving drone."""
    n, B = final.n, final.B
    v, up = final.base_value, final.base_hp > 0
    base = v[B:][~up[B:]].sum() - v[:B][~up[:B]].sum()
    blue_left, red_left = int(final.alive[:n].sum()), int(final.alive[n:].sum())
    if cfg.elimination_bonus and (blue_left == 0) != (red_left == 0):
        base += v[B:][up[B:]].sum() if blue_left else -v[:B][up[:B]].sum()
    return base + cfg.drone_value * (blue_left - red_left)


IDENTITY_CONFIGS = {
    "default": Config(max_steps=150),
    "dense": Config(n_drones=3, drone_value=0.37, max_steps=120, **DENSE),
    "dense_no_bonus": Config(n_drones=2, drone_value=0.25, elimination_bonus=False, max_steps=120, **DENSE),
}


@pytest.mark.parametrize("name", list(IDENTITY_CONFIGS))
def test_episode_return_identity(name):
    cfg = IDENTITY_CONFIGS[name]
    names = sorted(POLICIES)    # random and every scripted policy, all ordered pairs
    seeds = iter(game_seeds(5, 2 * len(names) ** 2))
    games = kills = 0
    for b in names:
        for r in names:
            for _ in range(2):
                res = play_episode(cfg, POLICIES[b], POLICIES[r], next(seeds), record=True)
                final = res.trajectory[-1][0]
                assert res.blue_return == pytest.approx(implied_return(cfg, final), abs=1e-9)
                games += 1
                kills += int(res.drone_kills.sum())
    assert games == 2 * len(names) ** 2 and kills > games   # drone kills actually happened
