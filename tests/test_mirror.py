"""Spec §13 test 9: mirror equivariance with injected uniforms; involution; action round trip.

Positions in [0, W] make x -> W - x inexact in floating point at the ulp level,
so float fields are compared within 1e-9 while every discrete outcome (alive,
base HP, termination, shots, hits) must match exactly.
"""

import numpy as np
import pytest

from conftest import DENSE
from drone_battle.actions import legal_action_array
from drone_battle.config import Config
from drone_battle.env import _step_with_uniforms, observe, reset
from drone_battle.geometry import PI, wrap
from drone_battle.mirror import (mirror_action, mirror_state, mirror_team_action, swap_blocks)
from drone_battle.state import states_equal

ATOL = 1e-9

CONFIGS = {
    "dense": Config(n_drones=3, drone_value=0.1, **DENSE),
    "dense_exclusive_4dirs": Config(n_drones=2, move_dirs=4, move_turn_mode="exclusive", **DENSE),
    "hetero_weapons": Config(n_drones=2, weapon_range=(3.0, 5.0, 4.0, 6.0),
                             weapon_cone=(PI / 2, PI, PI / 3, 2 * PI), **DENSE),
    "default": Config(),
}


def random_state(cfg, rng):
    """A random, possibly unreachable but valid state: anywhere in the arena, some drones dead,
    some bases damaged, a few drones exactly on the walls."""
    s = reset(cfg, rng)
    N, B2 = cfg.n_total, 2 * cfg.bases_per_team
    s.pos = rng.uniform((0, 0), (cfg.width, cfg.height), size=(N, 2))
    on_wall = rng.random(N) < 0.15
    s.pos[on_wall, 0] = rng.choice([0.0, cfg.width], size=on_wall.sum())
    s.heading = rng.uniform(-PI, PI, size=N)
    s.alive = rng.random(N) < 0.85
    s.alive[rng.integers(cfg.n_drones)] = True               # keep both teams in the game
    s.alive[cfg.n_drones + rng.integers(cfg.n_drones)] = True
    s.base_hp = rng.integers(0, cfg.base_hp + 1, size=B2)
    s.base_hp[rng.integers(cfg.bases_per_team)] = cfg.base_hp
    s.base_hp[cfg.bases_per_team + rng.integers(cfg.bases_per_team)] = cfg.base_hp
    return s


def random_actions(cfg, rng, alive):
    legal = legal_action_array(cfg)
    a = legal[rng.integers(len(legal), size=cfg.n_total)].copy()
    a[~alive] = rng.integers(-3, 12, size=(int((~alive).sum()), 2))   # junk for dead drones
    return a


def mirror_shots(shots, n, B):
    out = []
    for i, kind, j, p, h in shots:
        jj = (j + n) % (2 * n) if kind == "drone" else (j + B) % (2 * B)
        out.append(((i + n) % (2 * n), kind, jj, p, h))
    return sorted(out)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_step_equivariance(name):
    cfg = CONFIGS[name]
    rng = np.random.default_rng(2024)
    n, B = cfg.n_drones, cfg.bases_per_team
    shots_seen = kills_seen = 0
    for _ in range(400):
        s = random_state(cfg, rng)
        a = random_actions(cfg, rng, s.alive)
        u = rng.random(cfg.n_total)

        s1, r1, te1, tr1, i1 = _step_with_uniforms(cfg, s, a, u)
        s2, r2, te2, tr2, i2 = _step_with_uniforms(cfg, mirror_state(s, cfg), mirror_action(a, cfg),
                                                   swap_blocks(u, n))
        assert states_equal(mirror_state(s1, cfg), s2, atol=ATOL)
        assert r2 == pytest.approx(-r1, abs=1e-12)
        assert (te1, tr1) == (te2, tr2)

        np.testing.assert_array_equal(swap_blocks(i1["drones_killed"], n), i2["drones_killed"])
        np.testing.assert_array_equal(swap_blocks(i1["bases_hit"], B), i2["bases_hit"])
        np.testing.assert_array_equal(swap_blocks(i1["bases_destroyed"], B), i2["bases_destroyed"])
        np.testing.assert_array_equal(swap_blocks(i1["moved"], n), i2["moved"])
        d1 = swap_blocks(i1["disp"], n) * np.array([-1.0, 1.0])
        np.testing.assert_allclose(d1, i2["disp"], atol=ATOL)

        m1, m2 = mirror_shots(i1["shots"], n, B), sorted(i2["shots"])
        assert [(i, k, j, h) for i, k, j, _, h in m1] == [(i, k, j, h) for i, k, j, _, h in m2]
        np.testing.assert_allclose([p for *_, p, _ in m1], [p for *_, p, _ in m2], atol=1e-12)
        shots_seen += len(m1)
        kills_seen += int(i1["drones_killed"].sum())
    if name != "default":  # make sure the dense scenarios actually exercise combat
        assert shots_seen > 200 and kills_seen > 20


@pytest.mark.parametrize("name", list(CONFIGS))
def test_mirror_is_an_involution(name):
    cfg = CONFIGS[name]
    rng = np.random.default_rng(7)
    for _ in range(200):
        s = random_state(cfg, rng)
        assert states_equal(mirror_state(mirror_state(s, cfg), cfg), s, atol=ATOL)


def test_mirror_does_not_mutate_or_alias():
    cfg = CONFIGS["dense"]
    s = random_state(cfg, np.random.default_rng(0))
    before = s.copy()
    m = mirror_state(s, cfg)
    assert states_equal(s, before)
    for f in s.ARRAY_FIELDS:
        assert not np.shares_memory(getattr(m, f), getattr(s, f))


@pytest.mark.parametrize("move_dirs", [4, 8])
def test_action_mapping(move_dirs):
    cfg = Config(move_dirs=move_dirs)
    D = move_dirs
    a = np.array([[k, t] for k in range(D + 1) for t in range(3)])
    m = mirror_team_action(a, cfg)
    np.testing.assert_array_equal(mirror_team_action(m, cfg), a)
    # hover stays hover; east <-> west; north and south are fixed; turns flip.
    lookup = {tuple(x): tuple(y) for x, y in zip(a.tolist(), m.tolist())}
    assert lookup[(0, 0)] == (0, 2) and lookup[(0, 1)] == (0, 1)
    east, north, west, south = 1, 1 + D // 4, 1 + D // 2, 1 + 3 * D // 4
    assert lookup[(east, 1)] == (west, 1) and lookup[(north, 1)] == (north, 1)
    assert lookup[(south, 1)] == (south, 1)
    if D == 8:
        assert lookup[(2, 1)] == (4, 1)  # NE <-> NW
        assert lookup[(8, 1)] == (6, 1)  # SE <-> SW
    # Moves mirror geometrically: direction angle theta -> pi - theta.
    for k in range(1, D + 1):
        th = (k - 1) * 2 * PI / D
        th2 = (lookup[(k, 1)][0] - 1) * 2 * PI / D
        assert abs(float(wrap(th2 - (PI - th)))) < 1e-12
    joint = np.concatenate([a[:6], a[6:12]])
    np.testing.assert_array_equal(mirror_action(mirror_action(joint, cfg), cfg), joint)


def test_observe():
    cfg = CONFIGS["dense"]
    s = random_state(cfg, np.random.default_rng(3))
    assert observe(s, 0, cfg) is s
    assert states_equal(observe(s, 1, cfg), mirror_state(s, cfg))
    red_view = observe(s, 1, cfg)   # red sees its own drones as rows 0..n-1, reflected
    np.testing.assert_allclose(red_view.pos[:cfg.n_drones, 0], cfg.width - s.pos[cfg.n_drones:, 0])
    np.testing.assert_array_equal(red_view.pos[:cfg.n_drones, 1], s.pos[cfg.n_drones:, 1])
    with pytest.raises(ValueError):
        observe(s, 2, cfg)
