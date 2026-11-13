"""reset(): base and drone placement constraints (spec §7)."""

import numpy as np
import pytest

from drone_battle.config import Config
from drone_battle.env import reset
from drone_battle.geometry import PI, wrap
from drone_battle.mirror import mirror_heading, mirror_points


def _min_sep(p):
    d = np.hypot(*(p[:, None] - p[None]).transpose(2, 0, 1))
    return d[np.triu_indices(len(p), 1)].min() if len(p) > 1 else np.inf


def test_zone_reset_constraints(rng):
    cfg = Config(n_drones=4)
    n, B, W, H = cfg.n_drones, cfg.bases_per_team, cfg.width, cfg.height
    for _ in range(300):
        s = reset(cfg, rng)
        blue_b, red_b = s.base_pos[:B], s.base_pos[B:]
        assert np.all(blue_b[:, 0] >= cfg.base_min_wall)
        assert np.all(blue_b[:, 0] <= W / 2 - cfg.base_min_mid)
        assert np.all((blue_b[:, 1] >= cfg.base_min_wall) & (blue_b[:, 1] <= H - cfg.base_min_wall))
        assert _min_sep(blue_b) >= cfg.base_min_sep
        np.testing.assert_allclose(red_b, mirror_points(blue_b, W))

        blue = s.pos[:n]
        assert np.all((blue[:, 0] >= W / 2 - cfg.zone_mid_max) & (blue[:, 0] <= W / 2 - cfg.zone_mid_min))
        assert np.all((blue[:, 1] >= cfg.zone_wall_margin) & (blue[:, 1] <= H - cfg.zone_wall_margin))
        assert _min_sep(blue) >= cfg.drone_min_sep
        # mirror_init: red drone i is the mirror image of blue drone i
        np.testing.assert_allclose(s.pos[n:], mirror_points(blue, W))
        assert np.all(np.abs(wrap(s.heading[n:] - mirror_heading(s.heading[:n]))) < 1e-12)

        assert np.all((s.heading >= -PI) & (s.heading < PI))
        assert s.alive.all() and np.all(s.base_hp == cfg.base_hp) and s.t == 0 and not s.terminated
        np.testing.assert_array_equal(s.base_value, cfg.base_values * 2)


def test_bases_placement_and_fixed_headings(rng):
    cfg = Config(placement="bases", random_headings=False, mirror_init=False, n_drones=5)
    n, B = cfg.n_drones, cfg.bases_per_team
    for _ in range(200):
        s = reset(cfg, rng)
        for team in (0, 1):
            drones = s.pos[team * n:(team + 1) * n]
            bases = s.base_pos[team * B:(team + 1) * B]
            assigned = bases[np.arange(n) % B]
            assert np.all(np.hypot(*(drones - assigned).T) <= cfg.bases_spawn_radius + 1e-9)
            assert _min_sep(drones) >= cfg.drone_min_sep
        assert np.all(s.heading[:n] == 0.0)
        assert np.all(s.heading[n:] == -PI)  # pi, wrapped


def test_independent_red_spawn_differs(rng):
    cfg = Config(mirror_init=False)
    s = reset(cfg, rng)
    assert not np.allclose(s.pos[cfg.n_drones:], mirror_points(s.pos[:cfg.n_drones], cfg.width))
    assert np.all(s.pos[cfg.n_drones:, 0] > cfg.width / 2)


def test_explicit_base_layout(rng):
    layout = ((3.0, 4.0), (6.0, 15.0))
    cfg = Config(base_layout=layout)
    s = reset(cfg, rng)
    np.testing.assert_array_equal(s.base_pos[:2], layout)
    np.testing.assert_array_equal(s.base_pos[2:], [[cfg.width - 3.0, 4.0], [cfg.width - 6.0, 15.0]])


def test_impossible_constraints_raise(rng):
    cfg = Config(n_drones=8, drone_min_sep=50.0)
    with pytest.raises(RuntimeError, match="10000 tries"):
        reset(cfg, rng)


def test_reset_is_deterministic():
    cfg = Config()
    a, b = reset(cfg, np.random.default_rng(7)), reset(cfg, np.random.default_rng(7))
    for f in a.ARRAY_FIELDS:
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f))
