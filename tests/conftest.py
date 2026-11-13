"""Shared fixtures and scenario builders."""

from __future__ import annotations

import math

import numpy as np
import pytest

from drone_battle.config import Config
from drone_battle.mirror import mirror_points
from drone_battle.state import make_state

HOVER = (0, 1)

# A small, crowded arena: random play produces many shots, kills and base hits.
DENSE = dict(width=16.0, height=10.0, base_min_mid=2.0, base_min_wall=1.0, base_min_sep=2.0,
             zone_mid_min=0.5, zone_mid_max=3.0, zone_wall_margin=1.0, drone_min_sep=1.0,
             base_hp=2)

# The original 30x20 arena (the defaults before 2026-11-13). Hand-placed scenarios use it
# so their coordinates keep their meaning; the sweep's scale grid is anchored to it too.
SMALL_ARENA = dict(width=30.0, height=20.0, base_min_mid=5.0, base_min_wall=2.0, base_min_sep=4.0,
                   zone_mid_min=3.0, zone_mid_max=8.0, zone_wall_margin=2.0)


def one_v_one(**overrides) -> Config:
    """1 drone and B bases per team at fixed spots far from the middle of the 30x20 arena."""
    kw = dict(SMALL_ARENA, n_drones=1, bases_per_team=1, base_values=(1.0,), base_layout=((2.0, 10.0),))
    kw.update(overrides)
    return Config(**kw)


def scenario(cfg: Config, pos, heading, *, alive=None, base_hp=None):
    """State with explicit drones; bases from cfg.base_layout (mirrored for red)."""
    blue = np.array(cfg.base_layout, dtype=float)
    base_pos = np.concatenate([blue, mirror_points(blue, cfg.width)])
    return make_state(cfg, pos, heading, base_pos, alive=alive, base_hp=base_hp)


def hover(cfg: Config) -> np.ndarray:
    return np.tile(np.array(HOVER, dtype=np.int64), (cfg.n_total, 1))


def p_static(cfg: Config, d: float) -> float:
    """Hit probability against a non-moving target from a non-moving shooter."""
    return min(cfg.hit_pmax, cfg.hit_c / (cfg.weapon_cone * d))


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


PI = math.pi
