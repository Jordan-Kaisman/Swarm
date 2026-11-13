import math

import numpy as np
import pytest

from drone_battle.config import Config, config_from_overrides, parse_overrides, parse_value


def test_defaults_validate():
    cfg = Config()
    assert cfg.n_total == 6
    assert cfg.weapon_range_array().shape == (6,)
    assert np.all(cfg.weapon_cone_array() == math.pi / 2)


@pytest.mark.parametrize("kw, fragment", [
    (dict(base_values=(1.0,)), "base_values"),
    (dict(turn_delta=math.pi), "turn_delta"),
    (dict(weapon_cone=0.5), "turn_delta"),
    (dict(zone_mid_max=20.0), "deployment zone"),
    (dict(zone_mid_min=5.0, zone_mid_max=4.0), "deployment zone"),
    (dict(base_min_mid=14.0), "base region"),
    (dict(base_min_wall=11.0), "base region"),
    (dict(hit_pmax=1.5), "hit_pmax"),
    (dict(alpha_target_motion=-0.1), "alpha_target_motion"),
    (dict(beta_shooter_motion=2.0), "beta_shooter_motion"),
    (dict(base_hit_prob=1.1), "base_hit_prob"),
    (dict(move_dirs=6), "move_dirs"),
    (dict(move_turn_mode="both"), "move_turn_mode"),
    (dict(fire_mode="manual"), "fire_mode"),
    (dict(placement="anywhere"), "placement"),
    (dict(base_layout=((20.0, 5.0), (3.0, 3.0))), "base_layout"),
    (dict(weapon_range=(4.0, 4.0)), "weapon_range"),
    (dict(n_drones=0), "n_drones"),
    (dict(max_steps=0), "max_steps"),
])
def test_validate_rejects(kw, fragment):
    with pytest.raises(ValueError, match=fragment):
        Config(**kw)


def test_sequences_normalized_to_tuples_and_hashable():
    cfg = Config(base_values=[1, 2], base_layout=[[3, 4], [5, 9]], weapon_range=[4.0] * 6)
    assert cfg.base_values == (1.0, 2.0)
    assert cfg.base_layout == ((3.0, 4.0), (5.0, 9.0))
    assert cfg.weapon_range == (4.0,) * 6
    hash(cfg)


def test_per_drone_weapon_arrays():
    cones = (math.pi / 2,) * 3 + (math.pi / 3,) * 3
    cfg = Config(weapon_cone=cones)
    np.testing.assert_array_equal(cfg.weapon_cone_array(), cones)


def test_parse_values():
    assert parse_value("3") == 3
    assert parse_value("2.5") == 2.5
    assert parse_value("pi/4") == pytest.approx(math.pi / 4)
    assert parse_value("False") is False
    assert parse_value("true") is True
    assert parse_value("exclusive") == "exclusive"
    assert parse_value("(1.0, 2.0)") == (1.0, 2.0)


def test_overrides():
    cfg = config_from_overrides(["n_drones=4", "turn_delta=pi/8", "move_turn_mode=exclusive"])
    assert cfg.n_drones == 4 and cfg.move_turn_mode == "exclusive"
    assert cfg.turn_delta == pytest.approx(math.pi / 8)
    with pytest.raises(ValueError, match="unknown"):
        parse_overrides(["not_a_field=1"])
    with pytest.raises(ValueError, match="key=value"):
        parse_overrides(["n_drones"])
