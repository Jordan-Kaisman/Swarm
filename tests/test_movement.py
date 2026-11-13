"""Spec §13 test 4: movement, turning and action validation."""

import math

import numpy as np
import pytest

from conftest import hover, one_v_one, scenario
from drone_battle.actions import (decode, direction_table, encode, legal_actions,
                                  legal_action_array, n_actions)
from drone_battle.env import _step_with_uniforms
from drone_battle.geometry import PI

U_MISS = np.array([0.999999, 0.999999])


def step1(cfg, s, actions):
    return _step_with_uniforms(cfg, s, np.array(actions, dtype=np.int64), U_MISS)


def far_apart(cfg, blue_xy=(10.0, 10.0), headings=(0.0, PI)):
    """Blue at blue_xy, red parked far away in the opposite corner."""
    return scenario(cfg, [list(blue_xy), [28.0, 1.0]], list(headings))


@pytest.mark.parametrize("move_dirs", [4, 8])
def test_every_direction_has_step_length(move_dirs):
    cfg = one_v_one(move_dirs=move_dirs)
    for k in range(1, move_dirs + 1):
        s1, *_, info = step1(cfg, far_apart(cfg), [[k, 1], [0, 1]])
        d = s1.pos[0] - np.array([10.0, 10.0])
        assert np.hypot(*d) == pytest.approx(1.0, abs=1e-12)
        assert math.atan2(d[1], d[0]) == pytest.approx(math.remainder((k - 1) * 2 * PI / move_dirs, 2 * PI))
        assert info["moved"][0] and not info["moved"][1]


def test_diagonal_step_length_one():
    cfg = one_v_one()
    s1, *_ = step1(cfg, far_apart(cfg), [[2, 1], [0, 1]])  # north-east
    np.testing.assert_allclose(s1.pos[0], [10 + math.sqrt(0.5), 10 + math.sqrt(0.5)])
    assert np.hypot(*(s1.pos[0] - [10, 10])) == pytest.approx(1.0, abs=1e-12)


def test_direction_numbering():
    table = direction_table(one_v_one())
    np.testing.assert_array_equal(table[0], [0, 0])     # hover
    np.testing.assert_array_equal(table[1], [1, 0])     # east
    np.testing.assert_array_equal(table[3], [0, 1])     # north
    np.testing.assert_array_equal(table[5], [-1, 0])    # west
    np.testing.assert_array_equal(table[7], [0, -1])    # south


def test_wall_clipping_and_actual_displacement():
    cfg = one_v_one()
    s1, *_, info = step1(cfg, far_apart(cfg, blue_xy=(0.3, 10.0)), [[5, 1], [0, 1]])   # west
    np.testing.assert_allclose(s1.pos[0], [0.0, 10.0])
    np.testing.assert_allclose(info["disp"][0], [-0.3, 0.0])
    assert info["moved"][0]

    s1, *_, info = step1(cfg, far_apart(cfg, blue_xy=(0.0, 0.0)), [[6, 1], [0, 1]])    # south-west into corner
    np.testing.assert_array_equal(s1.pos[0], [0.0, 0.0])
    np.testing.assert_array_equal(info["disp"][0], [0.0, 0.0])
    assert not info["moved"][0]

    s1, *_, info = step1(cfg, far_apart(cfg, blue_xy=(14.5, 19.8)), [[2, 1], [0, 1]])  # north-east
    assert s1.pos[0, 1] == cfg.height
    np.testing.assert_allclose(info["disp"][0], [math.sqrt(0.5), 0.2])


def test_turning():
    cfg = one_v_one()
    s1, *_ = step1(cfg, far_apart(cfg, headings=(0.5, PI)), [[0, 2], [0, 0]])
    assert s1.heading[0] == pytest.approx(0.5 + PI / 4)
    assert s1.heading[1] == pytest.approx(-PI - PI / 4 + 2 * PI)   # pi - delta, wrapped
    s1, *_ = step1(cfg, far_apart(cfg, headings=(3 * PI / 4, 0.0)), [[0, 2], [0, 1]])
    assert s1.heading[0] == pytest.approx(-PI)                       # wraps to -pi, never +pi
    assert -PI <= s1.heading[0] < PI
    assert s1.heading[1] == 0.0                                      # no turn: unchanged exactly


def test_move_and_turn_together():
    cfg = one_v_one()
    s1, *_ = step1(cfg, far_apart(cfg), [[1, 0], [0, 1]])
    np.testing.assert_allclose(s1.pos[0], [11.0, 10.0])
    assert s1.heading[0] == pytest.approx(-PI / 4)


def test_dead_drones_do_not_move_or_turn():
    cfg = one_v_one(n_drones=2)
    s = scenario(cfg, [[10, 10], [10, 14], [28, 1], [28, 19]], [0.0, 1.0, PI, PI], alive=[True, False, True, True])
    acts = np.array([[1, 1], [3, 2], [0, 1], [0, 1]])
    s1, *_, info = _step_with_uniforms(cfg, s, acts, np.full(4, 0.999))
    np.testing.assert_array_equal(s1.pos[1], [10, 14])
    assert s1.heading[1] == 1.0
    assert not s1.alive[1] and not info["moved"][1]
    # Junk actions for dead drones are ignored rather than rejected.
    acts[1] = [99, -7]
    _step_with_uniforms(cfg, s, acts, np.full(4, 0.999))


@pytest.mark.parametrize("bad", [[9, 1], [-1, 1], [0, 3], [0, -1]])
def test_out_of_range_actions_raise(bad):
    cfg = one_v_one()
    with pytest.raises(ValueError, match="out of range"):
        step1(cfg, far_apart(cfg), [bad, [0, 1]])


def test_bad_shapes_and_dtypes_raise():
    cfg = one_v_one()
    s = far_apart(cfg)
    with pytest.raises(ValueError, match="shape"):
        _step_with_uniforms(cfg, s, np.zeros((1, 2), np.int64), U_MISS)
    with pytest.raises(TypeError, match="integer"):
        _step_with_uniforms(cfg, s, np.ones((2, 2)), U_MISS)
    with pytest.raises(ValueError, match="u must have shape"):
        _step_with_uniforms(cfg, s, hover(cfg), np.zeros(3))


def test_exclusive_mode():
    cfg = one_v_one(move_turn_mode="exclusive")
    s = far_apart(cfg)
    for ok in ([0, 0], [0, 1], [0, 2], [1, 1], [8, 1]):
        step1(cfg, s, [ok, [0, 1]])
    for illegal in ([1, 0], [3, 2]):
        with pytest.raises(ValueError, match="exclusive"):
            step1(cfg, s, [illegal, [0, 1]])
    # Illegal combos from dead drones are ignored.
    sd = scenario(cfg, [[10, 10], [28, 1]], [0, PI], alive=[True, False])
    step1(cfg, sd, [[0, 1], [1, 0]])


def test_action_space_sizes_and_encoding():
    assert n_actions(one_v_one()) == 27
    assert n_actions(one_v_one(move_turn_mode="exclusive")) == 11
    assert n_actions(one_v_one(move_dirs=4)) == 15
    assert n_actions(one_v_one(move_dirs=4, move_turn_mode="exclusive")) == 7
    excl = legal_actions(one_v_one(move_turn_mode="exclusive"))
    assert excl == [(0, 0), (0, 1), (0, 2)] + [(k, 1) for k in range(1, 9)]
    for m, t in legal_actions(one_v_one()):
        assert decode(encode(m, t)) == (m, t)
    arr = legal_action_array(one_v_one())
    np.testing.assert_array_equal(encode(arr[:, 0], arr[:, 1]), np.arange(27))
