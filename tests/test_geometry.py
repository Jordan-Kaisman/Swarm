"""Spec §13 tests 1-3: wrap, in-cone, hit model."""

import math

import numpy as np
import pytest

from conftest import one_v_one, p_static, scenario
from drone_battle.env import hit_probability_matrix
from drone_battle.geometry import (PI, TWO_PI, abs_angle_diff, crossing_sin, drone_hit_prob,
                                   in_cone, pairwise, wrap)


def ang_close(a, b, tol=1e-12):
    return abs(float(wrap(a - b))) <= tol


# ----------------------------------------------------------------- 1. wrap
class TestWrap:
    def test_plus_minus_pi(self):
        assert wrap(PI) == -PI
        assert wrap(-PI) == -PI

    def test_multiples_of_two_pi(self):
        for k in range(-6, 7):
            assert abs(float(wrap(TWO_PI * k))) < 1e-12
        for k in (-5, -3, -1, 1, 3, 5):  # odd multiples of pi land on -pi (up to rounding)
            w = float(wrap(PI * k))
            assert -PI <= w < PI and ang_close(w, PI)

    def test_negative_inputs(self):
        assert wrap(-0.3) == -0.3
        assert float(wrap(-1.5 * PI)) == pytest.approx(0.5 * PI)
        assert float(wrap(-7.0)) == pytest.approx(-7.0 + TWO_PI)
        assert float(wrap(-100.0)) == pytest.approx(-100.0 + 16 * TWO_PI)

    def test_range_and_equivalence(self):
        x = np.linspace(-60, 60, 200_001)
        w = wrap(x)
        assert w.min() >= -PI and w.max() < PI
        np.testing.assert_allclose(np.cos(w), np.cos(x), atol=1e-9)
        np.testing.assert_allclose(np.sin(w), np.sin(x), atol=1e-9)

    def test_identity_inside_range_is_exact(self, rng):
        x = rng.uniform(-PI, PI, 10_000)
        assert np.array_equal(wrap(x), x)

    def test_tiny_negative_offset_guard(self):
        # (a + pi) mod 2pi can round up to exactly 2pi; the result must still be < pi.
        w = wrap(np.array([-PI - 1e-17, -PI - 1e-300, 3 * PI - 1e-16]))
        assert np.all(w < PI) and np.all(w >= -PI)

    def test_does_not_alias_input(self):
        x = np.array([0.1, 0.2])
        w = wrap(x)
        w[0] = 5.0
        assert x[0] == 0.1

    def test_abs_angle_diff_matches_wrap(self, rng):
        a, b = rng.uniform(-PI, PI, 100_000), rng.uniform(-PI, PI, 100_000)
        np.testing.assert_allclose(abs_angle_diff(a, b), np.abs(wrap(a - b)), atol=1e-12)
        assert abs_angle_diff(PI, -PI) == 0.0
        assert abs_angle_diff(-PI, 0.0) == pytest.approx(PI)


# --------------------------------------------------------------- 2. in-cone
def cone_lof(lx, ly, heading, cone=PI / 2):
    return bool(in_cone(lx, ly, math.hypot(lx, ly), heading, cone))


def cone_test(heading, bearing, cone=PI / 2, d=1.0):
    return cone_lof(d * math.cos(bearing), d * math.sin(bearing), heading, cone)


class TestInCone:
    @pytest.mark.parametrize("heading", [0.0, 0.3, -2.0, PI - 0.01, -PI])
    def test_on_axis(self, heading):
        assert cone_test(heading, heading)

    @pytest.mark.parametrize("heading", [0.0, 1.0, -3.0])
    @pytest.mark.parametrize("side", [-1, 1])
    def test_just_inside_and_outside(self, heading, side):
        half = PI / 4
        assert cone_test(heading, heading + side * (half - 1e-7))
        assert not cone_test(heading, heading + side * (half + 1e-7))

    def test_exactly_on_edge_counts_as_in(self):
        assert cone_lof(1.0, 1.0, 0.0)          # bearing atan2(1, 1) == pi/4 exactly
        assert cone_lof(1.0, -1.0, 0.0)
        assert cone_lof(-1.0, 1.0, PI / 2)      # 3pi/4 vs heading pi/2

    @pytest.mark.parametrize("heading", [0.0, 1.0, -PI, PI / 2])
    def test_directly_behind(self, heading):
        assert not cone_test(heading, heading + PI)

    def test_target_at_distance_zero_is_in(self):
        for lx, ly in ((0.0, 0.0), (-1e-12, 0.0), (0.0, -5e-10)):
            assert cone_lof(lx, ly, 0.0)
        assert not cone_lof(-2e-9, 0.0, 0.0)    # just beyond EPS_DIST, directly behind

    def test_across_the_pi_seam(self):
        assert cone_test(PI - 0.05, -PI + 0.05, cone=0.3)
        assert cone_test(-PI + 0.05, PI - 0.05, cone=0.3)
        assert not cone_test(PI - 0.05, -PI + 0.5, cone=0.3)

    def test_full_cone(self):
        for b in np.linspace(-PI, PI, 37):
            assert cone_test(0.4, b, cone=2 * PI)


# ------------------------------------------------------------ 3. hit model
def P_formula(cfg, d, sin_phi=0.0, moved_t=False, moved_s=False):
    return float(drone_hit_prob(d, cfg.weapon_cone, sin_phi, moved_t, moved_s, hit_c=cfg.hit_c,
                                hit_pmax=cfg.hit_pmax, alpha=cfg.alpha_target_motion,
                                beta=cfg.beta_shooter_motion))


class TestHitModel:
    cfg = one_v_one()

    @pytest.mark.parametrize("d", [0.75, 1.0, 1.7, 2.5, 4.0])
    def test_formula_at_distances(self, d):
        cfg = self.cfg
        assert P_formula(cfg, d) == pytest.approx(min(0.9, 1.0 / ((PI / 2) * d)))

    def test_pmax_cap(self):
        cfg = self.cfg
        for d in (0.0, 1e-12, 0.1, 0.5, 2 / PI / 0.9):
            assert P_formula(cfg, d) == pytest.approx(cfg.hit_pmax)
        assert P_formula(cfg, 0.8) < cfg.hit_pmax

    def test_crossing_sin(self):
        # LOF along +x; target moving along +y crosses it fully, along +x not at all.
        assert float(crossing_sin(2.0, 0.0, 2.0, 0.0, 1.0)) == pytest.approx(1.0)
        assert float(crossing_sin(2.0, 0.0, 2.0, 1.0, 0.0)) == 0.0
        assert float(crossing_sin(2.0, 0.0, 2.0, 0.0, 0.0)) == 0.0    # did not move
        r = math.sqrt(0.5)
        assert float(crossing_sin(1.0, 1.0, math.sqrt(2), 1.0, 0.0)) == pytest.approx(r)

    def test_pairwise(self):
        src = np.array([[0.0, 0.0], [1.0, 1.0]])
        dst = np.array([[3.0, 4.0], [1.0, 1.0], [0.0, -2.0]])
        lx, ly, d = pairwise(src, dst)
        np.testing.assert_array_equal(lx, [[3, 1, 0], [2, 0, -1]])
        np.testing.assert_array_equal(ly, [[4, 1, -2], [3, 0, -3]])
        np.testing.assert_allclose(d, np.hypot(lx, ly))
        assert d[0, 0] == 5.0

    def test_motion_terms(self):
        cfg = self.cfg
        p0 = P_formula(cfg, 2.0)
        assert P_formula(cfg, 2.0, sin_phi=1.0, moved_t=True) == pytest.approx(p0 * (1 - cfg.alpha_target_motion))
        assert P_formula(cfg, 2.0, sin_phi=1.0, moved_t=False) == pytest.approx(p0)
        assert P_formula(cfg, 2.0, moved_s=True) == pytest.approx(p0 * (1 - cfg.beta_shooter_motion))

    # Matrix-level checks through the environment's geometry phase.
    def _P(self, cfg, d, bearing=0.0, heading=0.0, disp=None, alive=(True, True)):
        """Blue at (12, 10) facing `heading`; red at distance d along `bearing`; red faces away."""
        tx, ty = 12 + d * math.cos(bearing), 10 + d * math.sin(bearing)
        s = scenario(cfg, [[12, 10], [tx, ty]], [heading, bearing], alive=list(alive))
        P, _ = hit_probability_matrix(cfg, s, disp)
        return P

    @pytest.mark.parametrize("d", [1.0, 2.0, 3.0, 4.0])
    def test_matrix_matches_formula_in_range(self, d):
        cfg = self.cfg
        P = self._P(cfg, d)
        assert P[0, 1] == pytest.approx(p_static(cfg, d))
        assert P[1, 0] == 0.0  # red faces away from blue
        assert P[0, 0] == P[1, 1] == 0.0  # no self/friendly fire

    def test_zero_beyond_range(self):
        cfg = self.cfg
        assert self._P(cfg, cfg.weapon_range)[0, 1] > 0
        assert self._P(cfg, cfg.weapon_range + 1e-6)[0, 1] == 0.0

    def test_zero_outside_cone(self):
        cfg = self.cfg
        assert self._P(cfg, 2.0, bearing=PI / 4 - 1e-6)[0, 1] > 0
        assert self._P(cfg, 2.0, bearing=PI / 4 + 1e-6)[0, 1] == 0.0
        assert self._P(cfg, 2.0, bearing=PI)[0, 1] == 0.0

    def test_dead_shooter_or_target(self):
        cfg = self.cfg
        assert self._P(cfg, 2.0, alive=(False, True))[0, 1] == 0.0
        assert self._P(cfg, 2.0, alive=(True, False))[0, 1] == 0.0

    def test_crossing_vs_radial_target(self):
        cfg = self.cfg
        p0 = p_static(cfg, 2.0)
        crossing = np.array([[0.0, 0.0], [0.0, 1.0]])      # red moves perpendicular to the LOF
        radial = np.array([[0.0, 0.0], [1.0, 0.0]])        # red moves straight away
        oblique = np.array([[0.0, 0.0], [math.sqrt(0.5), math.sqrt(0.5)]])
        assert self._P(cfg, 2.0, disp=crossing)[0, 1] == pytest.approx(p0 * (1 - cfg.alpha_target_motion))
        assert self._P(cfg, 2.0, disp=radial)[0, 1] == pytest.approx(p0)
        assert self._P(cfg, 2.0, disp=oblique)[0, 1] == pytest.approx(
            p0 * (1 - cfg.alpha_target_motion * math.sqrt(0.5)))

    def test_moving_shooter(self):
        cfg = self.cfg
        p0 = p_static(cfg, 2.0)
        disp = np.array([[0.0, 1.0], [0.0, 0.0]])
        assert self._P(cfg, 2.0, disp=disp)[0, 1] == pytest.approx(p0 * (1 - cfg.beta_shooter_motion))
        both = np.array([[0.0, 1.0], [0.0, 1.0]])  # parallel motion: target crosses the LOF
        assert self._P(cfg, 2.0, disp=both)[0, 1] == pytest.approx(
            p0 * (1 - cfg.alpha_target_motion) * (1 - cfg.beta_shooter_motion))

    def test_base_targets(self):
        cfg = one_v_one(base_hit_prob=0.7)
        # Blue drone 3 units west of red's base (28, 10), facing it; red drone far away.
        s = scenario(cfg, [[25, 10], [16, 2]], [0.0, -PI / 2])
        P, dist = hit_probability_matrix(cfg, s)
        assert P[0, 2 + 1] == pytest.approx(0.7)       # column N + red base index
        assert P[0, 2 + 0] == 0.0                     # own base
        s = scenario(cfg, [[25, 10], [16, 2]], [PI / 2, -PI / 2])  # facing north: base out of cone
        assert hit_probability_matrix(cfg, s)[0][0, 3] == 0.0
