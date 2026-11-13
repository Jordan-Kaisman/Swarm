"""Angles, pairwise geometry, cone test and the drone hit model (spec §2, §6.3-6.4).

These functions are the single source of truth for the geometry; env.step()
calls them on (shooter x target) matrices. Line-of-fire vectors are passed as
separate x / y arrays (cheaper than strided (..., 2) arrays on small inputs).
"""

from __future__ import annotations

import math

import numpy as np

PI = math.pi
TWO_PI = 2.0 * math.pi
EPS_ANGLE = 1e-9   # cone-edge tolerance: a target exactly on the edge counts as in
EPS_DIST = 1e-9    # below this distance a target counts as in the cone
EPS_MOVE = 1e-9    # a drone "moved" iff its actual displacement exceeds this


def wrap(a) -> np.ndarray:
    """Wrap angle(s) to [-pi, pi) as (a + pi) mod 2pi - pi.

    Values already in [-pi, pi) are returned bit-for-bit unchanged, so
    re-wrapping a wrapped heading never introduces rounding drift. Always
    returns a new array.
    """
    a = np.asarray(a, dtype=np.float64)
    outside = (a < -PI) | (a >= PI)
    out = a.copy()
    if np.count_nonzero(outside):  # (count_nonzero is much cheaper than .any() on small arrays)
        w = np.mod(a[outside] + PI, TWO_PI) - PI
        w[w >= PI] -= TWO_PI  # guard: mod can round up to exactly 2pi
        out[outside] = w
    return out


def abs_angle_diff(a, b) -> np.ndarray:
    """|wrap(a - b)| for angles a, b already in [-pi, pi].

    Uses the identity |wrap(x)| = min(|x|, 2pi - |x|) for |x| <= 2pi, which
    avoids the modulo on the hot path. Agrees with wrap() up to rounding.
    """
    d = np.abs(np.subtract(a, b))
    return np.minimum(d, TWO_PI - d)


def pairwise(src: np.ndarray, dst: np.ndarray):
    """Line-of-fire components and distances: LOF[i, j] = dst[j] - src[i].

    src: (m, 2), dst: (k, 2) -> (lx, ly, dist), each of shape (m, k).
    """
    lx = dst[:, 0] - src[:, 0, None]
    ly = dst[:, 1] - src[:, 1, None]
    return lx, ly, np.hypot(lx, ly)


def in_cone(lx, ly, dist, heading, cone) -> np.ndarray:
    """|wrap(atan2(ly, lx) - heading)| <= cone/2 + EPS_ANGLE, or dist < EPS_DIST.

    heading and cone are the shooter's and broadcast against lx, ly, dist.
    A target exactly on the cone edge counts as in.
    """
    off_axis = abs_angle_diff(np.arctan2(ly, lx), heading)
    return (off_axis <= 0.5 * np.asarray(cone) + EPS_ANGLE) | (np.asarray(dist) < EPS_DIST)


def crossing_sin(lx, ly, dist, ux, uy) -> np.ndarray:
    """|sin phi| = |cross(unit target displacement (ux, uy), unit LOF)|.

    (ux, uy) is the zero vector for a target that did not move, giving 0.
    """
    return np.abs(ux * ly - uy * lx) / np.maximum(dist, EPS_DIST)


def drone_hit_prob(dist, cone, sin_phi, moved_target, moved_shooter, *,
                   hit_c: float, hit_pmax: float, alpha: float, beta: float):
    """P = min(pmax, c / (theta * max(d, EPS))) * (1 - alpha |sin phi| moved_j) * (1 - beta moved_i).

    All array arguments broadcast together. Range and cone masks are applied by
    the caller.
    """
    p = np.minimum(hit_pmax, hit_c / (np.multiply(cone, np.maximum(dist, EPS_DIST))))
    p = p * (1.0 - alpha * np.multiply(sin_phi, moved_target))
    return p * (1.0 - beta * np.asarray(moved_shooter, dtype=np.float64))
