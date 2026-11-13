"""Game state as a struct of arrays (spec §4).

Drone arrays have length 2n (blue rows 0..n-1, red rows n..2n-1); base arrays
have length 2B laid out the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

if TYPE_CHECKING:
    from .config import Config

BLUE, RED = 0, 1


@dataclass(eq=False)
class State:
    pos: np.ndarray           # (2n, 2) float64
    heading: np.ndarray       # (2n,) float64, wrapped to [-pi, pi)
    alive: np.ndarray         # (2n,) bool
    weapon_range: np.ndarray  # (2n,) float64
    weapon_cone: np.ndarray   # (2n,) float64
    base_pos: np.ndarray      # (2B, 2) float64
    base_hp: np.ndarray       # (2B,) int64; a base is alive iff hp > 0
    base_value: np.ndarray    # (2B,) float64
    t: int = 0                # bookkeeping only; policies must not read it
    terminated: bool = False

    ARRAY_FIELDS: ClassVar[tuple[str, ...]] = (
        "pos", "heading", "alive", "weapon_range", "weapon_cone",
        "base_pos", "base_hp", "base_value",
    )

    @property
    def n(self) -> int:
        """Drones per team."""
        return self.pos.shape[0] // 2

    @property
    def B(self) -> int:
        """Bases per team."""
        return self.base_pos.shape[0] // 2

    def copy(self) -> "State":
        """Deep copy of every array."""
        return State(*(getattr(self, f).copy() for f in self.ARRAY_FIELDS),
                     t=self.t, terminated=self.terminated)

    # Team / index helpers -------------------------------------------------
    def drone_team(self, i: int) -> int:
        return i // self.n

    def base_team(self, b: int) -> int:
        return b // self.B

    def drone_slice(self, team: int) -> slice:
        return slice(team * self.n, (team + 1) * self.n)

    def base_slice(self, team: int) -> slice:
        return slice(team * self.B, (team + 1) * self.B)

    def alive_drones(self, team: int) -> int:
        return int(self.alive[self.drone_slice(team)].sum())

    def alive_bases(self, team: int) -> int:
        return int((self.base_hp[self.base_slice(team)] > 0).sum())


def make_state(
    cfg: "Config",
    pos,
    heading,
    base_pos,
    *,
    alive=None,
    base_hp=None,
    base_value=None,
    t: int = 0,
) -> State:
    """Build a State from explicit arrays (handy for tests and scripted scenarios).

    Weapon arrays come from `cfg`; base HP defaults to `cfg.base_hp` and base
    values to `cfg.base_values` for both teams.
    """
    from .geometry import wrap

    N, B2 = cfg.n_total, 2 * cfg.bases_per_team
    pos = np.array(pos, dtype=np.float64).reshape(N, 2)
    heading = wrap(np.array(heading, dtype=np.float64).reshape(N))
    base_pos = np.array(base_pos, dtype=np.float64).reshape(B2, 2)
    alive = np.ones(N, bool) if alive is None else np.array(alive, dtype=bool).reshape(N)
    base_hp = (np.full(B2, cfg.base_hp, np.int64) if base_hp is None
               else np.array(base_hp, dtype=np.int64).reshape(B2))
    if base_value is None:
        base_value = np.tile(np.array(cfg.base_values, dtype=np.float64), 2)
    else:
        base_value = np.array(base_value, dtype=np.float64).reshape(B2)
    return State(pos, heading, alive, cfg.weapon_range_array(), cfg.weapon_cone_array(),
                 base_pos, base_hp, base_value, t=t, terminated=False)


def states_equal(a: State, b: State, atol: float = 0.0) -> bool:
    """Exact equality of discrete fields; float fields within `atol`.

    Headings are compared as angles (so -pi and pi - 1e-16 are close).
    """
    from .geometry import wrap

    if a.t != b.t or a.terminated != b.terminated:
        return False
    for f in ("alive", "base_hp"):
        if not np.array_equal(getattr(a, f), getattr(b, f)):
            return False
    for f in ("pos", "weapon_range", "weapon_cone", "base_pos", "base_value"):
        x, y = getattr(a, f), getattr(b, f)
        if x.shape != y.shape or not np.all(np.abs(x - y) <= atol):
            return False
    if a.heading.shape != b.heading.shape:
        return False
    return bool(np.all(np.abs(wrap(a.heading - b.heading)) <= atol))
