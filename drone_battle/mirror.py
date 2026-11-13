"""Mirror transform / canonicalization (spec §8).

Reflection about the midline x = W/2 with the team blocks swapped, so red sees
itself as blue. Reward is negated under the mirror.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from .geometry import PI, wrap
from .state import State

if TYPE_CHECKING:
    from .config import Config


def swap_blocks(x: np.ndarray, half: int) -> np.ndarray:
    """Swap the first and second blocks of length `half` along axis 0 (returns a new array)."""
    x = np.asarray(x)
    return np.concatenate((x[half:], x[:half]))


def mirror_points(p: np.ndarray, width: float) -> np.ndarray:
    """x -> W - x for an (..., 2) array of points (returns a new array)."""
    out = np.array(p, dtype=np.float64, copy=True)
    out[..., 0] = width - out[..., 0]
    return out


def mirror_heading(h: np.ndarray) -> np.ndarray:
    """psi -> wrap(pi - psi)."""
    return wrap(PI - np.asarray(h, dtype=np.float64))


def mirror_state(state: State, cfg: "Config") -> State:
    """Reflect positions/headings about the midline and swap team blocks."""
    n, B = state.n, state.B
    return State(
        pos=mirror_points(swap_blocks(state.pos, n), cfg.width),
        heading=mirror_heading(swap_blocks(state.heading, n)),
        alive=swap_blocks(state.alive, n),
        weapon_range=swap_blocks(state.weapon_range, n),
        weapon_cone=swap_blocks(state.weapon_cone, n),
        base_pos=mirror_points(swap_blocks(state.base_pos, B), cfg.width),
        base_hp=swap_blocks(state.base_hp, B),
        base_value=swap_blocks(state.base_value, B),
        t=state.t,
        terminated=state.terminated,
    )


@lru_cache(maxsize=None)
def _move_map(move_dirs: int) -> np.ndarray:
    """mirror of move k: hover -> hover; direction j = k-1 -> (D/2 - j) mod D."""
    table = np.zeros(move_dirs + 1, dtype=np.int64)
    j = np.arange(move_dirs)
    table[1:] = (move_dirs // 2 - j) % move_dirs + 1
    table.setflags(write=False)
    return table


_TURN_MAP = np.array([2, 1, 0], dtype=np.int64)  # reflection reverses rotation sense
_TURN_MAP.setflags(write=False)


def mirror_team_action(actions: np.ndarray, cfg: "Config") -> np.ndarray:
    """Mirror (..., 2) per-drone actions without swapping teams.

    Used by the runner to map red's canonical-frame actions back to the world.
    Out-of-range values (e.g. junk for dead drones) are passed through unchanged.
    """
    a = np.asarray(actions)
    move, turn = a[..., 0], a[..., 1]
    mmap = _move_map(cfg.move_dirs)
    ok_m = (move >= 0) & (move < len(mmap))
    ok_t = (turn >= 0) & (turn < 3)
    out = np.empty(a.shape, dtype=np.int64)
    out[..., 0] = np.where(ok_m, mmap[np.where(ok_m, move, 0)], move)
    out[..., 1] = np.where(ok_t, _TURN_MAP[np.where(ok_t, turn, 0)], turn)
    return out


def mirror_action(actions: np.ndarray, cfg: "Config") -> np.ndarray:
    """Mirror a joint (2n, 2) action: swap team blocks and reflect moves/turns."""
    a = np.asarray(actions)
    return mirror_team_action(swap_blocks(a, a.shape[0] // 2), cfg)
