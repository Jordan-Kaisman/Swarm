"""Action space: direction table, flat encoding and legality (spec §5).

An action is (move, turn): move 0 = hover, k >= 1 = world-frame direction at
angle (k-1)*2pi/move_dirs; turn in {0, 1, 2} = {-delta (CW), 0, +delta (CCW)}.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import Config

HOVER = 0
TURN_CW, TURN_NONE, TURN_CCW = 0, 1, 2
N_TURNS = 3

_R = math.sqrt(0.5)
# Exact tables so that mirrored directions are exact negations in x.
_UNIT_DIRS = {
    4: ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)),
    8: ((1.0, 0.0), (_R, _R), (0.0, 1.0), (-_R, _R),
        (-1.0, 0.0), (-_R, -_R), (0.0, -1.0), (_R, -_R)),
}


def _frozen(arr: np.ndarray) -> np.ndarray:
    arr.setflags(write=False)
    return arr


@lru_cache(maxsize=None)
def unit_directions(move_dirs: int) -> np.ndarray:
    """(move_dirs, 2) unit vectors; row j is direction index j = move - 1."""
    return _frozen(np.array(_UNIT_DIRS[move_dirs], dtype=np.float64))


@lru_cache(maxsize=None)
def _direction_table(move_dirs: int, step_len: float) -> np.ndarray:
    table = np.zeros((move_dirs + 1, 2))
    table[1:] = step_len * unit_directions(move_dirs)
    return _frozen(table)


def direction_table(cfg: "Config") -> np.ndarray:
    """(move_dirs + 1, 2) displacement per move index; row 0 (hover) is zero."""
    return _direction_table(cfg.move_dirs, float(cfg.step_len))


def encode(move, turn):
    """Flat index a = move * 3 + turn (ints in, int out; array-likes in, array out)."""
    if isinstance(move, (int, np.integer)) and isinstance(turn, (int, np.integer)):
        return int(move) * N_TURNS + int(turn)
    return np.asarray(move) * N_TURNS + np.asarray(turn)


def decode(a):
    """Inverse of encode: returns (move, turn)."""
    if isinstance(a, (int, np.integer)):
        return int(a) // N_TURNS, int(a) % N_TURNS
    a = np.asarray(a)
    return a // N_TURNS, a % N_TURNS


@lru_cache(maxsize=None)
def _legal_table(move_dirs: int, mode: str) -> np.ndarray:
    legal = np.ones((move_dirs + 1, N_TURNS), dtype=bool)
    if mode == "exclusive":
        legal[1:, :] = False
        legal[1:, TURN_NONE] = True
    return _frozen(legal)


def legal_table(cfg: "Config") -> np.ndarray:
    """(move_dirs + 1, 3) bool table: legal[move, turn]."""
    return _legal_table(cfg.move_dirs, cfg.move_turn_mode)


@lru_cache(maxsize=None)
def _legal_array(move_dirs: int, mode: str) -> np.ndarray:
    moves, turns = np.nonzero(_legal_table(move_dirs, mode))
    return _frozen(np.stack([moves, turns], axis=1).astype(np.int64))


def legal_action_array(cfg: "Config") -> np.ndarray:
    """(K, 2) int array of legal (move, turn) pairs, in flat-index order."""
    return _legal_array(cfg.move_dirs, cfg.move_turn_mode)


def legal_actions(cfg: "Config") -> list[tuple[int, int]]:
    """List of legal (move, turn) pairs: 27 in simultaneous mode, 11 in exclusive mode (8 dirs)."""
    return [(int(m), int(t)) for m, t in legal_action_array(cfg)]


def n_actions(cfg: "Config") -> int:
    return len(legal_action_array(cfg))
