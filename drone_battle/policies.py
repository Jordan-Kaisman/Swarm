"""Scripted baseline policies (spec §11).

Signature: policy(obs, cfg, rng) -> (n, 2) int array of (move, turn), in the
canonical frame (the policy's own drones are rows 0..n-1 and it owns the left
half). Policies must not read `obs.t` and must not mutate `obs`.

In "exclusive" mode a drone can move or turn but not both. The scripted
policies then turn when their aim point is within weapon range (+ one step) but
outside the cone, and move otherwise.
"""

from __future__ import annotations

import functools
from typing import Callable

import numpy as np

from .actions import HOVER, TURN_NONE, legal_action_array, unit_directions
from .config import Config, parse_value
from .geometry import EPS_DIST, wrap
from .state import State

Policy = Callable[[State, Config, np.random.Generator], np.ndarray]

_TURN_PREFERENCE = np.array([TURN_NONE, 0, 2])  # on ties, prefer not turning


# ---------------------------------------------------------------------------
# Helpers (vectorized over the team's drones)

def turn_toward(heading: np.ndarray, bearing: np.ndarray, cfg: Config) -> np.ndarray:
    """Turn index minimizing |wrap(bearing - heading_after_turn)|."""
    after = heading[:, None] + (_TURN_PREFERENCE[None, :] - 1) * cfg.turn_delta
    err = np.abs(wrap(bearing[:, None] - after))
    return _TURN_PREFERENCE[err.argmin(axis=1)]


def move_toward(pos: np.ndarray, target: np.ndarray, cfg: Config) -> np.ndarray:
    """Direction with the largest projection onto (target - pos); hover within step_len/2."""
    vec = target - pos
    dist = np.hypot(vec[:, 0], vec[:, 1])
    move = (vec @ unit_directions(cfg.move_dirs).T).argmax(axis=1) + 1
    return np.where(dist < 0.5 * cfg.step_len, HOVER, move)


def nearest(src: np.ndarray, pts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """For each row of src (n, 2), the nearest of the masked pts (m, 2). Requires mask.any()."""
    d2 = ((pts[None, :, :] - src[:, None, :]) ** 2).sum(axis=-1)
    d2[:, ~mask] = np.inf
    return pts[d2.argmin(axis=1)]


def _clip_norm(v: np.ndarray, radius: float) -> np.ndarray:
    norm = np.hypot(v[:, 0], v[:, 1])
    scale = np.where(norm > radius, radius / np.maximum(norm, 1e-12), 1.0)
    return v * scale[:, None]


def _bearing(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    d = target - pos
    return np.arctan2(d[:, 1], d[:, 0])


def _assemble(obs: State, cfg: Config, move: np.ndarray, turn: np.ndarray,
              aim: np.ndarray | None) -> np.ndarray:
    """Stack (move, turn), make it legal for the move/turn mode, idle dead drones."""
    n = cfg.n_drones
    move, turn = move.astype(np.int64), turn.astype(np.int64)
    if cfg.move_turn_mode == "exclusive":
        wants_both = (move != HOVER) & (turn != TURN_NONE)
        if aim is not None:
            pos, heading = obs.pos[:n], obs.heading[:n]
            d = aim - pos
            dist = np.hypot(d[:, 0], d[:, 1])
            off_axis = np.abs(wrap(np.arctan2(d[:, 1], d[:, 0]) - heading))
            aim_first = (dist <= obs.weapon_range[:n] + cfg.step_len) & (off_axis > 0.5 * obs.weapon_cone[:n])
        else:
            aim_first = np.zeros(n, bool)
        move = np.where(wants_both & aim_first, HOVER, move)
        turn = np.where(move != HOVER, TURN_NONE, turn)
    alive = obs.alive[:n]
    return np.stack([np.where(alive, move, HOVER), np.where(alive, turn, TURN_NONE)], axis=1)


def _idle(cfg: Config) -> np.ndarray:
    return np.tile(np.array([HOVER, TURN_NONE], dtype=np.int64), (cfg.n_drones, 1))


def _go_and_aim(obs: State, cfg: Config, goal: np.ndarray, aim: np.ndarray | None) -> np.ndarray:
    n = cfg.n_drones
    pos, heading = obs.pos[:n], obs.heading[:n]
    move = move_toward(pos, goal, cfg)
    turn = (turn_toward(heading, _bearing(pos, aim), cfg) if aim is not None
            else np.full(n, TURN_NONE))
    return _assemble(obs, cfg, move, turn, aim)


# ---------------------------------------------------------------------------
# Policies

def random_policy(obs: State, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Uniform over legal actions."""
    legal = legal_action_array(cfg)
    return legal[rng.integers(len(legal), size=cfg.n_drones)].copy()


def rusher(obs: State, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Move toward the nearest alive enemy base and turn toward it."""
    n, B = cfg.n_drones, cfg.bases_per_team
    enemy_alive = obs.base_hp[B:] > 0
    if not enemy_alive.any():
        return _idle(cfg)
    target = nearest(obs.pos[:n], obs.base_pos[B:], enemy_alive)
    return _go_and_aim(obs, cfg, target, target)


def hunter(obs: State, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Move toward the nearest alive enemy drone and turn toward it."""
    n = cfg.n_drones
    enemy_alive = obs.alive[n:]
    if not enemy_alive.any():
        return _idle(cfg)
    target = nearest(obs.pos[:n], obs.pos[n:], enemy_alive)
    return _go_and_aim(obs, cfg, target, target)


def defender(obs: State, cfg: Config, rng: np.random.Generator, r_def: float = 4.0) -> np.ndarray:
    """Hold a disc of radius r_def around the nearest own base; turn toward the nearest
    enemy drone; when enemies are in the own half, move to the point of the disc closest
    to the nearest intruder (intercept without leaving the disc)."""
    n, B = cfg.n_drones, cfg.bases_per_team
    pos = obs.pos[:n]
    own_alive = obs.base_hp[:B] > 0
    if not own_alive.any():
        return _idle(cfg)
    anchor = nearest(pos, obs.base_pos[:B], own_alive)

    enemy_pos, enemy_alive = obs.pos[n:], obs.alive[n:]
    aim = nearest(pos, enemy_pos, enemy_alive) if enemy_alive.any() else None
    intruders = enemy_alive & (enemy_pos[:, 0] < cfg.midline)
    toward = nearest(pos, enemy_pos, intruders) if intruders.any() else pos
    goal = anchor + _clip_norm(toward - anchor, r_def)
    return _go_and_aim(obs, cfg, goal, aim)


def _raid(obs: State, cfg: Config, members: np.ndarray, r_group: float, engage_margin: float,
          strafe: bool) -> np.ndarray:
    """Actions for the drones in `members` attacking one enemy base as a pack; other rows idle.

    Target: the alive enemy base nearest the centroid of the pack's alive drones. A drone
    advances iff D_i >= D_max - r_group (D = distance to the target, max over the pack), so
    the pack moves at the pace of its rearmost drone. It faces the nearest enemy drone when
    one is within weapon_range + engage_margin (a threat), else the target. In weapon range
    of the target with no threat, it hovers. With `strafe`, an advancing drone under threat
    moves in the legal direction closest to perpendicular to its line of fire to the nearest
    enemy drone, on the side with the larger component toward the target.
    """
    n, B = cfg.n_drones, cfg.bases_per_team
    pos, heading = obs.pos[:n], obs.heading[:n]
    pack = members & obs.alive[:n]
    enemy_bases = obs.base_hp[B:] > 0
    if not pack.any() or not enemy_bases.any():
        return _idle(cfg)
    centroid = pos[pack].mean(axis=0, keepdims=True)
    target = nearest(centroid, obs.base_pos[B:], enemy_bases)[0]
    to_target = target - pos
    D = np.hypot(to_target[:, 0], to_target[:, 1])
    advance = D >= D[pack].max() - r_group

    weapon_range = obs.weapon_range[:n]
    enemy_alive = obs.alive[n:]
    if enemy_alive.any():
        foe = nearest(pos, obs.pos[n:], enemy_alive)
        lof = foe - pos
        d_foe = np.hypot(lof[:, 0], lof[:, 1])
        threat = d_foe <= weapon_range + engage_margin
        aim = np.where(threat[:, None], foe, target)
    else:
        threat = np.zeros(n, bool)
        aim = np.broadcast_to(target, pos.shape)
    hold = (D <= weapon_range) & ~threat
    move = np.where(advance & ~hold, move_toward(pos, np.broadcast_to(target, pos.shape), cfg), HOVER)

    if strafe and threat.any():
        u = lof / np.maximum(d_foe, EPS_DIST)[:, None]
        perp = np.stack([-u[:, 1], u[:, 0]], axis=1)                 # CCW normal to the line of fire
        side = np.where((perp * to_target).sum(axis=1) >= 0, 1.0, -1.0)
        sideways = (perp * side[:, None]) @ unit_directions(cfg.move_dirs).T
        dodge = advance & threat & (d_foe > EPS_DIST)                # no line of fire at d = 0
        move = np.where(dodge, sideways.argmax(axis=1) + 1, move)

    turn = turn_toward(heading, _bearing(pos, aim), cfg)
    actions = _assemble(obs, cfg, move, turn, aim)
    actions[~members] = (HOVER, TURN_NONE)
    return actions


def raid(obs: State, cfg: Config, rng: np.random.Generator, r_group: float = 2.5,
         engage_margin: float = 2.0) -> np.ndarray:
    """The whole team attacks one enemy base together, at the pace of its rearmost drone."""
    return _raid(obs, cfg, np.ones(cfg.n_drones, bool), r_group, engage_margin, strafe=False)


def raid_strafe(obs: State, cfg: Config, rng: np.random.Generator, r_group: float = 2.5,
                engage_margin: float = 2.0) -> np.ndarray:
    """raid, but advancing drones under threat move across the enemy's line of fire (alpha)."""
    return _raid(obs, cfg, np.ones(cfg.n_drones, bool), r_group, engage_margin, strafe=True)


def split(obs: State, cfg: Config, rng: np.random.Generator, k: int = 1, r_def: float = 4.0,
          r_group: float = 2.5, engage_margin: float = 2.0) -> np.ndarray:
    """Drones with canonical index < k run defender; the rest raid as a pack of their own.
    Roles are fixed by index."""
    raiders = np.arange(cfg.n_drones) >= k
    actions = _raid(obs, cfg, raiders, r_group, engage_margin, strafe=False)
    if not raiders.all():
        actions[~raiders] = defender(obs, cfg, rng, r_def=r_def)[~raiders]
    return actions


POLICIES: dict[str, Policy] = {
    "random": random_policy,
    "rusher": rusher,
    "hunter": hunter,
    "defender": defender,
    "raid": raid,
    "raid_strafe": raid_strafe,
    "split": split,
}


def get_policy(spec: str) -> Policy:
    """Look up a policy by name, optionally with keyword params: 'defender:r_def=6'."""
    name, _, params = spec.partition(":")
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}")
    policy = POLICIES[name]
    if params:
        kwargs = {}
        for item in params.split(","):
            key, sep, raw = item.partition("=")
            if not sep:
                raise ValueError(f"bad policy parameter {item!r} in {spec!r}")
            kwargs[key.strip()] = parse_value(raw)
        policy = functools.partial(policy, **kwargs)
        policy.__name__ = spec  # type: ignore[attr-defined]
    return policy
