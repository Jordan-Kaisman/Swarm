"""Environment core: reset (spec §7), step (§6) and observe (§8).

`step` is pure: it never mutates its input and returns a new State. All
randomness in a step is exactly 2n uniforms, drawn whether or not anyone fires;
`_step_with_uniforms` takes them explicitly so tests can inject them.
"""

from __future__ import annotations

import numpy as np

from .actions import N_TURNS, direction_table, legal_table
from .config import Config
from .geometry import EPS_MOVE, PI, crossing_sin, drone_hit_prob, in_cone, pairwise, wrap
from .mirror import mirror_heading, mirror_points, mirror_state
from .state import BLUE, RED, State

DRONE, BASE = "drone", "base"
MAX_SAMPLING_TRIES = 10_000


# ---------------------------------------------------------------------------
# Per-config constants, cached on the (frozen) Config instance.

class _Kernel:
    __slots__ = ("n", "B", "N", "M", "rows", "action_shape", "flat_disp", "flat_dturn",
                 "legal_flat", "upper", "exclusive", "enemy", "base_sign", "drone_sign")

    def __init__(self, cfg: Config) -> None:
        n, B = cfg.n_drones, cfg.bases_per_team
        N = 2 * n
        self.n, self.B, self.N, self.M = n, B, N, N + 2 * B
        self.rows = np.arange(N)
        # Lookup tables indexed by the flat action a = move * 3 + turn.
        self.action_shape = (cfg.move_dirs + 1, N_TURNS)
        self.flat_disp = np.repeat(direction_table(cfg), N_TURNS, axis=0)            # (K, 2)
        self.flat_dturn = np.tile(np.array([-cfg.turn_delta, 0.0, cfg.turn_delta]),
                                  cfg.move_dirs + 1)                                  # (K,)
        self.legal_flat = legal_table(cfg).ravel()
        self.upper = np.array([cfg.width, cfg.height], dtype=np.float64)
        self.exclusive = cfg.move_turn_mode == "exclusive"
        # Target columns: 2n drones followed by 2B bases.
        shooter_team = np.arange(N) // n
        target_team = np.concatenate([np.arange(N) // n, np.arange(2 * B) // B])
        self.enemy = shooter_team[:, None] != target_team[None, :]
        # Blue-perspective sign of losing an asset: blue's assets -1, red's +1.
        self.base_sign = np.repeat([-1.0, 1.0], B)
        self.drone_sign = np.repeat([-1.0, 1.0], n)


def _kernel(cfg: Config) -> _Kernel:
    k = cfg.__dict__.get("_env_kernel")
    if k is None:
        k = _Kernel(cfg)
        cfg.__dict__["_env_kernel"] = k  # not a dataclass field: invisible to eq/hash/replace
    return k


# ---------------------------------------------------------------------------
# Reset

def _min_pairwise_distance(p: np.ndarray) -> float:
    if len(p) < 2:
        return np.inf
    _, _, d = pairwise(p, p)
    return float(d[np.triu_indices(len(p), 1)].min())


def _rejection_sample(propose, min_sep: float, what: str) -> np.ndarray:
    for _ in range(MAX_SAMPLING_TRIES):
        pts = propose()
        if _min_pairwise_distance(pts) >= min_sep:
            return pts
    raise RuntimeError(f"could not place {what} after {MAX_SAMPLING_TRIES} tries; "
                       "the separation constraints are probably too tight")


def _sample_blue_bases(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    if cfg.base_layout is not None:
        return np.array(cfg.base_layout, dtype=np.float64)
    lo = (cfg.base_min_wall, cfg.base_min_wall)
    hi = (cfg.midline - cfg.base_min_mid, cfg.height - cfg.base_min_wall)
    B = cfg.bases_per_team
    return _rejection_sample(lambda: rng.uniform(lo, hi, size=(B, 2)), cfg.base_min_sep, "bases")


def _sample_blue_drones(cfg: Config, rng: np.random.Generator, blue_bases: np.ndarray):
    """Positions and headings for one team, in blue's frame."""
    n = cfg.n_drones
    if cfg.placement == "zone":
        lo = (cfg.midline - cfg.zone_mid_max, cfg.zone_wall_margin)
        hi = (cfg.midline - cfg.zone_mid_min, cfg.height - cfg.zone_wall_margin)

        def propose():
            return rng.uniform(lo, hi, size=(n, 2))
    else:  # "bases": uniform in a disc around bases assigned round-robin, clipped to the arena
        centers = blue_bases[np.arange(n) % len(blue_bases)]
        upper = np.array([cfg.width, cfg.height])

        def propose():
            r = cfg.bases_spawn_radius * np.sqrt(rng.random(n))
            phi = rng.uniform(-PI, PI, size=n)
            pts = centers + r[:, None] * np.stack([np.cos(phi), np.sin(phi)], axis=1)
            return np.clip(pts, 0.0, upper)

    pos = _rejection_sample(propose, cfg.drone_min_sep, "drones")
    heading = rng.uniform(-PI, PI, size=n) if cfg.random_headings else np.zeros(n)
    return pos, wrap(heading)


def reset(cfg: Config, rng: np.random.Generator) -> State:
    """Sample a fresh episode: mirrored bases, then drones by `cfg.placement`."""
    blue_bases = _sample_blue_bases(cfg, rng)
    base_pos = np.concatenate([blue_bases, mirror_points(blue_bases, cfg.width)])

    blue_pos, blue_head = _sample_blue_drones(cfg, rng, blue_bases)
    if cfg.mirror_init:
        red_src_pos, red_src_head = blue_pos, blue_head
    else:  # sampled independently by the same rules, then reflected into red's half
        red_src_pos, red_src_head = _sample_blue_drones(cfg, rng, blue_bases)
    pos = np.concatenate([blue_pos, mirror_points(red_src_pos, cfg.width)])
    heading = np.concatenate([blue_head, mirror_heading(red_src_head)])

    N, B2 = cfg.n_total, 2 * cfg.bases_per_team
    return State(
        pos=pos,
        heading=heading,
        alive=np.ones(N, dtype=bool),
        weapon_range=cfg.weapon_range_array(),
        weapon_cone=cfg.weapon_cone_array(),
        base_pos=base_pos,
        base_hp=np.full(B2, cfg.base_hp, dtype=np.int64),
        base_value=np.tile(np.asarray(cfg.base_values, dtype=np.float64), 2),
        t=0,
        terminated=False,
    )


# ---------------------------------------------------------------------------
# Step phases. Each phase computes from the arrays produced by the previous one.
# The count_nonzero() checks are fast paths (cheaper than .any() on tiny arrays);
# they skip work whose result is known, and never change the outcome.

def _validate_actions(k: _Kernel, actions, alive: np.ndarray) -> np.ndarray:
    """6.1: check shape, value ranges and mode legality; dead drones' actions are ignored.

    Returns the flat action index move * 3 + turn, with dead drones set to (hover, no turn).
    """
    a = np.asarray(actions)
    if a.shape != (k.N, 2):
        raise ValueError(f"actions must have shape ({k.N}, 2), got {a.shape}")
    if a.dtype.kind not in "iu":
        raise TypeError(f"actions must be an integer array, got dtype {a.dtype}")
    move, turn = a[:, 0], a[:, 1]
    if np.count_nonzero(alive) != k.N:
        move, turn = np.where(alive, move, 0), np.where(alive, turn, 1)
    try:
        flat = np.ravel_multi_index((move, turn), k.action_shape)  # raises on out-of-range
    except (ValueError, TypeError):
        bad = (move < 0) | (move >= k.action_shape[0]) | (turn < 0) | (turn >= k.action_shape[1])
        raise ValueError(f"action out of range for drone(s) {np.flatnonzero(bad).tolist()}: "
                         f"{a[bad].tolist()}") from None
    if k.exclusive:
        illegal = ~k.legal_flat[flat]
        if np.count_nonzero(illegal):
            raise ValueError(f"illegal action in 'exclusive' mode for drone(s) "
                             f"{np.flatnonzero(illegal).tolist()}: {a[illegal].tolist()}")
    return flat


def _move_and_turn(k: _Kernel, state: State, flat: np.ndarray):
    """6.2: heading += (turn-1)*delta (wrapped); pos += step_len*dir(move), clipped to the arena.

    disp is the actual (post-clip) displacement and moved = |disp| > EPS_MOVE.
    """
    heading = wrap(state.heading + k.flat_dturn[flat])
    pos = np.minimum(np.maximum(state.pos + k.flat_disp[flat], 0.0), k.upper)
    disp = pos - state.pos
    disp_len = np.hypot(disp[:, 0], disp[:, 1])
    return heading, pos, disp, disp_len, disp_len > EPS_MOVE


def _hit_probabilities(cfg: Config, k: _Kernel, pos, heading, disp, disp_len, moved, alive,
                       weapon_range, weapon_cone, base_pos, base_alive):
    """6.3 + 6.4: P[i, j] for every shooter i and target column j (2n drones, then 2B bases).

    Uses post-move positions and post-turn headings. P is zero unless shooter
    and target are both alive, on opposite teams, and the target is within the
    shooter's range and cone. Returns (P, dist), with P = None when no shooter
    has any valid target.
    """
    N = k.N
    lx, ly, dist = pairwise(pos, np.concatenate((pos, base_pos)))   # LOF_ij = p_j - p_i, (N, M)

    valid = dist <= weapon_range[:, None]
    valid &= k.enemy
    valid &= np.concatenate((alive, base_alive))
    valid &= alive[:, None]
    if not np.count_nonzero(valid):
        return None, dist
    valid &= in_cone(lx, ly, dist, heading[:, None], weapon_cone[:, None])
    if not np.count_nonzero(valid):
        return None, dist

    P = np.empty(dist.shape)
    # Drone targets.
    if np.count_nonzero(moved):
        inv = np.where(moved, 1.0 / np.maximum(disp_len, EPS_MOVE), 0.0)   # zero vector if j didn't move
        sin_phi = crossing_sin(lx[:, :N], ly[:, :N], dist[:, :N], disp[:, 0] * inv, disp[:, 1] * inv)
    else:
        sin_phi = 0.0
    P[:, :N] = drone_hit_prob(dist[:, :N], weapon_cone[:, None], sin_phi, moved, moved[:, None],
                              hit_c=cfg.hit_c, hit_pmax=cfg.hit_pmax,
                              alpha=cfg.alpha_target_motion, beta=cfg.beta_shooter_motion)
    # Base targets: a fixed per-shot probability.
    P[:, N:] = cfg.base_hit_prob
    P *= valid
    return P, dist


def _select_targets(k: _Kernel, P: np.ndarray, dist: np.ndarray):
    """6.5 auto-fire, drones_first: every drone with a valid target fires exactly one shot.

    If any enemy drone has P > 0 the shot goes to a drone, otherwise to a base.
    Highest P wins; ties go to the smaller distance, then the lower index (no RNG).
    Returns (fires (N,) bool, target column (N,) int, P of the chosen shot (N,)).
    """
    N, rows = k.N, k.rows
    Pd = P[:, :N]
    has_drone_target = Pd[rows, Pd.argmax(axis=1)] > 0.0
    if np.count_nonzero(has_drone_target):
        P = P.copy()
        P[has_drone_target, N:] = 0.0              # drones first: ignore bases for these shooters
    target = P.argmax(axis=1)                      # first maximum = lowest index
    p_best = P[rows, target]
    fires = p_best > 0.0
    cand = P == p_best[:, None]
    cand &= fires[:, None]
    if np.count_nonzero(cand) > np.count_nonzero(fires):   # some shooter has an exact tie
        tied = cand.sum(axis=1) > 1
        cand_dist = np.where(cand[tied], dist[tied], np.inf)
        target[tied] = (cand_dist == cand_dist.min(axis=1)[:, None]).argmax(axis=1)
    return fires, target, p_best


def _resolve(k: _Kernel, state: State, fires, target, p_shot, u):
    """6.6: drone i's shot hits iff u[i] < P. All outcomes apply at once, so a drone
    killed this step still fired this step, and a base loses one HP per hit."""
    hit = fires & (u < p_shot)
    if not np.count_nonzero(hit):
        return (hit, np.zeros(k.N, bool), np.zeros(2 * k.B, np.int64), state.alive.copy(),
                state.base_hp.copy(), np.zeros(2 * k.B, bool))
    hits_per_target = np.bincount(target[hit], minlength=k.M)
    killed = (hits_per_target[:k.N] > 0) & state.alive
    base_hits = hits_per_target[k.N:]
    new_alive = state.alive & ~killed
    new_hp = np.maximum(state.base_hp - base_hits, 0)
    destroyed = (state.base_hp > 0) & (new_hp <= 0)
    return hit, killed, base_hits, new_alive, new_hp, destroyed


def _reward_and_termination(cfg: Config, k: _Kernel, state: State, killed, destroyed,
                            new_alive, new_hp):
    """6.7 reward (blue's perspective, this step only) and 6.8 termination."""
    n, B = k.n, k.B
    r = 0.0
    if np.count_nonzero(destroyed):
        r += float(k.base_sign @ (state.base_value * destroyed))
    if cfg.drone_value != 0.0 and np.count_nonzero(killed):
        r += cfg.drone_value * float(k.drone_sign @ killed)
    blue_up = np.count_nonzero(new_alive[:n]) > 0
    red_up = np.count_nonzero(new_alive[n:]) > 0
    bases_alive = new_hp > 0
    if cfg.elimination_bonus and blue_up != red_up:
        # The survivor also collects the eliminated team's still-alive bases.
        if blue_up:
            r += float(state.base_value[B:] @ bases_alive[B:])
        else:
            r -= float(state.base_value[:B] @ bases_alive[:B])
    terminated = not (blue_up and red_up and np.count_nonzero(bases_alive[:B])
                      and np.count_nonzero(bases_alive[B:]))
    return r, terminated


def _step_with_uniforms(cfg: Config, state: State, actions, u):
    """One simultaneous step given the 2n uniforms `u` (drone i's shot hits iff u[i] < P)."""
    if state.terminated:
        raise RuntimeError("step() called on a terminated state")
    k = _kernel(cfg)
    u = np.asarray(u, dtype=np.float64)
    if u.shape != (k.N,):
        raise ValueError(f"u must have shape ({k.N},), got {u.shape}")

    flat = _validate_actions(k, actions, state.alive)
    heading, pos, disp, disp_len, moved = _move_and_turn(k, state, flat)
    P, dist = _hit_probabilities(cfg, k, pos, heading, disp, disp_len, moved, state.alive,
                                 state.weapon_range, state.weapon_cone, state.base_pos,
                                 state.base_hp > 0)
    if P is None:  # no shooter has a valid target: nobody fires
        fires, target, p_shot = np.zeros(k.N, bool), np.zeros(k.N, np.int64), np.zeros(k.N)
    else:
        fires, target, p_shot = _select_targets(k, P, dist)
    hit, killed, base_hits, new_alive, new_hp, destroyed = _resolve(k, state, fires, target, p_shot, u)
    reward, terminated = _reward_and_termination(cfg, k, state, killed, destroyed, new_alive, new_hp)
    t = state.t + 1
    truncated = (not terminated) and t >= cfg.max_steps

    new_state = State(
        pos=pos,
        heading=heading,
        alive=new_alive,
        weapon_range=state.weapon_range.copy(),
        weapon_cone=state.weapon_cone.copy(),
        base_pos=state.base_pos.copy(),
        base_hp=new_hp,
        base_value=state.base_value.copy(),
        t=t,
        terminated=terminated,
    )

    # 6.9 info
    shooters = np.flatnonzero(fires)
    shots = [
        (i, DRONE, j, p, h) if j < k.N else (i, BASE, j - k.N, p, h)
        for i, j, p, h in zip(shooters.tolist(), target[shooters].tolist(),
                              p_shot[shooters].tolist(), hit[shooters].tolist())
    ] if shooters.size else []
    info = {
        "shots": shots,                  # [(shooter, "drone"|"base", target_idx, P, hit)]
        "drones_killed": killed,         # (2n,) bool
        "bases_hit": base_hits,          # (2B,) int, hits taken this step
        "bases_destroyed": destroyed,    # (2B,) bool, destroyed this step
        "disp": disp,                    # (2n, 2) actual displacement
        "moved": moved,                  # (2n,) bool
    }
    return new_state, reward, terminated, truncated, info


def step(cfg: Config, state: State, actions, rng: np.random.Generator):
    """Advance one step. Returns (new_state, reward_blue, terminated, truncated, info).

    Draws exactly 2n uniforms from `rng` every step (common random numbers).
    """
    if state.terminated:
        raise RuntimeError("step() called on a terminated state")
    return _step_with_uniforms(cfg, state, actions, rng.random(cfg.n_total))


# ---------------------------------------------------------------------------
# Observation and analysis helpers

def observe(state: State, team: int, cfg: Config) -> State:
    """The only path from state to policy input. v0: the full state, in the team's canonical frame.

    Blue gets the state itself (policies must not mutate it); red gets the mirror.
    """
    if team == BLUE:
        return state
    if team == RED:
        return mirror_state(state, cfg)
    raise ValueError(f"team must be 0 (blue) or 1 (red), got {team}")


def hit_probability_matrix(cfg: Config, state: State, disp: np.ndarray | None = None):
    """P[i, j] for the state's positions/headings, treated as post-move.

    Columns are the 2n drones followed by the 2B bases. `disp` (2n, 2) gives
    this step's displacements (default: nobody moved). Returns (P, dist).
    """
    k = _kernel(cfg)
    disp = np.zeros((k.N, 2)) if disp is None else np.asarray(disp, dtype=np.float64)
    disp_len = np.hypot(disp[:, 0], disp[:, 1])
    P, dist = _hit_probabilities(cfg, k, state.pos, state.heading, disp, disp_len,
                                 disp_len > EPS_MOVE, state.alive, state.weapon_range,
                                 state.weapon_cone, state.base_pos, state.base_hp > 0)
    return (np.zeros(dist.shape) if P is None else P), dist
