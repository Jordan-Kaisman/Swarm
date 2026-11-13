"""Episode runner and match statistics (spec §12).

Each game gets its own SeedSequence, split into three independent streams
(environment, blue policy, red policy), so the environment's RNG stream is the
same regardless of which policies play (common random numbers).

Two outcomes are reported per game:
- the knockout outcome, from the final state (`classify_outcome`);
- the return-based outcome, the sign of blue's undiscounted return (`return_outcome`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .env import DRONE, observe, reset, step
from .mirror import mirror_team_action
from .state import BLUE, RED, State

# Outcome causes.
BASES = "bases_destroyed"
ELIMINATION = "elimination"
MUTUAL_ELIMINATION = "mutual_elimination"
SIMULTANEOUS = "simultaneous_knockout"  # both teams knocked out in one step, not both by elimination
TRUNCATION = "truncation"

# Knockout buckets (blue's perspective) and the report rows, in order.
WIN, LOSS, DRAW, TRUNCATED = "win", "loss", "draw", "truncated"
KNOCKOUT_ROWS = (
    (WIN, BASES), (WIN, ELIMINATION),
    (LOSS, BASES), (LOSS, ELIMINATION),
    (DRAW, MUTUAL_ELIMINATION), (DRAW, SIMULTANEOUS),
    (TRUNCATED, TRUNCATION),
)

# Return-based outcome: the sign of blue's episode return. |return| <= RETURN_TOL counts as
# even, so that float rounding in sums like 0.1 + 0.2 - 0.3 doesn't pick a side.
BLUE_AHEAD, EVEN, RED_AHEAD = "blue_ahead", "even", "red_ahead"
RETURN_OUTCOMES = (BLUE_AHEAD, EVEN, RED_AHEAD)
RETURN_TOL = 1e-9

# The return-based outcome that agrees with each knockout bucket. Anything else is a
# disagreement (a truncated game has no knockout winner, so it agrees only when even).
AGREEING_RETURN = {WIN: BLUE_AHEAD, LOSS: RED_AHEAD, DRAW: EVEN, TRUNCATED: EVEN}


@dataclass
class EpisodeResult:
    blue_return: float
    length: int
    terminated: bool
    truncated: bool
    winner: int | None          # BLUE, RED, or None (draw / truncated)
    cause: str
    drone_kills: np.ndarray     # (2,) enemy drones killed by (blue, red)
    base_kills: np.ndarray      # (2,) enemy bases destroyed by (blue, red)
    shots_drone: np.ndarray     # (2,) shots at drones by (blue, red)
    hits_drone: np.ndarray      # (2,)
    shots_base: np.ndarray      # (2,) shots at bases
    hits_base: np.ndarray       # (2,)
    return_terms: np.ndarray    # (3,) blue_return split into RETURN_TERMS; sums to blue_return
    trajectory: list | None = field(default=None, repr=False)


def joint_action(cfg: Config, state: State, blue_policy, red_policy,
                 blue_rng: np.random.Generator, red_rng: np.random.Generator) -> np.ndarray:
    """Query both policies in their canonical frames and map red's back to the world frame."""
    blue = np.asarray(blue_policy(observe(state, BLUE, cfg), cfg, blue_rng))
    red = mirror_team_action(np.asarray(red_policy(observe(state, RED, cfg), cfg, red_rng)), cfg)
    return np.concatenate([blue, red]).astype(np.int64, copy=False)


def classify_outcome(cfg: Config, state: State, truncated: bool) -> tuple[int | None, str]:
    """Winner and cause, from the final state.

    A team is knocked out when it has no alive drones or no alive bases. If
    exactly one team is knocked out, the other wins. Both at once is a draw.
    """
    if not state.terminated:
        return None, TRUNCATION if truncated else "ongoing"
    n, B = state.n, state.B
    has_drones = (bool(state.alive[:n].any()), bool(state.alive[n:].any()))
    has_bases = (bool((state.base_hp[:B] > 0).any()), bool((state.base_hp[B:] > 0).any()))
    out = tuple(not (d and b) for d, b in zip(has_drones, has_bases))
    if out[BLUE] and out[RED]:
        return None, MUTUAL_ELIMINATION if not any(has_drones) else SIMULTANEOUS
    loser = BLUE if out[BLUE] else RED
    return 1 - loser, ELIMINATION if not has_drones[loser] else BASES


RETURN_TERMS = ("base-value", "elimination bonus", "drone-value")


def return_terms(cfg: Config, state: State) -> np.ndarray:
    """Blue's episode return split into RETURN_TERMS, from the final state of an episode that
    started from reset() (spec §6.7):
    - base-value: value of red bases destroyed - value of blue bases destroyed;
    - elimination bonus: +- the eliminated team's alive bases, if exactly one team has no drones;
    - drone-value: drone_value * (blue survivors - red survivors), the per-kill rewards summed.
    """
    n, B = state.n, state.B
    v, up = state.base_value, state.base_hp > 0
    base = float(v[B:][~up[B:]].sum() - v[:B][~up[:B]].sum())
    blue_left, red_left = int(np.count_nonzero(state.alive[:n])), int(np.count_nonzero(state.alive[n:]))
    bonus = 0.0
    if cfg.elimination_bonus and (blue_left == 0) != (red_left == 0):
        bonus = float(v[B:][up[B:]].sum()) if blue_left else -float(v[:B][up[:B]].sum())
    return np.array([base, bonus, cfg.drone_value * (blue_left - red_left)])


def knockout_bucket(result: EpisodeResult) -> str:
    """win / loss / draw from blue's perspective, or truncated (not a game outcome)."""
    if result.truncated:
        return TRUNCATED
    if result.winner is None:
        return DRAW
    return WIN if result.winner == BLUE else LOSS


def return_outcome(blue_return: float, tol: float = RETURN_TOL) -> str:
    """Sign of blue's episode return: blue ahead, even, or red ahead."""
    if blue_return > tol:
        return BLUE_AHEAD
    if blue_return < -tol:
        return RED_AHEAD
    return EVEN


def play_episode(cfg: Config, blue_policy, red_policy, seed, record: bool = False) -> EpisodeResult:
    """Play one game. `seed` is an int, a sequence of ints, or a SeedSequence."""
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    # Derive the three streams without ss.spawn(), which mutates ss and would make replaying
    # the same SeedSequence object give a different game.
    env_rng, blue_rng, red_rng = (
        np.random.default_rng(np.random.SeedSequence(ss.entropy, spawn_key=ss.spawn_key + (k,),
                                                     pool_size=ss.pool_size))
        for k in range(3))
    n, B = cfg.n_drones, cfg.bases_per_team

    state = reset(cfg, env_rng)
    trajectory = [(state, None)] if record else None
    total = 0.0
    drone_kills = np.zeros(2, np.int64)
    base_kills = np.zeros(2, np.int64)
    shots = np.zeros((2, 2), np.int64)  # [team, kind] with kind 0 = drone, 1 = base
    hits = np.zeros((2, 2), np.int64)
    terminated = truncated = False

    while not (terminated or truncated):
        actions = joint_action(cfg, state, blue_policy, red_policy, blue_rng, red_rng)
        state, reward, terminated, truncated, info = step(cfg, state, actions, env_rng)
        total += reward
        killed, destroyed = info["drones_killed"], info["bases_destroyed"]
        drone_kills += (int(killed[n:].sum()), int(killed[:n].sum()))
        base_kills += (int(destroyed[B:].sum()), int(destroyed[:B].sum()))
        for shooter, kind, _, _, hit in info["shots"]:
            team, k = shooter // n, 0 if kind == DRONE else 1
            shots[team, k] += 1
            hits[team, k] += hit
        if record:
            trajectory.append((state, info))

    winner, cause = classify_outcome(cfg, state, truncated)
    return EpisodeResult(
        blue_return=total, length=state.t, terminated=terminated, truncated=truncated,
        winner=winner, cause=cause, drone_kills=drone_kills, base_kills=base_kills,
        shots_drone=shots[:, 0], hits_drone=hits[:, 0], shots_base=shots[:, 1], hits_base=hits[:, 1],
        return_terms=return_terms(cfg, state), trajectory=trajectory,
    )


def game_seeds(seed: int, games: int) -> list[np.random.SeedSequence]:
    return np.random.SeedSequence(seed).spawn(games)


def run_match(cfg: Config, blue_policy, red_policy, games: int, seed: int = 0) -> list[EpisodeResult]:
    return [play_episode(cfg, blue_policy, red_policy, s) for s in game_seeds(seed, games)]


def summarize(results: list[EpisodeResult]) -> dict:
    """Aggregate statistics for the match report."""
    G = len(results)
    returns = np.array([r.blue_return for r in results])
    lengths = np.array([r.length for r in results])
    causes: dict[str, dict[str, int]] = {b: {} for b in (WIN, LOSS, DRAW, TRUNCATED)}
    crosstab = {row: dict.fromkeys(RETURN_OUTCOMES, 0) for row in KNOCKOUT_ROWS}
    disagreements = dict.fromkeys(KNOCKOUT_ROWS, 0)
    for r in results:
        bucket, ro = knockout_bucket(r), return_outcome(r.blue_return)
        causes[bucket][r.cause] = causes[bucket].get(r.cause, 0) + 1
        crosstab[bucket, r.cause][ro] += 1
        disagreements[bucket, r.cause] += ro != AGREEING_RETURN[bucket]
    q1, median, q3 = np.percentile(lengths, [25, 50, 75])
    terms = np.array([r.return_terms for r in results])

    def tot(attr: str) -> np.ndarray:
        return np.sum([getattr(r, attr) for r in results], axis=0)

    return {
        "games": G,
        "outcomes": causes,
        "counts": {k: sum(v.values()) for k, v in causes.items()},
        "return_outcomes": {ro: sum(row[ro] for row in crosstab.values()) for ro in RETURN_OUTCOMES},
        "crosstab": crosstab,            # (bucket, cause) -> return outcome -> count
        "disagreements": disagreements,  # (bucket, cause) -> count
        "lengths": lengths,
        "length_mean": float(lengths.mean()),
        "length_std": float(lengths.std(ddof=1)) if G > 1 else 0.0,
        "length_median": float(median),
        "length_q1": float(q1),
        "length_q3": float(q3),
        "truncated_frac": float(np.mean([r.truncated for r in results])),
        "return_mean": float(returns.mean()),
        "return_se": float(returns.std(ddof=1) / np.sqrt(G)) if G > 1 else 0.0,
        "terms_mean": terms.mean(axis=0),                                   # (3,), RETURN_TERMS order
        "terms_se": terms.std(axis=0, ddof=1) / np.sqrt(G) if G > 1 else np.zeros(3),
        "drone_kills": tot("drone_kills"),
        "base_kills": tot("base_kills"),
        "shots_drone": tot("shots_drone"),
        "hits_drone": tot("hits_drone"),
        "shots_base": tot("shots_base"),
        "hits_base": tot("hits_base"),
    }
