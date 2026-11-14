"""Meta-game over scripted policies: payoff matrix, equilibrium mixture, bootstrap (spec §12).

    python scripts/metagame.py                                  # 6 policies, 2000 games per pair
    python scripts/metagame.py --games 1000 --seeds 0 1         # pool 1000 games per seed per pair
    python scripts/metagame.py --games 200 --boot 50            # quick look
    python scripts/metagame.py --load out/metagame_<timestamp>.json   # recompute the tables only

Every unordered pair i < j is played with i as blue (exact mirror symmetry makes the other
order and the mirror pairs redundant): A[i, j] = blue's mean return, A[j, i] = -A[i, j], and
the diagonal is 0. The symmetric zero-sum matrix game is solved by linear programming. The
bootstrap resamples games within each cell, recomputes A and the equilibrium, and reports how
often each policy is in the support (weight > 0.01) and its mean weight.

With several --seeds, each seed contributes its own `games` game seeds per pair and the games
are pooled per cell; the bootstrap resamples the pooled games.

Per-game results are saved to out/metagame_<timestamp>.json, stamped with the git commit and
dirty flag, the full Config, the seeds, the policy list and the timestamp, so the tables can be
recomputed with --load without re-running the games. Files without a stamp still load; the
missing items show as "unknown". Requires scipy.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

import _path  # noqa: F401
from drone_battle.config import Config, config_from_overrides
from drone_battle.match import game_seeds, play_episode
from drone_battle.policies import get_policy

DEFAULT_POLICIES = ("rusher", "hunter", "defender", "raid", "raid_strafe", "split")
SUPPORT_TOL = 0.01          # a policy is "in the support" when its weight exceeds this
CHUNK = 100                 # games per worker task
FIELDS = ("return", "length", "truncated", "cause", "winner", "seed")   # per game; "seed" = root seed
REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- playing
def _play_chunk(task: tuple) -> tuple[int, dict]:
    """Play the games for one slice of one pair's game seeds; returns (pair index, columns)."""
    pair_idx, cfg, blue, red, root, seeds = task
    blue_p, red_p = get_policy(blue), get_policy(red)
    cols: dict[str, list] = {f: [] for f in FIELDS}
    for ss in seeds:
        r = play_episode(cfg, blue_p, red_p, ss)
        cols["return"].append(float(r.blue_return))
        cols["length"].append(int(r.length))
        cols["truncated"].append(bool(r.truncated))
        cols["cause"].append(r.cause)
        cols["winner"].append(None if r.winner is None else int(r.winner))
        cols["seed"].append(int(root))
    return pair_idx, cols


def play_pairs(cfg: Config, names: list[str], games: int, seeds: list[int], workers: int) -> list[dict]:
    """Play `games` games per seed for every pair i < j (i as blue), pooled per pair in seed order.
    Each seed gives its own game seeds; every pair uses the same ones (common random numbers)."""
    pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
    by_seed = {s: game_seeds(s, games) for s in seeds}
    tasks = [(p, cfg, names[i], names[j], s, by_seed[s][g:g + CHUNK])
             for p, (i, j) in enumerate(pairs) for s in seeds for g in range(0, games, CHUNK)]
    results = [{"blue": names[i], "red": names[j], **{f: [] for f in FIELDS}} for i, j in pairs]
    remaining = [len(seeds) * len(range(0, games, CHUNK))] * len(pairs)

    def collect(pair_idx: int, cols: dict) -> None:
        for f in FIELDS:
            results[pair_idx][f].extend(cols[f])
        remaining[pair_idx] -= 1
        if remaining[pair_idx] == 0:
            done = sum(r == 0 for r in remaining)
            print(f"  [{done:2d}/{len(pairs)}] {names[pairs[pair_idx][0]]} vs {names[pairs[pair_idx][1]]}",
                  file=sys.stderr, flush=True)

    if workers <= 1:
        for t in tasks:
            collect(*_play_chunk(t))
    else:
        with Pool(workers) as pool:
            for out in pool.imap(_play_chunk, tasks):   # in order, so each pair's games stay in seed order
                collect(*out)
    return results


# ---------------------------------------------------------------- analysis
def payoff_matrix(names: list[str], cells: dict[tuple[int, int], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """A[i, j] (mean return of i against j) and its standard error, from the i < j cells."""
    k = len(names)
    A, SE = np.zeros((k, k)), np.zeros((k, k))
    for (i, j), r in cells.items():
        A[i, j], A[j, i] = r.mean(), -r.mean()
        SE[i, j] = SE[j, i] = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0.0
    return A, SE


def solve_zero_sum(A: np.ndarray) -> tuple[np.ndarray, float]:
    """Max-min mixture x of the row player and the game value v:
    maximize v subject to (A^T x)_j >= v for every column j, sum(x) = 1, x >= 0.
    For a skew-symmetric A (a symmetric game), v = 0 and x is an equilibrium strategy."""
    k = A.shape[0]
    c = np.zeros(k + 1)
    c[-1] = -1.0                                         # maximize v
    A_ub = np.hstack([-A.T, np.ones((k, 1))])            # v - (A^T x)_j <= 0
    A_eq = np.hstack([np.ones((1, k)), np.zeros((1, 1))])
    bounds = [(0.0, None)] * k + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(k), A_eq=A_eq, b_eq=[1.0], bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    x = np.clip(res.x[:k], 0.0, None)
    return x / x.sum(), float(res.x[-1]) + 0.0   # + 0.0 turns -0.0 into 0.0


def bootstrap(names: list[str], cells: dict[tuple[int, int], np.ndarray], n_boot: int,
              rng: np.random.Generator) -> np.ndarray:
    """(n_boot, k) equilibrium weights, each from A rebuilt with games resampled within each cell."""
    means = {key: r[rng.integers(len(r), size=(n_boot, len(r)))].mean(axis=1) for key, r in cells.items()}
    k = len(names)
    weights = np.empty((n_boot, k))
    for b in range(n_boot):
        A = np.zeros((k, k))
        for (i, j), m in means.items():
            A[i, j], A[j, i] = m[b], -m[b]
        weights[b] = solve_zero_sum(A)[0]
    return weights


def analyse(data: dict, n_boot: int, boot_seed: int) -> dict:
    names = data["policies"]
    index = {name: i for i, name in enumerate(names)}
    cells, pairs = {}, []
    for p in data["pairs"]:
        i, j = index[p["blue"]], index[p["red"]]
        r = np.asarray(p["return"], dtype=float)
        lengths = np.asarray(p["length"])
        q1, med, q3 = np.percentile(lengths, [25, 50, 75])
        cells[i, j] = r
        pairs.append({"blue": p["blue"], "red": p["red"], "mean": r.mean(),
                      "se": r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0.0,
                      "trunc": float(np.mean(p["truncated"])), "median": med, "q1": q1, "q3": q3})
    A, SE = payoff_matrix(names, cells)
    x, v = solve_zero_sum(A)
    w = bootstrap(names, cells, n_boot, np.random.default_rng(boot_seed)) if n_boot > 0 else np.empty((0, len(names)))
    return {"names": names, "A": A, "SE": SE, "pairs": pairs, "x": x, "value": v,
            "support_freq": (w > SUPPORT_TOL).mean(axis=0) if len(w) else None,
            "mean_weight": w.mean(axis=0) if len(w) else None, "n_boot": n_boot}


# ---------------------------------------------------------------- stamps
def git_stamp() -> dict:
    """The repo's current commit and whether the working tree has uncommitted changes."""
    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": "unknown"}


def run_seeds(data: dict) -> list[int] | None:
    """The seeds a saved run used (older files store a single "seed"); None if unknown."""
    if "seeds" in data:
        return list(data["seeds"])
    return [data["seed"]] if "seed" in data else None


def seeds_line(data: dict) -> str:
    seeds, games = run_seeds(data), data.get("games", "?")
    if seeds is None:
        return f"{games} games per pair, seeds unknown"
    if len(seeds) == 1:
        return f"{games} games per pair, seed {seeds[0]}"
    return (f"seeds {', '.join(map(str, seeds))} pooled: {games} games per pair per seed,"
            f" {games * len(seeds)} per pair")


def stamp_line(data: dict) -> str:
    git = data.get("git") or {}
    commit, dirty = git.get("commit", "unknown"), git.get("dirty", "unknown")
    state = {True: "dirty", False: "clean"}.get(dirty, "dirty flag unknown")
    where = "commit unknown" if commit == "unknown" else f"commit {commit[:12]} ({state})"
    return f"{where}, created {data.get('created', 'unknown')}"


# ---------------------------------------------------------------- report
def format_report(an: dict, data: dict) -> str:
    names, A, SE, k = an["names"], an["A"], an["SE"], len(an["names"])
    w = max(len(n) for n in names) + 2
    cell = 16
    lines = [f"Meta-game: {k} policies, {len(an['pairs'])} pairs (first listed is blue); {seeds_line(data)}",
             stamp_line(data), ""]

    lines.append("Payoff A[i, j] = mean return of row policy i against column policy j (± SE)")
    lines.append(" " * w + "".join(f"{n:>{cell}s}" for n in names))
    for i in range(k):
        row = "".join(f"{'0':>{cell}s}" if i == j else f"{f'{A[i, j]:+.3f} ± {SE[i, j]:.3f}':>{cell}s}"
                      for j in range(k))
        lines.append(f"{names[i]:<{w}s}{row}")
    lines.append("")

    pw = 2 * w + 4
    lines.append(f"{'pair (blue vs red)':<{pw}s}{'blue return ± SE':>18s}{'trunc':>8s}   median length (IQR)")
    for p in an["pairs"]:
        label = f"{p['blue']} vs {p['red']}"
        lines.append(f"{label:<{pw}s}{p['mean']:>+10.3f} ± {p['se']:.3f}{p['trunc']:>8.3f}"
                     f"   {p['median']:g} ({p['q1']:g}–{p['q3']:g})")
    lines.append("")

    boot = an["support_freq"] is not None
    lines.append(f"Equilibrium mixture (LP; game value {an['value']:+.2e}, 0 for a symmetric game)"
                 + (f" and bootstrap ({an['n_boot']} resamples, support = weight > {SUPPORT_TOL})" if boot else ""))
    hdr = f"{'policy':<{w}s}{'weight':>8s}"
    if boot:
        hdr += f"{'support freq':>14s}{'mean weight':>13s}"
    hdr += "   worst case (row min of A)"
    lines.append(hdr)
    for i in range(k):
        j = int(np.argmin(A[i]))
        worst = f"{A[i, j]:+.3f}" + (f" ± {SE[i, j]:.3f} vs {names[j]}" if j != i else " vs itself")
        row = f"{names[i]:<{w}s}{an['x'][i]:>8.3f}"
        if boot:
            row += f"{an['support_freq'][i]:>14.2f}{an['mean_weight'][i]:>13.3f}"
        lines.append(row + "   " + worst)
    return "\n".join(lines)


# ---------------------------------------------------------------- main
def run_metagame(cfg: Config, names: list[str], games: int, seeds: list[int], workers: int, out_dir: Path,
                 tag: str = "", extra: dict | None = None) -> tuple[dict, Path]:
    """Play every pair (`games` per seed, pooled), save the stamped per-game results as
    out_dir/metagame_<timestamp>[_<tag>].json, and return (data, path). `extra` is stored in the
    JSON alongside the results."""
    for name in names:
        get_policy(name)                                         # fail fast on a bad name
    seeds = [int(s) for s in seeds]
    print(f"playing {len(names) * (len(names) - 1) // 2} pairs x {games} games x {len(seeds)} seed(s)"
          f" on {workers} workers" + (f" ({tag})" if tag else ""), file=sys.stderr)
    stamp = git_stamp()                                          # before playing: the code that ran
    t0 = time.perf_counter()
    pairs = play_pairs(cfg, names, games, seeds, workers)
    elapsed = time.perf_counter() - t0
    created = datetime.now().strftime("%Y%m%d-%H%M%S")
    data = {"created": created, "git": stamp, "elapsed_s": round(elapsed, 1), "seeds": seeds, "games": games,
            "policies": names, "config": dataclasses.asdict(cfg), **(extra or {}), "pairs": pairs}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"metagame_{created}{'_' + tag if tag else ''}.json"
    path.write_text(json.dumps(data))
    return data, path


def resolve_seeds(ap: argparse.ArgumentParser, args: argparse.Namespace) -> list[int]:
    if args.seeds is not None and args.seed is not None:
        ap.error("give either --seeds or --seed, not both")
    return args.seeds if args.seeds is not None else [0 if args.seed is None else args.seed]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES))
    ap.add_argument("--games", type=int, default=2000, help="games per pair per seed")
    ap.add_argument("--seeds", type=int, nargs="+", help="seeds whose games are pooled (default: 0)")
    ap.add_argument("--seed", type=int, help="a single seed; same as --seeds SEED")
    ap.add_argument("--boot", type=int, default=200, help="bootstrap resamples (0 to skip)")
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="out", help="directory for the per-game JSON")
    ap.add_argument("--load", metavar="JSON", help="recompute the tables from a saved run instead of playing")
    ap.add_argument("--config", nargs="*", default=[], metavar="KEY=VALUE", help="Config overrides")
    args = ap.parse_args()

    if args.load:
        data = json.loads(Path(args.load).read_text())
        print(f"loaded {args.load}\n")
    else:
        data, path = run_metagame(config_from_overrides(args.config), list(args.policies), args.games,
                                  resolve_seeds(ap, args), args.workers, Path(args.out))
        print(f"played in {data['elapsed_s']:.0f}s; saved per-game results to {path}\n")

    print(format_report(analyse(data, args.boot, args.boot_seed), data))


if __name__ == "__main__":
    main()
