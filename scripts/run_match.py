"""Play scripted policies against each other and report statistics (spec §12).

    python scripts/run_match.py --blue rusher --red hunter --games 1000 --seed 0
    python scripts/run_match.py --blue rusher --red hunter --games 1 --render out.mp4
    python scripts/run_match.py --blue hunter --red hunter --games 2000 --hist lengths.png
    python scripts/run_match.py --blue defender:r_def=6 --red rusher --config base_hp=8 n_drones=4

Policies: random, rusher, hunter, defender (optionally name:key=value,...).
--config takes Config overrides as key=value (values may use `pi`) or a JSON file.
"""

from __future__ import annotations

import argparse
import time

import _path  # noqa: F401
from drone_battle.config import config_from_overrides
from drone_battle.match import (KNOCKOUT_ROWS, RETURN_OUTCOMES, TRUNCATED, game_seeds, knockout_bucket,
                                play_episode, summarize)
from drone_battle.policies import POLICIES, get_policy

CAUSE_LABELS = {
    "bases_destroyed": "bases destroyed",
    "elimination": "elimination",
    "mutual_elimination": "mutual elimination",
    "simultaneous_knockout": "simultaneous knockout",
    "truncation": "truncation",
}
RETURN_LABELS = {"blue_ahead": "blue ahead", "even": "even", "red_ahead": "red ahead"}


def _row_label(bucket: str, cause: str) -> str:
    return bucket if bucket == TRUNCATED else f"{bucket} / {CAUSE_LABELS.get(cause, cause)}"


def _pct(x: int, total: int) -> str:
    return f"{x:5d} ({100 * x / total:5.1f}%)" if total else f"{x:5d}"


def _rate(hits: int, shots: int) -> str:
    return f"{hits / shots:.3f}" if shots else "  n/a"


def format_report(stats: dict, blue: str, red: str, seed: int, elapsed: float) -> str:
    G = stats["games"]
    lines = [f"{blue} (blue) vs {red} (red): {G} games, seed {seed}, {elapsed:.1f}s", ""]

    lines.append("Knockout outcome (blue's perspective)")
    for bucket in ("win", "loss", "draw", "truncated"):
        count = stats["counts"][bucket]
        by_cause = ", ".join(f"{CAUSE_LABELS.get(c, c)} {k}" for c, k in sorted(stats["outcomes"][bucket].items()))
        lines.append(f"  {bucket:<12s}{_pct(count, G)}" + (f"   [{by_cause}]" if by_cause else ""))
    lines.append("  (truncation is the stalemate detector, not a game outcome)")
    lines.append("")

    lines.append("Return-based outcome (sign of blue's undiscounted return)")
    for ro in RETURN_OUTCOMES:
        lines.append(f"  {RETURN_LABELS[ro]:<12s}{_pct(stats['return_outcomes'][ro], G)}")
    lines.append("")

    w = 32
    lines.append(f"{'Knockout vs return':<{w}s}" + "".join(f"{RETURN_LABELS[ro]:>12s}" for ro in RETURN_OUTCOMES)
                 + f"{'total':>9s}")
    for row in KNOCKOUT_ROWS:
        cells = stats["crosstab"][row]
        lines.append(f"  {_row_label(*row):<{w - 2}s}" + "".join(f"{cells[ro]:12d}" for ro in RETURN_OUTCOMES)
                     + f"{sum(cells.values()):9d}")
    lines.append(f"  {'total':<{w - 2}s}" + "".join(f"{stats['return_outcomes'][ro]:12d}" for ro in RETURN_OUTCOMES)
                 + f"{G:9d}")
    dis = {row: k for row, k in stats["disagreements"].items() if k}
    where = ", ".join(f"{_row_label(*row)} {k}" for row, k in dis.items())
    lines.append(f"  disagreements: {sum(dis.values())}" + (f"   [{where}]" if where else ""))
    lines.append("")

    lines.append(f"Episode length     mean {stats['length_mean']:.1f} +- {stats['length_std']:.1f} (std),"
                 f" median {stats['length_median']:g}, IQR {stats['length_q1']:g}-{stats['length_q3']:g}"
                 f"   truncated fraction {stats['truncated_frac']:.3f}")
    lines.append(f"Blue mean return   {stats['return_mean']:+.4f} +- {stats['return_se']:.4f} (SE)")
    lines.append("")
    lines.append(f"{'per game':<20s}{'blue':>10s}{'red':>10s}")
    dk, bk = stats["drone_kills"], stats["base_kills"]
    sd, hd, sb, hb = stats["shots_drone"], stats["hits_drone"], stats["shots_base"], stats["hits_base"]
    rows = [
        ("drone kills", dk / G, "{:10.2f}"),
        ("base kills", bk / G, "{:10.2f}"),
        ("shots fired", (sd + sb) / G, "{:10.1f}"),
        ("  at drones", sd / G, "{:10.1f}"),
        ("  at bases", sb / G, "{:10.1f}"),
    ]
    for label, vals, fmt in rows:
        lines.append(f"{label:<20s}" + "".join(fmt.format(v) for v in vals))
    lines.append(f"{'hit rate (overall)':<20s}" + "".join(f"{_rate(h, s):>10s}" for h, s in zip(hd + hb, sd + sb)))
    lines.append(f"{'hit rate (drones)':<20s}" + "".join(f"{_rate(h, s):>10s}" for h, s in zip(hd, sd)))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blue", default="rusher", help=f"one of {sorted(POLICIES)}")
    ap.add_argument("--red", default="hunter", help=f"one of {sorted(POLICIES)}")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", metavar="PATH", help="animate the first game (.mp4, or .gif fallback)")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--hist", metavar="PATH", help="save a histogram of episode lengths (e.g. lengths.png)")
    ap.add_argument("--config", nargs="*", default=[], metavar="KEY=VALUE", help="Config overrides")
    args = ap.parse_args()

    cfg = config_from_overrides(args.config)
    blue, red = get_policy(args.blue), get_policy(args.red)

    t0 = time.perf_counter()
    results = []
    for g, seed in enumerate(game_seeds(args.seed, args.games)):
        record = bool(args.render) and g == 0
        res = play_episode(cfg, blue, red, seed, record=record)
        if record:
            from drone_battle.render import animate
            path = animate(res.trajectory, cfg, args.render, fps=args.fps)
            print(f"rendered game 0 ({res.length} steps, {res.cause}) to {path}")
            res.trajectory = None
        results.append(res)
    elapsed = time.perf_counter() - t0

    stats = summarize(results)
    print(format_report(stats, args.blue, args.red, args.seed, elapsed))
    if args.hist:
        from drone_battle.render import length_histogram
        path = length_histogram(stats["lengths"], [knockout_bucket(r) for r in results], args.hist,
                                title=f"{args.blue} (blue) vs {args.red} (red): episode length")
        print(f"\nsaved episode-length histogram to {path}")


if __name__ == "__main__":
    main()
