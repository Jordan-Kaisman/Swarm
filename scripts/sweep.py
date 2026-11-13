"""Meta-game sweep over arena scale and base HP (spec §12).

    python scripts/sweep.py                                        # 3 scales x 3 base_hp, 1000 games per pair
    python scripts/sweep.py --scales 1.0 --base-hp 5 --games 200    # one quick config
    python scripts/sweep.py --load out/metagame_*_s*.json           # summary from saved runs

For each scale s and base HP, the arena and the layout distances scale with s (width 30s,
height 20s, base_min_mid 5s, base_min_wall 2s, base_min_sep 4s, zone_mid_min 3s,
zone_mid_max 8s, zone_wall_margin 2s, max_steps round(300s)); the drone parameters (step
length, turning, weapon range and cone, hit model) do not. Each config runs the full
meta-game (scripts/metagame.py) and keeps its own per-game JSON; one summary table follows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

import _path  # noqa: F401
import metagame
from drone_battle.config import Config

SCALED = {"width": 30.0, "height": 20.0, "base_min_mid": 5.0, "base_min_wall": 2.0, "base_min_sep": 4.0,
          "zone_mid_min": 3.0, "zone_mid_max": 8.0, "zone_wall_margin": 2.0}
BASE_MAX_STEPS = 300
SHOW_WEIGHT = 0.01                       # equilibrium weights below this are left blank
ABBREV = {"random": "RN", "rusher": "RU", "hunter": "HU", "defender": "DE", "raid": "RA",
          "raid_strafe": "RS", "split": "SP"}
DOUBLE_KO = ("mutual_elimination", "simultaneous_knockout")


def scaled_config(scale: float, base_hp: int) -> Config:
    kw = {k: round(v * scale, 6) for k, v in SCALED.items()}
    return Config(**kw, max_steps=round(BASE_MAX_STEPS * scale), base_hp=int(base_hp))


def abbrev(name: str) -> str:
    return ABBREV.get(name, name[:2].upper())


def summarize_run(data: dict, n_boot: int, boot_seed: int) -> dict:
    """One summary row: equilibrium, bootstrap support, game lengths, truncation, how games end."""
    an = metagame.analyse(data, n_boot, boot_seed)
    pairs = data["pairs"]
    lengths = np.concatenate([p["length"] for p in pairs])
    trunc = [float(np.mean(p["truncated"])) for p in pairs]
    worst = int(np.argmax(trunc))
    causes = [c for p in pairs for c, t in zip(p["cause"], p["truncated"]) if not t]
    ended = max(len(causes), 1)
    sweep = data.get("sweep", {})
    return {
        "scale": sweep.get("scale", data["config"]["width"] / SCALED["width"]),
        "base_hp": sweep.get("base_hp", data["config"]["base_hp"]),
        "names": an["names"], "x": an["x"], "support_freq": an["support_freq"],
        "support_size": int(np.sum(an["x"] >= SHOW_WEIGHT)),
        "median_length": float(np.median(lengths)),
        "max_trunc": trunc[worst], "max_trunc_pair": (pairs[worst]["blue"], pairs[worst]["red"]),
        "ko_shares": (causes.count("bases_destroyed") / ended, causes.count("elimination") / ended,
                      sum(causes.count(c) for c in DOUBLE_KO) / ended),
        "runtime": data.get("elapsed_s", float("nan")),
    }


def _frac(v: float) -> str:
    """Compact 0..1 value: '.73', '1.0', '0'."""
    if v >= 0.995:
        return "1.0"
    return "0" if v < 0.005 else f"{v:.2f}".lstrip("0")


def format_summary(rows: list[dict], n_boot: int) -> str:
    names = rows[0]["names"]
    k = len(names)
    c = 5
    head = "".join(f"{abbrev(n):>{c}s}" for n in names)
    eq_w = c * k + 4
    lines = [
        f"{'':11s}{'equilibrium weight':^{eq_w}s} {'bootstrap support freq':^{c * k}s}",
        f"{'s':>5s}{'hp':>4s}  {head}{'n':>4s} {head}{'med len':>9s}  {'max trunc (pair)':<17s}"
        f"{'KO base/elim/dbl':>18s}{'time':>7s}",
    ]
    for r in rows:
        weights = "".join(f"{_frac(w) if w >= SHOW_WEIGHT else '':>{c}s}" for w in r["x"])
        freqs = "".join(f"{_frac(f):>{c}s}" for f in r["support_freq"]) if r["support_freq"] is not None else ""
        b, e, d = (100 * v for v in r["ko_shares"])
        pair = "-".join(abbrev(n) for n in r["max_trunc_pair"])
        lines.append(f"{r['scale']:>5.2f}{r['base_hp']:>4d}  {weights}{r['support_size']:>4d} {freqs}"
                     f"{r['median_length']:>9g}  {r['max_trunc']:.3f} {pair:<11s}"
                     f"{f'{b:.0f}/{e:.0f}/{d:.0f}%':>18s}{r['runtime']:>6.0f}s")
    legend = ", ".join(f"{abbrev(n)} {n}" for n in names)
    lines += ["", f"{legend}.",
              f"Equilibrium weights below {SHOW_WEIGHT} are blank; n = number of policies at or above it."
              f" Support freq: share of {n_boot} bootstrap resamples with weight > {metagame.SUPPORT_TOL}.",
              "med len: median length over all games of all non-mirror pairs. KO: how the non-truncated games"
              " ended (base knockout / elimination / double knockout)."]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.33, 1.67])
    ap.add_argument("--base-hp", type=int, nargs="+", default=[3, 5, 8])
    ap.add_argument("--policies", nargs="+", default=list(metagame.DEFAULT_POLICIES))
    ap.add_argument("--games", type=int, default=1000, help="games per pair")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="out")
    ap.add_argument("--load", nargs="+", metavar="JSON", help="summarize saved runs instead of playing")
    args = ap.parse_args()

    rows = []
    if args.load:
        for path in args.load:
            rows.append(summarize_run(json.loads(Path(path).read_text()), args.boot, args.boot_seed))
    else:
        for s in args.scales:
            for hp in args.base_hp:
                cfg = scaled_config(s, hp)
                tag = f"s{s:g}_hp{hp}"
                data, path = metagame.run_metagame(cfg, list(args.policies), args.games, args.seed, args.workers,
                                                   Path(args.out), tag=tag, extra={"sweep": {"scale": s, "base_hp": hp}})
                print(f"{tag}: {data['elapsed_s']:.0f}s, saved {path}", file=sys.stderr, flush=True)
                rows.append(summarize_run(data, args.boot, args.boot_seed))
    print(format_summary(rows, args.boot))


if __name__ == "__main__":
    main()
