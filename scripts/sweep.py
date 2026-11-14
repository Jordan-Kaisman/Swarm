"""Meta-game sweeps over Config fields (spec §12).

    # Field sweeps at the current defaults: --grid fields form a product; each --zip group of
    # fields is varied together (equal-length lists).
    python scripts/sweep.py --zip alpha_target_motion=0.5,0.35,0.25 beta_shooter_motion=0.25,0.35,0.5
    python scripts/sweep.py --grid base_hp=3,5,8 n_drones=2,3 --seeds 0 1 --games 500

    # The scale x base_hp preset, anchored at the original 30x20 arena so earlier results
    # reproduce. It is the default when no --grid or --zip is given.
    python scripts/sweep.py --preset scale_hp --scales 1.0 1.33 1.67 --base-hp 3 5 8

    # Summary from saved runs, without re-running.
    python scripts/sweep.py --load out/metagame_*_s*.json

Values are comma-separated and parsed like --config values (pi, true, arithmetic). For a
tuple-valued field, give each value in parentheses: base_values=(1,1),(1,2).

Preset scale_hp: for each scale s and base HP, the arena and the layout distances scale with s
(width 30s, height 20s, base_min_mid 5s, base_min_wall 2s, base_min_sep 4s, zone_mid_min 3s,
zone_mid_max 8s, zone_wall_margin 2s, max_steps round(300s)); the drone parameters (step length,
turning, weapon range and cone, hit model) do not.

Each config runs the full meta-game (scripts/metagame.py, seeds pooled) and keeps its own stamped
per-game JSON; one summary table follows.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

import _path  # noqa: F401
import metagame
from drone_battle.config import Config, config_from_overrides, parse_value

# Anchored to the original 30x20 arena (not the current Config defaults) so earlier sweep results reproduce.
SCALED = {"width": 30.0, "height": 20.0, "base_min_mid": 5.0, "base_min_wall": 2.0, "base_min_sep": 4.0,
          "zone_mid_min": 3.0, "zone_mid_max": 8.0, "zone_wall_margin": 2.0}
BASE_MAX_STEPS = 300
SHOW_WEIGHT = 0.01                       # equilibrium weights below this are left blank
ABBREV = {"random": "RN", "rusher": "RU", "hunter": "HU", "defender": "DE", "raid": "RA",
          "raid_strafe": "RS", "raid_angle": "RG", "split": "SP"}
SHORT = {"scale": "s", "base_hp": "hp", "alpha_target_motion": "α", "beta_shooter_motion": "β"}
TAG = {"scale": "s", "base_hp": "hp"}    # file-name prefixes; other fields use their initials
DOUBLE_KO = ("mutual_elimination", "simultaneous_knockout")
CONFIG_FIELDS = {f.name for f in dataclasses.fields(Config)}


# ---------------------------------------------------------------- configs
def scaled_config(scale: float, base_hp: int) -> Config:
    kw = {k: round(v * scale, 6) for k, v in SCALED.items()}
    return Config(**kw, max_steps=round(BASE_MAX_STEPS * scale), base_hp=int(base_hp))


def preset_scale_hp(scales: list[float], base_hps: list[int]) -> list[tuple[dict, Config]]:
    return [({"scale": s, "base_hp": hp}, scaled_config(s, hp)) for s in scales for hp in base_hps]


def parse_values(item: str) -> tuple[str, list]:
    """'FIELD=V1,V2,...' -> (FIELD, [V1, V2, ...])."""
    key, sep, raw = item.partition("=")
    key = key.strip()
    if not sep or not raw.strip():
        raise ValueError(f"expected FIELD=V1,V2,... but got {item!r}")
    if key not in CONFIG_FIELDS:
        raise ValueError(f"unknown Config field {key!r}")
    value = parse_value(raw)
    return key, list(value) if isinstance(value, tuple) else [value]


def build_axes(grid: list[str] | None, zips: list[list[str]] | None) -> list[list[dict]]:
    """Each axis is a list of {field: value} settings: one field per --grid item, several fields
    per --zip group. The configs are the product over the axes."""
    axes: list[list[dict]] = []
    seen: set[str] = set()

    def claim(key: str) -> None:
        if key in seen:
            raise ValueError(f"field {key!r} is swept more than once")
        seen.add(key)

    for item in grid or []:
        key, values = parse_values(item)
        claim(key)
        axes.append([{key: v} for v in values])
    for group in zips or []:
        parsed = [parse_values(item) for item in group]
        if len({len(values) for _, values in parsed}) != 1:
            raise ValueError(f"zipped fields need value lists of equal length: {' '.join(group)}")
        keys = [key for key, _ in parsed]
        for key in keys:
            claim(key)
        axes.append([dict(zip(keys, combo)) for combo in zip(*(values for _, values in parsed))])
    return axes


def field_settings(base: Config, axes: list[list[dict]]) -> list[tuple[dict, Config]]:
    settings = []
    for combo in itertools.product(*axes):
        params = {k: v for d in combo for k, v in d.items()}
        settings.append((params, base.replace(**params)))
    return settings


def _fmt(v) -> str:
    return f"{v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def tag_for(params: dict) -> str:
    parts = [TAG.get(k, "".join(w[0] for w in k.split("_"))) + _fmt(v) for k, v in params.items()]
    return re.sub(r"[^A-Za-z0-9._+-]+", "-", "_".join(parts)).strip("-")


# ---------------------------------------------------------------- summary
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
    return {
        "params": dict(data.get("sweep", {})),
        "names": an["names"], "x": an["x"], "support_freq": an["support_freq"], "A": an["A"], "SE": an["SE"],
        "support_size": int(np.sum(an["x"] >= SHOW_WEIGHT)),
        "median_length": float(np.median(lengths)),
        "max_trunc": trunc[worst], "max_trunc_pair": (pairs[worst]["blue"], pairs[worst]["red"]),
        "ko_shares": (causes.count("bases_destroyed") / ended, causes.count("elimination") / ended,
                      sum(causes.count(c) for c in DOUBLE_KO) / ended),
        "runtime": data.get("elapsed_s", float("nan")),
        "seeds_line": metagame.seeds_line(data), "stamp_line": metagame.stamp_line(data),
    }


def _frac(v: float) -> str:
    """Compact 0..1 value: '.73', '1.0', '0'."""
    if v >= 0.995:
        return "1.0"
    return "0" if v < 0.005 else f"{v:.2f}".lstrip("0")


def _union(lists) -> list:
    out: list = []
    for items in lists:
        out += [x for x in items if x not in out]
    return out


def format_summary(rows: list[dict], n_boot: int) -> str:
    keys = _union(r["params"].keys() for r in rows)
    names = _union(r["names"] for r in rows)
    widths = {k: max(len(SHORT.get(k, k)), *(len(_fmt(r["params"].get(k, "-"))) for r in rows)) + 2 for k in keys}
    pw = sum(widths.values())
    c, k = 5, len(names)
    head = "".join(f"{abbrev(n):>{c}s}" for n in names)
    lines = [
        f"{'':{pw}s}  {'equilibrium weight':^{c * k + 4}s} {'bootstrap support freq':^{c * k}s}",
        "".join(f"{SHORT.get(key, key):>{widths[key]}s}" for key in keys)
        + f"  {head}{'n':>4s} {head}{'med len':>9s}  {'max trunc (pair)':<17s}{'KO base/elim/dbl':>18s}{'time':>7s}",
    ]
    for r in rows:
        col = {n: i for i, n in enumerate(r["names"])}
        weights = "".join(f"{_frac(r['x'][col[n]]) if n in col and r['x'][col[n]] >= SHOW_WEIGHT else '':>{c}s}"
                          for n in names)
        freqs = ("".join(f"{_frac(r['support_freq'][col[n]]) if n in col else '':>{c}s}" for n in names)
                 if r["support_freq"] is not None else " " * (c * k))
        b, e, d = (100 * v for v in r["ko_shares"])
        pair = "-".join(abbrev(n) for n in r["max_trunc_pair"])
        lines.append("".join(f"{_fmt(r['params'].get(key, '-')):>{widths[key]}s}" for key in keys)
                     + f"  {weights}{r['support_size']:>4d} {freqs}{r['median_length']:>9g}"
                     f"  {r['max_trunc']:.3f} {pair:<11s}{f'{b:.0f}/{e:.0f}/{d:.0f}%':>18s}{r['runtime']:>6.0f}s")
    legend = ", ".join(f"{abbrev(n)} {n}" for n in names)
    lines += ["", f"{legend}.",
              f"Equilibrium weights below {SHOW_WEIGHT} are blank; n = number of policies at or above it."
              f" Support freq: share of {n_boot} bootstrap resamples with weight > {metagame.SUPPORT_TOL}.",
              "med len: median length over all games of all non-mirror pairs. KO: how the non-truncated games"
              " ended (base knockout / elimination / double knockout)."]
    for label, field in (("Games", "seeds_line"), ("Code", "stamp_line")):
        values = _union([r[field]] for r in rows)
        if field == "stamp_line":                     # one line per distinct commit, not per run time
            values = _union([v.split(", created")[0]] for v in values)
        lines.append(f"{label}: " + "; ".join(values) + ("" if len(values) == 1 else " (differs between runs)"))
    return "\n".join(lines)


def format_payoff_rows(rows: list[dict], policies: list[str]) -> str:
    """Per config, the payoff rows A[i, :] ± SE of the named policies."""
    cell = 16
    lines = ["Payoff rows: mean return of the row policy against each column policy (± SE)"]
    for r in rows:
        names = r["names"]
        w = max(len(n) for n in names) + 2
        label = ", ".join(f"{SHORT.get(k, k)} {_fmt(v)}" for k, v in r["params"].items()) or "config"
        lines += ["", f"{label:<{w}s}" + "".join(f"{n:>{cell}s}" for n in names)]
        for p in policies:
            if p not in names:
                lines.append(f"{p:<{w}s}  (not in this run)")
                continue
            i = names.index(p)
            A, SE = r["A"], r["SE"]
            cells = ["0" if i == j else f"{A[i, j]:+.3f} ± {SE[i, j]:.3f}" for j in range(len(names))]
            lines.append(f"{p:<{w}s}" + "".join(f"{x:>{cell}s}" for x in cells))
    return "\n".join(lines)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", nargs="+", metavar="FIELD=V1,V2", help="fields swept as a product")
    ap.add_argument("--zip", nargs="+", action="append", metavar="FIELD=V1,V2",
                    help="fields varied together (equal-length lists); may be given more than once")
    ap.add_argument("--preset", choices=["scale_hp"],
                    help="the scale x base_hp grid anchored at 30x20 (the default without --grid/--zip)")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.33, 1.67], help="preset scale_hp")
    ap.add_argument("--base-hp", type=int, nargs="+", default=[3, 5, 8], help="preset scale_hp")
    ap.add_argument("--config", nargs="*", default=[], metavar="KEY=VALUE",
                    help="base Config overrides for --grid/--zip sweeps")
    ap.add_argument("--policies", nargs="+", default=list(metagame.DEFAULT_POLICIES))
    ap.add_argument("--games", type=int, default=1000, help="games per pair per seed")
    ap.add_argument("--seeds", type=int, nargs="+", help="seeds whose games are pooled (default: 0)")
    ap.add_argument("--seed", type=int, help="a single seed; same as --seeds SEED")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="out")
    ap.add_argument("--payoff-rows", nargs="+", metavar="POLICY", help="also print these policies' payoff rows")
    ap.add_argument("--load", nargs="+", metavar="JSON", help="summarize saved runs instead of playing")
    args = ap.parse_args()

    rows = []
    if args.load:
        for path in args.load:
            rows.append(summarize_run(json.loads(Path(path).read_text()), args.boot, args.boot_seed))
    else:
        if (args.grid or args.zip) and args.preset:
            ap.error("--preset cannot be combined with --grid/--zip")
        try:
            if args.grid or args.zip:
                settings = field_settings(config_from_overrides(args.config), build_axes(args.grid, args.zip))
            else:
                if args.config:
                    ap.error("--config applies to --grid/--zip sweeps; the scale_hp preset is fixed")
                settings = preset_scale_hp(args.scales, args.base_hp)
        except ValueError as e:
            ap.error(str(e))
        seeds = metagame.resolve_seeds(ap, args)
        for params, cfg in settings:
            tag = tag_for(params)
            data, path = metagame.run_metagame(cfg, list(args.policies), args.games, seeds, args.workers,
                                               Path(args.out), tag=tag, extra={"sweep": params})
            print(f"{tag}: {data['elapsed_s']:.0f}s, saved {path}", file=sys.stderr, flush=True)
            rows.append(summarize_run(data, args.boot, args.boot_seed))
    print(format_summary(rows, args.boot))
    if args.payoff_rows:
        print("\n" + format_payoff_rows(rows, args.payoff_rows))


if __name__ == "__main__":
    main()
