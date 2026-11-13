"""Measure step() throughput with random legal actions (spec §14).

    python scripts/bench.py
    python scripts/bench.py --steps 50000 --sizes 4 --config width=12 height=8

Actions are pre-generated so only step() (plus the occasional reset) is timed.
Target: <= ~100 us per step at n = 4.
"""

from __future__ import annotations

import argparse
import time

import _path  # noqa: F401
import numpy as np

from drone_battle.actions import legal_action_array
from drone_battle.config import config_from_overrides
from drone_battle.env import reset, step


def bench(cfg, steps: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    legal = legal_action_array(cfg)
    actions = legal[rng.integers(len(legal), size=(steps, cfg.n_total))]
    state = reset(cfg, rng)
    shots = episodes = 0
    t0 = time.perf_counter()
    for k in range(steps):
        state, _, terminated, truncated, info = step(cfg, state, actions[k], rng)
        shots += len(info["shots"])
        if terminated or truncated:
            state = reset(cfg, rng)
            episodes += 1
    dt = time.perf_counter() - t0
    return {"us_per_step": 1e6 * dt / steps, "steps_per_sec": steps / dt,
            "shots_per_step": shots / steps, "episodes": episodes}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--sizes", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    base = config_from_overrides(args.config)
    print(f"{'n':>3s} {'us/step':>9s} {'steps/s':>10s} {'shots/step':>11s} {'resets':>7s}")
    for n in args.sizes:
        cfg = base.replace(n_drones=n)
        bench(cfg, min(2000, args.steps), args.seed)  # warm-up
        r = bench(cfg, args.steps, args.seed)
        flag = "  <- target <= ~100 us" if n == 4 else ""
        print(f"{n:3d} {r['us_per_step']:9.1f} {r['steps_per_sec']:10.0f} {r['shots_per_step']:11.2f} "
              f"{r['episodes']:7d}{flag}")


if __name__ == "__main__":
    main()
