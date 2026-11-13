"""Render a grid of random reset() layouts (build-order step 1).

    python scripts/show_layouts.py --count 20 --seed 0 --out out/layouts.png
    python scripts/show_layouts.py --config placement=bases n_drones=4
"""

import argparse

import _path  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from drone_battle.config import config_from_overrides
from drone_battle.env import reset
from drone_battle.render import render


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--out", default="out/layouts.png")
    ap.add_argument("--config", nargs="*", default=[], help="Config overrides key=value")
    args = ap.parse_args()

    cfg = config_from_overrides(args.config)
    rng = np.random.default_rng(args.seed)
    rows = -(-args.count // args.cols)
    fig, axes = plt.subplots(rows, args.cols, figsize=(4.2 * args.cols, 3.0 * rows))
    for k, ax in enumerate(np.atleast_1d(axes).ravel()):
        if k < args.count:
            render(reset(cfg, rng), cfg, ax=ax, title=f"layout {k}")
        else:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=80)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
