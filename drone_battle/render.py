"""Matplotlib rendering (spec §10) and the episode-length histogram (spec §12)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from matplotlib.patches import Polygon, Rectangle, Wedge  # noqa: E402

from .config import Config  # noqa: E402
from .state import State  # noqa: E402

TEAM_COLORS = ("#2464c4", "#d0392f")
TEAM_NAMES = ("blue", "red")
DEAD_COLOR = "#8a8a8a"
BASE_SIDE = 1.0      # drawn square side, in step lengths
DRONE_LEN = 0.7      # drawn triangle length
DRONE_HALF_W = 0.28  # drawn triangle half-width

# Match-statistics plots. Outcome colors are the first four slots of a validated categorical
# palette (adjacent pairs pass the colorblind and normal-vision separation checks); two of them
# are low-contrast on white, so the legend always carries the counts as text.
OUTCOME_ORDER = ("win", "loss", "draw", "truncated")
OUTCOME_COLORS = {"win": "#2a78d6", "loss": "#eb6834", "draw": "#1baf7a", "truncated": "#eda100"}
PLOT_SURFACE, PLOT_INK, PLOT_INK_MUTED, PLOT_GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
HIST_MAX_BINS = 60


def _drone_triangle(x: float, y: float, psi: float) -> list[tuple[float, float]]:
    c, s = math.cos(psi), math.sin(psi)
    nose = (x + DRONE_LEN * 0.6 * c, y + DRONE_LEN * 0.6 * s)
    back_x, back_y = x - DRONE_LEN * 0.4 * c, y - DRONE_LEN * 0.4 * s
    left = (back_x - DRONE_HALF_W * s, back_y + DRONE_HALF_W * c)
    right = (back_x + DRONE_HALF_W * s, back_y - DRONE_HALF_W * c)
    return [nose, left, right]


def render(state: State, cfg: Config, info: dict | None = None, ax=None, title: str | None = None):
    """Draw one frame. Returns the matplotlib Axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6.4))
    W, H = cfg.width, cfg.height
    n, B = state.n, state.B

    # Arena, halves, midline.
    ax.add_patch(Rectangle((0, 0), W / 2, H, color=TEAM_COLORS[0], alpha=0.07, lw=0, zorder=0))
    ax.add_patch(Rectangle((W / 2, 0), W / 2, H, color=TEAM_COLORS[1], alpha=0.07, lw=0, zorder=0))
    ax.add_patch(Rectangle((0, 0), W, H, fill=False, ec="#333333", lw=1.2, zorder=1))
    ax.plot([W / 2, W / 2], [0, H], color="#555555", lw=0.8, ls=(0, (4, 4)), zorder=1)

    # Shots (drawn under the drones).
    if info is not None:
        for shooter, kind, j, p, hit in info.get("shots", []):
            sx, sy = state.pos[shooter]
            tx, ty = state.pos[j] if kind == "drone" else state.base_pos[j]
            color = TEAM_COLORS[shooter // n]
            ax.plot([sx, tx], [sy, ty], color=color, lw=1.6 if hit else 0.9,
                    ls="-" if hit else (0, (3, 2)), alpha=0.9 if hit else 0.6, zorder=2)

    # Bases.
    for b in range(2 * B):
        x, y = state.base_pos[b]
        team = b // B
        hp = int(state.base_hp[b])
        if hp > 0:
            alpha = 0.3 + 0.7 * hp / cfg.base_hp
            ax.add_patch(Rectangle((x - BASE_SIDE / 2, y - BASE_SIDE / 2), BASE_SIDE, BASE_SIDE,
                                   fc=TEAM_COLORS[team], ec="black", lw=0.8, alpha=alpha, zorder=3))
            ax.text(x, y - BASE_SIDE * 0.95, f"{hp}/{state.base_value[b]:g}", ha="center", va="top",
                    fontsize=7, color="#222222", zorder=5)
        else:
            ax.plot([x], [y], marker="x", ms=11, mew=2.2, color=DEAD_COLOR, zorder=3)

    # Drones with weapon cones.
    for i in range(2 * n):
        x, y = state.pos[i]
        team = i // n
        if state.alive[i]:
            psi = float(state.heading[i])
            half = math.degrees(0.5 * float(state.weapon_cone[i]))
            ax.add_patch(Wedge((x, y), float(state.weapon_range[i]), math.degrees(psi) - half,
                               math.degrees(psi) + half, fc=TEAM_COLORS[team], ec=TEAM_COLORS[team],
                               alpha=0.10, lw=0.5, zorder=2))
            ax.add_patch(Polygon(_drone_triangle(x, y, psi), closed=True, fc=TEAM_COLORS[team],
                                 ec="black", lw=0.6, zorder=4))
        else:
            ax.plot([x], [y], marker="x", ms=6, mew=1.5, color=DEAD_COLOR, zorder=4)

    counts = " | ".join(
        f"{TEAM_NAMES[tm]}: {state.alive_drones(tm)}/{n} drones, {state.alive_bases(tm)}/{B} bases"
        for tm in (0, 1))
    head = f"t = {state.t}" + ("  (terminated)" if state.terminated else "")
    ax.set_title(title or f"{head}    {counts}", fontsize=9)
    pad = 0.6
    ax.set_xlim(-pad, W + pad)
    ax.set_ylim(-pad, H + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def render_to_file(state: State, cfg: Config, path: str | Path, info: dict | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6.4))
    render(state, cfg, info, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return Path(path)


def animate(trajectory: Sequence, cfg: Config, path: str | Path, fps: int = 6) -> Path:
    """Write an animation of `trajectory` (States or (State, info) pairs).

    Uses ffmpeg for .mp4 when available; otherwise falls back to a .gif via
    Pillow. Returns the path actually written.
    """
    frames = [(f, None) if isinstance(f, State) else (f[0], f[1]) for f in trajectory]
    path = Path(path)
    if path.suffix.lower() == ".mp4" and animation.writers.is_available("ffmpeg"):
        writer = animation.FFMpegWriter(fps=fps)
    else:
        path = path.with_suffix(".gif")
        writer = animation.PillowWriter(fps=fps)

    fig, ax = plt.subplots(figsize=(9, 6.4))
    fig.tight_layout()

    def draw(k: int):
        ax.clear()
        state, info = frames[k]
        render(state, cfg, info, ax=ax)
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), blit=False)
    anim.save(str(path), writer=writer, dpi=90)
    plt.close(fig)
    return path


def length_histogram(lengths: Sequence[int], buckets: Sequence[str], path: str | Path, *,
                     title: str | None = None) -> Path:
    """Save a histogram of episode lengths, stacked by knockout bucket (blue's perspective).

    `buckets[k]` is one of OUTCOME_ORDER for game k. Bins are whole steps, widened so that
    there are at most HIST_MAX_BINS of them over the terminated games. Truncated games all
    end at the step cap, so they get their own bar to the right of the main range instead of
    stretching the axis. Returns the path written.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    buckets = np.asarray(buckets)
    trunc = buckets == "truncated"
    done = lengths[~trunc]
    hi = int(done.max()) if done.size else 1
    width = max(1, math.ceil(hi / HIST_MAX_BINS))
    edges = 0.5 + width * np.arange(math.ceil(hi / width) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    x_cap = edges[-1] + 1.5 * width    # the truncated games' bar, one bin-width gap to the right

    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=PLOT_SURFACE)
    ax.set_facecolor(PLOT_SURFACE)
    bottom = np.zeros(len(centers))
    for b in OUTCOME_ORDER[:-1]:
        mask = buckets == b
        if not mask.any():
            continue
        counts, _ = np.histogram(lengths[mask], bins=edges)
        ax.bar(centers, counts, width=width, bottom=bottom, color=OUTCOME_COLORS[b],
               edgecolor=PLOT_SURFACE, linewidth=1.0, label=f"{b} ({int(mask.sum())})", zorder=2)
        bottom += counts
    top = bottom.max() if done.size else 0
    ticks = [t for t in MaxNLocator(nbins=8, integer=True).tick_values(0, hi) if 0 <= t <= edges[-1]]
    labels = [f"{t:g}" for t in ticks]
    if trunc.any():
        cap = int(lengths[trunc].max())
        ax.bar([x_cap], [int(trunc.sum())], width=width, color=OUTCOME_COLORS["truncated"],
               edgecolor=PLOT_SURFACE, linewidth=1.0, label=f"truncated ({int(trunc.sum())})", zorder=2)
        top = max(top, int(trunc.sum()))
        ticks.append(x_cap)
        labels.append(f"{cap}\n(cap)")
    ax.set_xticks(ticks, labels)
    ax.set_xlim(edges[0] - 0.5 * width, (x_cap if trunc.any() else edges[-1]) + width)
    ax.set_ylim(0, top * 1.15 if top else 1)

    q1, median, q3 = np.percentile(lengths, [25, 50, 75])
    x_med = median if median <= edges[-1] else x_cap
    ax.axvline(x_med, color=PLOT_INK, linewidth=1.0, linestyle="--", zorder=3)
    x_lo, x_hi = ax.get_xlim()
    right_half = x_med > 0.5 * (x_lo + x_hi)   # put the label on the side with more room
    ax.annotate(f"median {median:g}", (x_med, 1.0), xycoords=("data", "axes fraction"),
                xytext=(-4 if right_half else 4, -2), textcoords="offset points",
                ha="right" if right_half else "left", va="top", fontsize=9, color=PLOT_INK)

    ax.set_xlabel(f"episode length (steps){'' if width == 1 else f', bins of {width}'}",
                  color=PLOT_INK_MUTED)
    ax.set_ylabel("games", color=PLOT_INK_MUTED)
    ax.grid(axis="y", color=PLOT_GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=PLOT_INK_MUTED, length=0, pad=4)
    ax.legend(frameon=False, fontsize=9, labelcolor=PLOT_INK, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    std = lengths.std(ddof=1) if len(lengths) > 1 else 0.0
    subtitle = (f"{len(lengths)} games · mean {lengths.mean():.1f} ± {std:.1f} (std)"
                f" · median {median:g}, IQR {q1:g}–{q3:g}")
    ax.set_title(subtitle, fontsize=9, color=PLOT_INK_MUTED, loc="left")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=PLOT_INK)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=PLOT_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return Path(path)
