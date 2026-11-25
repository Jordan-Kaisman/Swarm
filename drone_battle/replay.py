"""Export a recorded episode as a self-contained browser replay."""

from __future__ import annotations

import json
from importlib.resources import files
from math import degrees
from pathlib import Path

from .actions import direction_table
from .config import Config
from .match import EpisodeResult

MOVE_NAMES = ("East", "Northeast", "North", "Northwest", "West", "Southwest", "South", "Southeast")


def _replay_data(result: EpisodeResult, cfg: Config, blue: str, red: str, seed: int | None) -> dict:
    if not result.trajectory:
        raise ValueError("The episode must be played with record=True to export a replay.")

    initial = result.trajectory[0][0]
    frames = []
    cumulative = 0.0
    for state, info in result.trajectory:
        frame = {
            "t": state.t,
            "pos": state.pos.tolist(),
            "heading": state.heading.tolist(),
            "alive": state.alive.tolist(),
            "hp": state.base_hp.tolist(),
            "cumulative": cumulative,
        }
        if info is not None:
            actions = info.get("actions")
            reward = info.get("reward")
            cumulative = cumulative + reward if cumulative is not None and reward is not None else None
            frame.update(
                actions=None if actions is None else actions.tolist(),
                reward=reward,
                cumulative=cumulative,
                disp=info["disp"].tolist(),
                shots=info["shots"],
                dronesKilled=info["drones_killed"].tolist(),
                basesHit=info["bases_hit"].tolist(),
                basesDestroyed=info["bases_destroyed"].tolist(),
            )
        frames.append(frame)

    return {
        "width": cfg.width,
        "height": cfg.height,
        "dronesPerTeam": initial.n,
        "basesPerTeam": initial.B,
        "basePos": initial.base_pos.tolist(),
        "baseValue": initial.base_value.tolist(),
        "baseMaxHp": initial.base_hp.tolist(),
        "range": initial.weapon_range.tolist(),
        "cone": initial.weapon_cone.tolist(),
        "moveNames": ("Hover", *MOVE_NAMES[::8 // cfg.move_dirs]),
        "moveVectors": direction_table(cfg).tolist(),
        "turnDegrees": degrees(cfg.turn_delta),
        "policies": (blue, red),
        "seed": seed,
        "winner": result.winner,
        "cause": result.cause,
        "finalReturn": result.blue_return,
        "frames": frames,
    }


def write_replay(result: EpisodeResult, cfg: Config, path: str | Path, *,
                 blue: str = "Blue", red: str = "Red", seed: int | None = None) -> Path:
    """Write a replay that opens directly in a browser, with no server required."""
    data = _replay_data(result, cfg, blue, red, seed)
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    assets = files("drone_battle") / "replay_assets"
    html = (assets / "replay.html").read_text(encoding="utf-8")
    html = html.replace("/* REPLAY_CSS */", (assets / "replay.css").read_text(encoding="utf-8"))
    html = html.replace("/* REPLAY_JS */", (assets / "replay.js").read_text(encoding="utf-8"))
    html = html.replace("<!-- REPLAY_DATA -->", payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
