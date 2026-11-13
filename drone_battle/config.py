"""Frozen environment configuration (spec §3) and command-line override parsing."""

from __future__ import annotations

import ast
import dataclasses
import json
import math
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

MOVE_DIRS = (4, 8)
MOVE_TURN_MODES = ("simultaneous", "exclusive")
FIRE_MODES = ("auto",)
TARGET_PRIORITIES = ("drones_first",)
PLACEMENTS = ("zone", "bases")


@dataclass(frozen=True)
class Config:
    """Every mechanic and constant of the game. Immutable and validated on construction.

    Lengths are in drone step lengths, angles in radians. The per-drone weapon
    parameters accept a scalar (uniform, as in v0) or a length-2n sequence
    (blue rows first).
    """

    # Arena
    width: float = 40.0
    height: float = 27.0
    # Teams
    n_drones: int = 3
    # Dynamics
    step_len: float = 1.0
    turn_delta: float = math.pi / 4
    move_dirs: int = 8
    move_turn_mode: str = "simultaneous"
    # Weapon
    weapon_range: float | tuple[float, ...] = 4.0
    weapon_cone: float | tuple[float, ...] = math.pi / 2
    hit_c: float = 1.0
    hit_pmax: float = 0.9
    alpha_target_motion: float = 0.5
    beta_shooter_motion: float = 0.25
    fire_mode: str = "auto"
    target_priority: str = "drones_first"
    # Sensing (reserved for fog of war; unused in v0)
    sense_range: float = 8.0
    sense_cone: float = 2 * math.pi
    # Bases
    bases_per_team: int = 2
    base_values: tuple[float, ...] = (1.0, 1.0)
    base_hp: int = 5
    base_hit_prob: float = 1.0
    base_min_mid: float = 6.5
    base_min_wall: float = 2.5
    base_min_sep: float = 5.5
    base_layout: tuple[tuple[float, float], ...] | None = None
    # Init
    placement: str = "zone"
    zone_mid_min: float = 4.0
    zone_mid_max: float = 10.5
    zone_wall_margin: float = 2.5
    bases_spawn_radius: float = 2.0
    drone_min_sep: float = 1.5
    mirror_init: bool = True
    random_headings: bool = True
    # Rewards
    drone_value: float = 0.1
    elimination_bonus: bool = True
    # Episode
    max_steps: int = 400

    def __post_init__(self) -> None:
        # Normalize sequences to tuples so the config stays hashable and immutable.
        setattr_ = object.__setattr__
        setattr_(self, "base_values", tuple(float(v) for v in self.base_values))
        if self.base_layout is not None:
            setattr_(self, "base_layout", tuple((float(x), float(y)) for x, y in self.base_layout))
        for name in ("weapon_range", "weapon_cone"):
            value = getattr(self, name)
            if not np.isscalar(value):
                setattr_(self, name, tuple(float(v) for v in value))
        self.validate()

    # ----------------------------------------------------------------- derived
    @property
    def n_total(self) -> int:
        """Total number of drones (2n)."""
        return 2 * self.n_drones

    @property
    def midline(self) -> float:
        return 0.5 * self.width

    def _drone_array(self, value: float | tuple[float, ...]) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(self.n_total, float(arr))
        return arr.copy()

    def weapon_range_array(self) -> np.ndarray:
        """Per-drone weapon range, shape (2n,)."""
        return self._drone_array(self.weapon_range)

    def weapon_cone_array(self) -> np.ndarray:
        """Per-drone full weapon cone angle, shape (2n,)."""
        return self._drone_array(self.weapon_cone)

    def replace(self, **changes: Any) -> "Config":
        return dataclasses.replace(self, **changes)

    # -------------------------------------------------------------- validation
    def validate(self) -> None:
        """Raise ValueError listing every violated constraint."""
        errors: list[str] = []

        def check(cond: bool, msg: str) -> None:
            if not cond:
                errors.append(msg)

        def is_prob(x: float) -> bool:
            return 0.0 <= x <= 1.0

        n, B = self.n_drones, self.bases_per_team
        half = 0.5 * self.width

        check(self.width > 0 and self.height > 0, "width and height must be positive")
        check(isinstance(self.n_drones, int) and n >= 1, "n_drones must be an int >= 1")
        check(isinstance(self.bases_per_team, int) and B >= 1, "bases_per_team must be an int >= 1")
        check(self.step_len > 0, "step_len must be positive")
        check(self.move_dirs in MOVE_DIRS, f"move_dirs must be one of {MOVE_DIRS}")
        check(self.move_turn_mode in MOVE_TURN_MODES, f"move_turn_mode must be one of {MOVE_TURN_MODES}")
        check(self.fire_mode in FIRE_MODES, f"fire_mode must be one of {FIRE_MODES}")
        check(self.target_priority in TARGET_PRIORITIES, f"target_priority must be one of {TARGET_PRIORITIES}")
        check(self.placement in PLACEMENTS, f"placement must be one of {PLACEMENTS}")

        # Weapons (per-drone arrays).
        ranges = np.atleast_1d(np.asarray(self.weapon_range, dtype=float))
        cones = np.atleast_1d(np.asarray(self.weapon_cone, dtype=float))
        for name, arr in (("weapon_range", ranges), ("weapon_cone", cones)):
            check(arr.size in (1, 2 * n), f"{name} must be a scalar or have length 2*n_drones")
        check(bool(np.all(ranges > 0)), "weapon_range must be positive")
        check(bool(np.all((cones > 0) & (cones <= 2 * math.pi))), "weapon_cone must be in (0, 2*pi]")
        check(0 < self.turn_delta <= float(cones.min()) + 1e-12,
              "turn_delta must be in (0, min(weapon_cone)] or some bearings can never be brought into the cone")
        check(self.hit_c > 0, "hit_c must be positive")
        for name in ("hit_pmax", "alpha_target_motion", "beta_shooter_motion", "base_hit_prob"):
            check(is_prob(getattr(self, name)), f"{name} must be in [0, 1]")

        # Bases.
        check(len(self.base_values) == B, "len(base_values) must equal bases_per_team")
        check(all(math.isfinite(v) for v in self.base_values), "base_values must be finite")
        check(isinstance(self.base_hp, int) and self.base_hp >= 1, "base_hp must be an int >= 1")
        check(min(self.base_min_mid, self.base_min_wall, self.base_min_sep) >= 0,
              "base_min_mid, base_min_wall and base_min_sep must be non-negative")
        check(self.base_min_wall <= half - self.base_min_mid,
              "base region is empty: need base_min_wall <= width/2 - base_min_mid")
        check(2 * self.base_min_wall <= self.height, "base region is empty: need 2*base_min_wall <= height")
        if self.base_layout is not None:
            check(len(self.base_layout) == B, "base_layout must list bases_per_team blue base positions")
            for x, y in self.base_layout:
                check(0 <= x < half and 0 <= y <= self.height,
                      f"base_layout position ({x}, {y}) must lie in blue's half of the arena")

        # Deployment.
        check(0 <= self.zone_mid_min <= self.zone_mid_max <= half,
              "deployment zone must satisfy 0 <= zone_mid_min <= zone_mid_max <= width/2")
        check(0 <= self.zone_wall_margin and 2 * self.zone_wall_margin <= self.height,
              "zone_wall_margin must be in [0, height/2]")
        check(self.bases_spawn_radius >= 0, "bases_spawn_radius must be non-negative")
        check(self.drone_min_sep >= 0, "drone_min_sep must be non-negative")

        # Rewards / episode.
        check(math.isfinite(self.drone_value), "drone_value must be finite")
        check(isinstance(self.max_steps, int) and self.max_steps >= 1, "max_steps must be an int >= 1")

        if errors:
            raise ValueError("Invalid Config:\n  - " + "\n  - ".join(errors))


# ---------------------------------------------------------------------------
# Override parsing for scripts:  key=value  (value: Python literal or a simple
# arithmetic expression in `pi`, e.g. turn_delta=pi/4), or a path to a JSON file.

_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.Pow: operator.pow}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_NAMES = {"pi": math.pi, "tau": 2 * math.pi, "inf": math.inf,
          "true": True, "false": False, "none": None, "null": None}


def _safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, bool, type(None))):
        return node.value
    if isinstance(node, ast.Name) and node.id.lower() in _NAMES:
        return _NAMES[node.id.lower()]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_safe_eval(e) for e in node.elts)
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def parse_value(raw: str) -> Any:
    """Parse a literal, a simple expression in `pi`, or fall back to a bare string."""
    raw = raw.strip()
    try:
        return _safe_eval(ast.parse(raw, mode="eval"))
    except (SyntaxError, ValueError):
        return raw


def parse_overrides(items: Iterable[str]) -> dict[str, Any]:
    """Turn ["key=value", "file.json", ...] into a dict of Config overrides."""
    valid = {f.name for f in dataclasses.fields(Config)}
    out: dict[str, Any] = {}
    for item in items:
        path = Path(item)
        if item.endswith(".json") and path.is_file():
            out.update(json.loads(path.read_text()))
            continue
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} must look like key=value")
        out[key.strip()] = parse_value(raw)
    unknown = sorted(set(out) - valid)
    if unknown:
        raise ValueError(f"unknown Config field(s): {unknown}")
    return out


def config_from_overrides(items: Iterable[str] | None, base: Config | None = None) -> Config:
    base = base or Config()
    overrides = parse_overrides(items or [])
    return dataclasses.replace(base, **overrides) if overrides else base
