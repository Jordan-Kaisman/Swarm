"""Drone Swarm Battle environment (v0). See docs/ENV_SPEC.md."""

from .config import Config, config_from_overrides
from .env import hit_probability_matrix, observe, reset, step
from .mirror import mirror_action, mirror_state, mirror_team_action
from .state import BLUE, RED, State, make_state

__all__ = [
    "Config", "config_from_overrides",
    "State", "make_state", "BLUE", "RED",
    "reset", "step", "observe", "hit_probability_matrix",
    "mirror_state", "mirror_action", "mirror_team_action",
]
