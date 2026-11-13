"""Spec §13 test 10: rusher vs rusher is fair (mirrored spawns, symmetric resolution)."""

import math

import numpy as np
import pytest

from drone_battle.config import Config
from drone_battle.match import run_match
from drone_battle.policies import rusher
from drone_battle.state import BLUE, RED

GAMES = 2000


@pytest.mark.slow
def test_rusher_vs_rusher_is_fair():
    results = run_match(Config(), rusher, rusher, games=GAMES, seed=0)
    wins = sum(r.winner == BLUE for r in results)
    losses = sum(r.winner == RED for r in results)
    decisive = wins + losses
    assert decisive > GAMES // 2
    p = wins / decisive
    assert abs(p - 0.5) <= 3 * math.sqrt(0.25 / decisive), (wins, losses)

    returns = np.array([r.blue_return for r in results])
    se = returns.std(ddof=1) / math.sqrt(GAMES)
    assert abs(returns.mean()) <= 3 * se
