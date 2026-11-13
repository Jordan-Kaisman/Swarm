"""scripts/metagame.py: the LP solver, the payoff matrix, and the saved-results round trip."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import metagame  # noqa: E402

from drone_battle.config import Config  # noqa: E402


def test_rock_paper_scissors_is_uniform():
    A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    x, v = metagame.solve_zero_sum(A)
    np.testing.assert_allclose(x, [1 / 3] * 3, atol=1e-9)
    assert v == pytest.approx(0.0, abs=1e-9)


def test_dominant_policy_is_pure():
    # Policy 1 beats both others; 0 and 2 are the rest of a cycle-free order.
    A = np.array([[0, -2, 1], [2, 0, 3], [-1, -3, 0]], dtype=float)
    x, v = metagame.solve_zero_sum(A)
    np.testing.assert_allclose(x, [0, 1, 0], atol=1e-9)
    assert v == pytest.approx(0.0, abs=1e-9)


def test_equilibrium_guarantees_the_value():
    rng = np.random.default_rng(0)
    for _ in range(50):
        M = rng.normal(size=(5, 5))
        A = M - M.T                                   # skew-symmetric: a symmetric zero-sum game
        x, v = metagame.solve_zero_sum(A)
        assert x.min() >= 0 and x.sum() == pytest.approx(1.0)
        assert v == pytest.approx(0.0, abs=1e-8)
        assert (A.T @ x).min() >= -1e-8                # x earns >= 0 against every pure reply


def test_payoff_matrix_is_skew_symmetric():
    cells = {(0, 1): np.array([1.0, 2.0, 3.0]), (0, 2): np.array([-1.0, -1.0]), (1, 2): np.array([0.5, 1.5])}
    A, SE = metagame.payoff_matrix(["a", "b", "c"], cells)
    np.testing.assert_allclose(A, -A.T)
    assert np.all(np.diag(A) == 0)
    assert A[0, 1] == 2.0 and A[2, 0] == 1.0 and A[1, 2] == 1.0
    assert SE[0, 1] == SE[1, 0] == pytest.approx(1.0 / np.sqrt(3)) and SE[0, 2] == 0.0


def test_saved_results_reproduce_the_tables(tmp_path):
    cfg = Config()
    names = ["rusher", "hunter", "raid"]
    pairs = metagame.play_pairs(cfg, names, games=6, seed=0, workers=1)
    assert [(p["blue"], p["red"]) for p in pairs] == [("rusher", "hunter"), ("rusher", "raid"), ("hunter", "raid")]
    assert all(len(p[f]) == 6 for p in pairs for f in metagame.FIELDS)
    data = {"created": "test", "seed": 0, "games": 6, "policies": names, "pairs": pairs}
    reloaded = json.loads(json.dumps(data))
    a, b = metagame.analyse(data, 20, 0), metagame.analyse(reloaded, 20, 0)
    np.testing.assert_array_equal(a["A"], b["A"])
    np.testing.assert_array_equal(a["mean_weight"], b["mean_weight"])
    assert metagame.format_report(a, data) == metagame.format_report(b, reloaded)


@pytest.mark.slow
def test_metagame_cli_save_and_load(tmp_path):
    run = [sys.executable, str(SCRIPTS / "metagame.py"), "--policies", "rusher", "hunter", "raid",
           "--games", "8", "--boot", "10", "--workers", "2", "--out", str(tmp_path)]
    played = subprocess.run(run, capture_output=True, text=True, check=True).stdout
    saved = list(tmp_path.glob("metagame_*.json"))
    assert len(saved) == 1
    loaded = subprocess.run([sys.executable, str(SCRIPTS / "metagame.py"), "--load", str(saved[0]), "--boot", "10"],
                            capture_output=True, text=True, check=True).stdout
    tables = lambda out: out[out.index("Meta-game:"):]   # noqa: E731
    assert tables(played) == tables(loaded)
    for heading in ("Payoff A[i, j]", "median length (IQR)", "Equilibrium mixture", "support freq", "worst case"):
        assert heading in played


# ---------------------------------------------------------------- sweep
import sweep  # noqa: E402

from drone_battle.env import reset  # noqa: E402


@pytest.mark.parametrize("scale", [1.0, 1.33, 1.67])
@pytest.mark.parametrize("base_hp", [3, 5, 8])
def test_scaled_configs_are_valid_and_reset(scale, base_hp):
    cfg, d = sweep.scaled_config(scale, base_hp), Config()
    for key, v in sweep.SCALED.items():
        assert getattr(cfg, key) == pytest.approx(v * scale)
    assert cfg.max_steps == round(300 * scale) and cfg.base_hp == base_hp
    drone_params = ("step_len", "turn_delta", "move_dirs", "weapon_range", "weapon_cone", "hit_c", "hit_pmax",
                    "alpha_target_motion", "beta_shooter_motion", "drone_min_sep", "n_drones")
    assert all(getattr(cfg, f) == getattr(d, f) for f in drone_params)
    s = reset(cfg, np.random.default_rng(0))
    assert np.all(s.pos[:, 0] <= cfg.width) and np.all(s.pos[:, 1] <= cfg.height)


def test_scale_one_hp_five_is_the_default_config():
    assert sweep.scaled_config(1.0, 5) == Config()


def test_sweep_summary_row():
    cfg = sweep.scaled_config(1.0, 3)
    names = ["rusher", "hunter", "defender"]
    pairs = metagame.play_pairs(cfg, names, games=6, seed=0, workers=1)
    data = {"created": "test", "elapsed_s": 1.0, "seed": 0, "games": 6, "policies": names,
            "config": {"width": cfg.width, "base_hp": cfg.base_hp}, "sweep": {"scale": 1.0, "base_hp": 3},
            "pairs": pairs}
    row = sweep.summarize_run(data, 10, 0)
    lengths = np.concatenate([p["length"] for p in pairs])
    assert row["median_length"] == np.median(lengths)
    trunc = [np.mean(p["truncated"]) for p in pairs]
    assert row["max_trunc"] == max(trunc)
    assert row["support_size"] == int(np.sum(row["x"] >= 0.01))
    if not all(t for p in pairs for t in p["truncated"]):
        assert sum(row["ko_shares"]) == pytest.approx(1.0)
    text = sweep.format_summary([row], 10)
    assert " 1.00   3" in text and "RU rusher" in text
