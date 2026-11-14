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
    pairs = metagame.play_pairs(cfg, names, games=6, seeds=[0], workers=1)
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


def test_sweep_grid_is_anchored_to_the_original_arena():
    # The scale grid is relative to the original 30x20 arena, not to the current defaults,
    # so earlier sweep results reproduce.
    from conftest import SMALL_ARENA
    assert sweep.SCALED == SMALL_ARENA
    assert sweep.scaled_config(1.0, 5) == Config(**SMALL_ARENA, max_steps=300, base_hp=5)


def test_sweep_summary_row():
    cfg = sweep.scaled_config(1.0, 3)
    names = ["rusher", "hunter", "defender"]
    pairs = metagame.play_pairs(cfg, names, games=6, seeds=[0], workers=1)
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
    assert any(line.split()[:2] == ["1", "3"] for line in text.splitlines()) and "RU rusher" in text
    assert "Games: 6 games per pair, seed 0" in text and "Code: commit unknown" in text


# ---------------------------------------------------------------- stamps and pooled seeds
def test_run_metagame_stamps_its_output(tmp_path):
    cfg = Config(n_drones=1, bases_per_team=1, base_values=(1.0,))
    data, path = metagame.run_metagame(cfg, ["rusher", "hunter"], 3, [4, 5], 1, tmp_path, tag="t")
    saved = json.loads(path.read_text())
    assert path.name.startswith("metagame_") and path.name.endswith("_t.json")
    assert set(saved["git"]) == {"commit", "dirty"} and saved["git"]["commit"] != "unknown"
    assert isinstance(saved["git"]["dirty"], bool)
    assert saved["seeds"] == [4, 5] and saved["games"] == 3 and saved["policies"] == ["rusher", "hunter"]
    assert Config(**{k: tuple(map(tuple, v)) if k == "base_layout" and v else
                     tuple(v) if isinstance(v, list) else v for k, v in saved["config"].items()}) == cfg
    assert saved["created"] and "pairs" in saved
    report = metagame.format_report(metagame.analyse(saved, 5, 0), saved)
    assert "seeds 4, 5 pooled: 3 games per pair per seed, 6 per pair" in report
    assert f"commit {saved['git']['commit'][:12]}" in report


def test_unstamped_files_still_load():
    pairs = [{"blue": "rusher", "red": "hunter", "return": [1.0, -1.0, 0.5], "length": [10, 12, 9],
              "truncated": [False] * 3, "cause": ["elimination"] * 3, "winner": [0, 1, 0]}]
    old = {"created": "20261113-000000", "seed": 0, "games": 3, "policies": ["rusher", "hunter"], "pairs": pairs}
    report = metagame.format_report(metagame.analyse(old, 5, 0), old)
    assert "3 games per pair, seed 0" in report and "commit unknown, created 20261113-000000" in report
    bare = {"games": 3, "policies": ["rusher", "hunter"], "pairs": pairs}
    report = metagame.format_report(metagame.analyse(bare, 5, 0), bare)
    assert "seeds unknown" in report and "commit unknown, created unknown" in report


def test_pooled_seeds():
    cfg = Config()
    names = ["rusher", "hunter"]
    one = metagame.play_pairs(cfg, names, games=4, seeds=[0], workers=1)[0]
    two = metagame.play_pairs(cfg, names, games=4, seeds=[0, 1], workers=1)[0]
    assert all(len(two[f]) == 8 for f in metagame.FIELDS)
    assert two["seed"] == [0] * 4 + [1] * 4
    for f in metagame.FIELDS:
        assert two[f][:4] == one[f]                    # seed 0's games are unchanged by pooling
    other = metagame.play_pairs(cfg, names, games=4, seeds=[1], workers=1)[0]
    assert two["return"][4:] == other["return"] and two["length"][4:] == other["length"]
    assert two["length"][:4] != two["length"][4:]      # distinct game seeds per seed


# ---------------------------------------------------------------- field sweeps
def test_zip_and_grid_axes():
    axes = sweep.build_axes(["base_hp=3,5"], [["alpha_target_motion=0.5,0.35,0.25",
                                                "beta_shooter_motion=0.25,0.35,0.5"]])
    settings = sweep.field_settings(Config(), axes)
    assert len(settings) == 2 * 3
    params = [p for p, _ in settings]
    assert params[0] == {"base_hp": 3, "alpha_target_motion": 0.5, "beta_shooter_motion": 0.25}
    assert params[-1] == {"base_hp": 5, "alpha_target_motion": 0.25, "beta_shooter_motion": 0.5}
    for p, cfg in settings:
        assert all(getattr(cfg, k) == v for k, v in p.items())
        assert cfg.width == Config().width                       # everything else at the base config
    assert sweep.tag_for(params[0]) == "hp3_atm0.5_bsm0.25"
    one = sweep.field_settings(Config(), sweep.build_axes(["move_turn_mode=\"exclusive\""], None))
    assert len(one) == 1 and one[0][1].move_turn_mode == "exclusive"
    tup = sweep.build_axes(["base_values=(1,1),(1,2)"], None)
    assert [d["base_values"] for d in tup[0]] == [(1, 1), (1, 2)]


@pytest.mark.parametrize("grid, zips, message", [
    (["not_a_field=1,2"], None, "unknown Config field"),
    (None, [["alpha_target_motion=0.5,0.35", "beta_shooter_motion=0.25"]], "equal length"),
    (["base_hp=3"], [["base_hp=5", "n_drones=2"]], "more than once"),
    (["base_hp"], None, "FIELD=V1,V2"),
])
def test_bad_sweep_specs(grid, zips, message):
    with pytest.raises(ValueError, match=message):
        sweep.build_axes(grid, zips)


def test_scale_hp_preset_reproduces_the_old_grid():
    settings = sweep.preset_scale_hp([1.0, 1.33], [3, 5])
    assert [p for p, _ in settings] == [{"scale": 1.0, "base_hp": 3}, {"scale": 1.0, "base_hp": 5},
                                        {"scale": 1.33, "base_hp": 3}, {"scale": 1.33, "base_hp": 5}]
    assert all(cfg == sweep.scaled_config(p["scale"], p["base_hp"]) for p, cfg in settings)
    assert [sweep.tag_for(p) for p, _ in settings] == ["s1_hp3", "s1_hp5", "s1.33_hp3", "s1.33_hp5"]


def test_payoff_rows_table():
    A = np.array([[0.0, 0.5], [-0.5, 0.0]])
    row = {"params": {"alpha_target_motion": 0.5}, "names": ["raid", "raid_angle"], "A": A, "SE": np.full((2, 2), 0.1)}
    text = sweep.format_payoff_rows([row], ["raid_angle", "raid_strafe"])
    assert "α 0.5" in text and "-0.500 ± 0.100" in text
    assert any(line.startswith("raid_strafe") and "(not in this run)" in line for line in text.splitlines())
