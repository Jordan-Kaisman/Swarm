"""Scripted policies (spec §11) and the match runner (spec §12)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import DENSE, SMALL_ARENA, hover, one_v_one, scenario
from drone_battle.actions import legal_table
from drone_battle.config import Config
from drone_battle.env import _step_with_uniforms, observe, reset
from drone_battle.match import joint_action
from drone_battle.geometry import PI
from drone_battle.match import (BASES, BLUE_AHEAD, ELIMINATION, EVEN, KNOCKOUT_ROWS, LOSS,
                                MUTUAL_ELIMINATION, RED_AHEAD, SIMULTANEOUS, TRUNCATION, WIN,
                                classify_outcome, knockout_bucket, return_outcome, run_match, summarize)
from drone_battle.mirror import mirror_action, mirror_state, swap_blocks
from drone_battle.policies import POLICIES, _raid, defender, get_policy, move_toward, raid, split, turn_toward
from drone_battle.render import length_histogram
from drone_battle.state import BLUE, RED, make_state, states_equal
from test_mirror import random_state

MODES = [Config(n_drones=3, **DENSE), Config(n_drones=3, move_turn_mode="exclusive", **DENSE),
         Config(n_drones=3, move_dirs=4, **DENSE)]


@pytest.mark.parametrize("cfg", MODES, ids=["simultaneous", "exclusive", "4dirs"])
@pytest.mark.parametrize("name", sorted(POLICIES))
def test_policies_return_legal_actions(cfg, name):
    rng = np.random.default_rng(0)
    legal = legal_table(cfg)
    for _ in range(200):
        s = random_state(cfg, rng)
        for team in (BLUE, RED):
            a = POLICIES[name](observe(s, team, cfg), cfg, rng)
            assert a.shape == (cfg.n_drones, 2) and a.dtype.kind == "i"
            assert legal[a[:, 0], a[:, 1]].all()


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_policies_ignore_t(name):
    cfg = Config(n_drones=3, **DENSE)
    s = random_state(cfg, np.random.default_rng(1))
    a = POLICIES[name](s, cfg, np.random.default_rng(5))
    s.t = 123
    np.testing.assert_array_equal(a, POLICIES[name](s, cfg, np.random.default_rng(5)))


def test_random_policy_covers_legal_actions():
    cfg = Config(n_drones=2, move_turn_mode="exclusive")
    s = reset(cfg, np.random.default_rng(0))
    rng = np.random.default_rng(0)
    seen = {tuple(x) for _ in range(500) for x in POLICIES["random"](s, cfg, rng).tolist()}
    assert len(seen) == 11


def test_move_toward():
    cfg = Config()
    pos = np.array([[10.0, 10.0]] * 5)
    targets = np.array([[15, 10], [10, 15], [7, 7], [10.3, 10.2], [12, 11.5]], dtype=float)
    np.testing.assert_array_equal(move_toward(pos, targets, cfg), [1, 3, 6, 0, 2])


def test_turn_toward():
    cfg = Config()
    heading = np.array([0.0, 0.0, 0.0, 0.0, PI - 0.5, PI - 0.1])
    bearing = np.array([1.0, -1.0, 0.3, 0.0, -PI + 0.5, -PI + 0.1])
    # CCW, CW, none (0.3 < delta/2), none, CCW across the seam, none (0.2 < delta/2 across the seam)
    np.testing.assert_array_equal(turn_toward(heading, bearing, cfg), [2, 0, 1, 1, 2, 1])
    # A target directly behind: either turn is optimal; both are a valid answer.
    assert turn_toward(np.array([0.0]), np.array([-PI]), cfg)[0] in (0, 2)


def test_rusher_and_hunter_close_in():
    cfg = Config(n_drones=3)
    for name, target_of in (("rusher", lambda s: s.base_pos[2:][s.base_hp[2:] > 0]),
                            ("hunter", lambda s: s.pos[3:][s.alive[3:]])):
        s = reset(cfg, np.random.default_rng(4))
        a = POLICIES[name](s, cfg, np.random.default_rng(0))
        acts = np.concatenate([a, np.tile([0, 1], (3, 1))])
        s1, *_ = _step_with_uniforms(cfg, s, acts, np.full(6, 0.999))
        for i in range(3):
            t = target_of(s)
            d0 = np.min(np.hypot(*(t - s.pos[i]).T))
            d1 = np.min(np.hypot(*(t - s1.pos[i]).T))
            assert d1 < d0


def test_defender_holds_its_disc():
    cfg = Config(n_drones=1, bases_per_team=1, base_values=(1.0,), base_layout=((5.0, 10.0),))
    # Far from its base and no intruders: head back toward the base.
    s = scenario(cfg, [[12, 10], [25, 10]], [0.0, PI])
    a = POLICIES["defender"](s, cfg, None)
    assert a[0, 0] == 5  # west
    # Inside the disc and no intruders: hover, aim at the nearest enemy.
    s = scenario(cfg, [[7, 10], [25, 12]], [PI / 2, PI])
    a = POLICIES["defender"](s, cfg, None)
    assert a[0, 0] == 0 and a[0, 1] == 0  # hover, turn clockwise toward the enemy
    # An intruder in the own half: go to the point of the disc nearest to it
    # (the move-toward helper hovers within step_len/2 of that point).
    s = scenario(cfg, [[8.8, 10], [13, 10]], [0.0, PI])
    assert POLICIES["defender"](s, cfg, None)[0, 0] == 0   # goal (9, 10) is 0.2 away: hover
    s = scenario(cfg, [[6, 10], [13, 10]], [0.0, PI])
    assert POLICIES["defender"](s, cfg, None)[0, 0] == 1   # east, toward the intruder
    assert get_policy("defender:r_def=1.2")(s, cfg, None)[0, 0] == 0  # goal (6.2, 10): hover


# ------------------------------------------------------- raid, raid_strafe, split
# Blue bases at (4, 4) and (4, 16); red's are mirrored at (26, 4) and (26, 16).
RAID_LAYOUT = ((4.0, 4.0), (4.0, 16.0))
FAR_RED = [[29.0, 1.0], [29.5, 2.5], [28.0, 0.5]]   # parked in a corner, facing the east wall


@pytest.mark.parametrize("move_dirs", [4, 8])
@pytest.mark.parametrize("mode", ["simultaneous", "exclusive"])
def test_unopposed_raid_keeps_the_pack_together(move_dirs, mode):
    cfg = Config(**SMALL_ARENA, base_layout=RAID_LAYOUT, move_dirs=move_dirs, move_turn_mode=mode)
    n, r_group = cfg.n_drones, 2.5
    # Centroid (6, 10.7) is nearest red's base at (26, 16). Distances to it: 21.2, 17.7, 23.3,
    # so the pack starts 5.6 deep and drone 1, the front one, must wait.
    s = scenario(cfg, [[6, 9], [9, 11], [3, 12]] + FAR_RED, [0.0] * n + [0.0] * n)
    target = np.array([26.0, 16.0])
    first = raid(s, cfg, None)
    assert first[1, 0] == 0 and first[0, 0] != 0 and first[2, 0] != 0
    spreads = []
    for _ in range(60):
        a = np.concatenate([raid(s, cfg, None), hover(cfg)[n:]])
        s, _, term, _, info = _step_with_uniforms(cfg, s, a, np.full(2 * n, 0.999))  # bases always hit
        assert s.alive.all()                    # unopposed: nobody shoots at the pack
        hits = info["bases_hit"]
        if hits.any():                          # arrived: the approach is over
            assert hits[3] > 0 and hits[:3].sum() == 0   # only the target, red's base at (26, 16)
            break
        D = np.hypot(*(target - s.pos[:n]).T)
        spreads.append(D.max() - D.min())
    else:
        pytest.fail("the raid never reached its target")
    formed = next(i for i, sp in enumerate(spreads) if sp <= r_group)
    assert max(spreads[formed:]) <= r_group + cfg.step_len


def test_raid_strafe_moves_across_the_line_of_fire():
    cfg = one_v_one()                                           # red base at (28, 10)
    # Blue at (20, 10); a red drone 5 away at (24, 13) is a threat (5 <= 4 + 2).
    s = scenario(cfg, [[20, 10], [24, 13]], [0.0, PI])
    a_raid, a_strafe = raid(s, cfg, None), POLICIES["raid_strafe"](s, cfg, None)
    assert a_raid[0, 0] == 1                     # raid: straight east, toward the base
    # Line of fire (0.8, 0.6); the normal with a component toward the base is (0.6, -0.8),
    # 53 degrees below east: the nearest legal direction is south-east.
    assert a_strafe[0, 0] == 8
    assert a_raid[0, 1] == a_strafe[0, 1] == 2   # both face the threat (CCW)
    # No threat: the two policies agree.
    s = scenario(cfg, [[12, 10], [24, 18]], [0.0, PI])
    np.testing.assert_array_equal(raid(s, cfg, None), POLICIES["raid_strafe"](s, cfg, None))


def test_split_assigns_roles_by_index():
    cfg = Config(**SMALL_ARENA, base_layout=RAID_LAYOUT)
    n = cfg.n_drones
    # Drone 0 sits by its base; drones 1 and 2 are forward with a small spread. If drone 0 were
    # part of the pack, it would be the rearmost and the other two would wait for it.
    s = scenario(cfg, [[3, 6], [12, 9], [12.5, 11]] + FAR_RED, [0.0] * 2 * n)
    a = split(s, cfg, None)
    np.testing.assert_array_equal(a[0], defender(s, cfg, None)[0])
    np.testing.assert_array_equal(a[1:], _raid(s, cfg, np.array([False, True, True]), 2.5, 2.0, False)[1:])
    assert (a[1:, 0] != 0).all()
    assert (raid(s, cfg, None)[1:, 0] == 0).all()       # the whole-team raid does wait
    # Same positions, roles follow the index: with k = 2, drone 1 defends and drone 2 raids alone.
    a2 = get_policy("split:k=2")(s, cfg, None)
    np.testing.assert_array_equal(a2[:2], defender(s, cfg, None)[:2])
    assert a2[2, 0] != 0
    # On random states too: rows < k defend, rows >= k raid as their own pack.
    rng = np.random.default_rng(0)
    for _ in range(100):
        st = random_state(cfg, rng)
        for k in (0, 1, 2, 3):
            act = split(st, cfg, None, k=k)
            np.testing.assert_array_equal(act[:k], defender(st, cfg, None)[:k])
            np.testing.assert_array_equal(act[k:], _raid(st, cfg, np.arange(n) >= k, 2.5, 2.0, False)[k:])


@pytest.mark.parametrize("name", ["raid", "raid_strafe", "split"])
def test_new_policies_play_the_same_from_either_side(name):
    """Game A: `name` as blue against an opponent. Game B starts from the mirrored state with
    the roles swapped and the mirrored uniforms. B must be the mirror image of A at every step:
    the policy acts on its canonical observation only, whichever side it plays."""
    cfg = Config(mirror_init=False)                   # asymmetric starts
    n = cfg.n_drones
    P = POLICIES[name]
    rng = np.random.default_rng(0)
    steps = 0
    for opp in ("rusher", "hunter", "defender", "raid", "raid_strafe", "split"):
        Q = POLICIES[opp]
        for _ in range(2):
            sA = reset(cfg, rng)
            sB = mirror_state(sA, cfg)
            for _ in range(cfg.max_steps):
                aA = joint_action(cfg, sA, P, Q, None, None)
                aB = joint_action(cfg, sB, Q, P, None, None)
                np.testing.assert_array_equal(aB, mirror_action(aA, cfg))
                u = rng.random(2 * n)
                sA, rA, teA, trA, _ = _step_with_uniforms(cfg, sA, aA, u)
                sB, rB, teB, trB, _ = _step_with_uniforms(cfg, sB, aB, swap_blocks(u, n))
                assert states_equal(mirror_state(sA, cfg), sB, atol=1e-9)
                assert rB == pytest.approx(-rA, abs=1e-12) and (teA, trA) == (teB, trB)
                steps += 1
                if teA or trA:
                    break
    assert steps > 100


def test_get_policy():
    assert get_policy("rusher") is POLICIES["rusher"]
    with pytest.raises(ValueError, match="unknown policy"):
        get_policy("camper")


# ------------------------------------------------------------------ runner
def _final(cfg, alive, base_hp):
    s = make_state(cfg, np.zeros((cfg.n_total, 2)), np.zeros(cfg.n_total),
                   np.zeros((2 * cfg.bases_per_team, 2)), alive=alive, base_hp=base_hp)
    s.terminated = True
    return s


def test_classify_outcome():
    cfg = Config(n_drones=2, bases_per_team=2)
    assert classify_outcome(cfg, _final(cfg, [1, 1, 0, 0], [5, 5, 5, 5]), False) == (BLUE, ELIMINATION)
    assert classify_outcome(cfg, _final(cfg, [0, 0, 1, 0], [5, 5, 5, 5]), False) == (RED, ELIMINATION)
    assert classify_outcome(cfg, _final(cfg, [1, 0, 1, 1], [5, 1, 0, 0]), False) == (BLUE, BASES)
    assert classify_outcome(cfg, _final(cfg, [1, 0, 1, 1], [0, 0, 3, 0]), False) == (RED, BASES)
    assert classify_outcome(cfg, _final(cfg, [0, 0, 0, 0], [5, 5, 5, 5]), False) == (None, MUTUAL_ELIMINATION)
    assert classify_outcome(cfg, _final(cfg, [1, 0, 1, 0], [0, 0, 0, 0]), False) == (None, SIMULTANEOUS)
    assert classify_outcome(cfg, _final(cfg, [0, 0, 1, 0], [5, 0, 0, 0]), False) == (None, SIMULTANEOUS)
    s = _final(cfg, [1, 1, 1, 1], [5, 5, 5, 5])
    s.terminated = False
    assert classify_outcome(cfg, s, True) == (None, TRUNCATION)


def winner_return_is_positive(cfg) -> bool:
    """The assumption under which a knockout winner always ends strictly ahead (spec §12).

    The winner keeps >= 1 base and has collected the loser's whole base value (destroyed, or
    through the elimination bonus), so its base-value term is >= min(base_values). It keeps
    >= 1 drone and the loser has at most n, so its drone term is >= -drone_value * (n - 1).
    """
    return (cfg.elimination_bonus and cfg.drone_value >= 0
            and cfg.drone_value * (cfg.n_drones - 1) < min(cfg.base_values))


def test_play_episode_bookkeeping():
    cfg = Config(n_drones=3, **DENSE)
    assert winner_return_is_positive(cfg)
    results = run_match(cfg, POLICIES["hunter"], POLICIES["rusher"], games=50, seed=3)
    for r in results:
        assert r.terminated != r.truncated
        assert np.all(r.hits_drone <= r.shots_drone) and np.all(r.hits_base <= r.shots_base)
        assert np.all(r.drone_kills <= cfg.n_drones) and np.all(r.base_kills <= cfg.bases_per_team)
        assert np.all(r.drone_kills <= r.hits_drone)
        if r.winner == BLUE:
            assert r.blue_return > 0
        elif r.winner == RED:
            assert r.blue_return < 0
    stats = summarize(results)
    assert sum(stats["counts"].values()) == 50
    assert stats["truncated_frac"] == pytest.approx(np.mean([r.truncated for r in results]))


# ------------------------------------------------------- return-based outcome
def test_return_outcome_sign_and_tolerance():
    assert return_outcome(1.0) == BLUE_AHEAD and return_outcome(-0.5) == RED_AHEAD
    assert return_outcome(0.0) == EVEN
    assert return_outcome(0.1 + 0.2 - 0.3) == EVEN   # 5.6e-17, float rounding, not a lead


@pytest.mark.parametrize("drone_value", [0.0, 0.1])
def test_simultaneous_knockout_by_elimination_and_last_base(drone_value):
    """Blue kills red's last drone while red destroys blue's last base, in the same step.
    Both teams are knocked out (a draw). Blue loses its base (-1) but gets the elimination
    bonus for red's surviving base (+1), so the base-value term is 0 and only the drone term
    is left: drone_value * (1 blue survivor - 0 red survivors). With drone_value > 0 the
    return-based outcome (blue ahead) disagrees with the knockout outcome (draw)."""
    cfg = one_v_one(drone_value=drone_value)
    # Blue's base is at (2, 10) with 1 HP left. The red drone is 2 east of it, facing it; the blue
    # drone is 3 east of the red drone (behind it, outside red's cone), facing it. u = 0: all hit.
    s = scenario(cfg, [[7.0, 10.0], [4.0, 10.0]], [PI, PI], base_hp=[1, cfg.base_hp])
    s1, r, te, tr, info = _step_with_uniforms(cfg, s, hover(cfg), np.zeros(2))
    assert te and not tr
    assert info["drones_killed"].tolist() == [False, True]
    assert info["bases_destroyed"].tolist() == [True, False]
    assert classify_outcome(cfg, s1, tr) == (None, SIMULTANEOUS)
    assert r == pytest.approx(drone_value, abs=1e-12)
    assert return_outcome(r) == (EVEN if drone_value == 0 else BLUE_AHEAD)


@pytest.mark.slow
def test_knockout_winner_has_positive_return_under_default_config():
    """A knockout winner ends strictly ahead whenever drone_value * (n - 1) < min(base_values)
    (with the elimination bonus on and drone_value >= 0); see winner_return_is_positive."""
    cfg = Config()
    assert winner_return_is_positive(cfg)
    names = ("rusher", "hunter", "defender")
    results = [res for i, b in enumerate(names) for j, r in enumerate(names)
               for res in run_match(cfg, POLICIES[b], POLICIES[r], games=40, seed=10 * i + j)]
    for res in results:
        bucket = knockout_bucket(res)
        if bucket == WIN:
            assert res.blue_return > 0
        elif bucket == LOSS:
            assert -res.blue_return > 0     # red, the winner, ends strictly ahead
    stats = summarize(results)
    winner_rows = [row for row in KNOCKOUT_ROWS if row[0] in (WIN, LOSS)]
    for row in winner_rows:
        assert stats["disagreements"][row] == 0, row
        assert sum(stats["crosstab"][row].values()) > 0, row   # the sample exercises every row
    assert sum(stats["return_outcomes"].values()) == len(results)


def test_summarize_length_quantiles():
    cfg = Config(n_drones=3, **DENSE)
    results = run_match(cfg, POLICIES["rusher"], POLICIES["hunter"], games=30, seed=1)
    stats = summarize(results)
    lengths = np.array([r.length for r in results])
    assert stats["length_median"] == np.median(lengths)
    assert stats["length_q1"] <= stats["length_median"] <= stats["length_q3"]
    assert stats["length_q1"] == np.percentile(lengths, 25) and stats["length_q3"] == np.percentile(lengths, 75)


def test_length_histogram_writes_png(tmp_path):
    path = length_histogram([1, 3, 3, 7, 300, 300], ["win", "loss", "draw", "win", "truncated", "truncated"],
                            tmp_path / "h.png", title="test")
    assert path.exists() and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_run_match_cli(tmp_path):
    root = Path(__file__).resolve().parents[1]
    hist = tmp_path / "lengths.png"
    out = subprocess.run([sys.executable, str(root / "scripts" / "run_match.py"), "--blue", "rusher",
                          "--red", "hunter", "--games", "20", "--seed", "0", "--hist", str(hist),
                          "--config", "n_drones=2"],
                         capture_output=True, text=True, check=True, cwd=tmp_path).stdout
    for heading in ("Knockout outcome", "Return-based outcome", "Knockout vs return", "disagreements:",
                    "median", "IQR"):
        assert heading in out
    assert hist.exists()
