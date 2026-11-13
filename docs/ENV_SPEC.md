# Environment Specification: Drone Swarm Battle (v0)

**Status:** v0 is fully observable, with one central controller per team, auto-fire, and identical drones. Defaults marked *(tune)* are starting points that will be tuned through balance runs.

---

## 1. Overview and build order

- Two teams, **blue** (team 0) and **red** (team 1), with n drones each.
- The arena is a rectangle. Blue owns the left half and red the right half.
- Each team has bases in its own half. Each base has a **value** (the reward for destroying it) and **hit points** (how many hits it takes).
- Drones have 1 HP. They can shoot enemy drones (probabilistic hits) and enemy bases (deterministic hits).
- Time is discrete. In each step, every alive drone chooses an action (move, turn), and all drones act **simultaneously**. Then all drones **auto-fire simultaneously**.
- The game is zero-sum. The episode **terminates** when a team has no alive drones or no alive bases. A step cap ends the episode as **truncation** and is not a game outcome.

**Build order** (render after each step before continuing):
1. `Config`, `State`, `reset`, `render`: look at about 20 random layouts.
2. Movement only, with the random policy, saved as an animation.
3. Geometry, hit model and firing, checked with the duel test (§13).
4. Bases, rewards and termination.
5. Mirror transform and `observe`, checked with the equivariance test.
6. Scripted policies, the match runner, and the benchmark.

**Repo layout:**
```
drone_battle/
  config.py     # frozen Config dataclass + validate()
  state.py      # State dataclass, copy(), team/index helpers
  geometry.py   # wrap, pairwise distance/bearing, in_cone, hit probability
  actions.py    # direction table, encoding, legal_actions(cfg)
  mirror.py     # mirror_state, mirror_action
  env.py        # reset, step (phases as private functions), observe
  render.py     # render(state, cfg, info), animate(trajectory, cfg, path)
  policies.py   # random, rusher, hunter, defender
scripts/run_match.py, scripts/bench.py
tests/
docs/ENV_SPEC.md
```

---

## 2. Units and geometry

- **Length unit:** the drone step length (`step_len = 1`).
- **Positions** are floats (x, y) in [0, W] × [0, H]. The midline is x = W/2. Blue's half is x < W/2.
- **Angles** are in radians, measured CCW from +x, and wrapped to [−π, π): `wrap(a) = (a + π) mod 2π − π`.
- **Distance** is Euclidean.
- **Line of fire (LOF)** from shooter i to target j is the vector `p_j − p_i`, using post-move positions.
- **In-cone test:** `|wrap(atan2(LOF_y, LOF_x) − ψ_i)| ≤ θ_w,i / 2 + EPS_ANGLE`, with `EPS_ANGLE = 1e-9`. A target exactly on the cone edge counts as in. If `d < EPS_DIST = 1e-9`, the target counts as in the cone.
- **In-range test:** `d ≤ R_w,i`.

*Why continuous positions:* a grid distorts exactly the mechanics under study. It makes ranges square or diamond-shaped, makes diagonal moves faster, creates ties on cone edges, and hides the effect of θ at short range.

---

## 3. Config (defaults)

| Group | Name | Default | Meaning |
|---|---|---|---|
| Arena | `width`, `height` | 30.0, 20.0 *(tune)* | Arena size, in step lengths. Crossing takes about 30 steps. |
| Teams | `n_drones` | 3 | Drones per team. |
| Dynamics | `step_len` | 1.0 | Length of every move (the same in all directions). |
| | `turn_delta` | π/4 | Turn increment per step. Must be ≤ min weapon cone (validate). |
| | `move_dirs` | 8 | 8 or 4 world-frame directions. |
| | `move_turn_mode` | `"simultaneous"` | `"simultaneous"`: move and turn in the same step (27 actions). `"exclusive"`: move *or* turn (11 actions). |
| Weapon | `weapon_range` | 4.0 *(tune)* | R_w (per-drone array, uniform in v0). |
| | `weapon_cone` | π/2 | θ_w, the full cone angle (per-drone array). |
| | `hit_c` | 1.0 *(tune)* | Scale constant in the hit model. |
| | `hit_pmax` | 0.9 | Maximum hit probability. |
| | `alpha_target_motion` | 0.5 *(tune)* | Accuracy penalty against a crossing target. |
| | `beta_shooter_motion` | 0.25 *(tune)* | Accuracy penalty when the shooter is moving. |
| | `fire_mode` | `"auto"` | Only `"auto"` exists in v0. Firing-as-action is reserved for later. |
| | `target_priority` | `"drones_first"` | Targeting rule (§6.5). |
| Sensing | `sense_range`, `sense_cone` | 8.0, 2π | **Unused in v0.** Reserved for fog of war. |
| Bases | `bases_per_team` | 2 | B. |
| | `base_values` | [1.0, 1.0] | Blue base values. Red's are mirrored. |
| | `base_hp` | 5 *(tune)* | Hits needed to destroy a base. |
| | `base_hit_prob` | 1.0 | Per-shot hit probability against a base in cone and range. |
| | `base_min_mid` | 5.0 | Minimum distance from a base to the midline. |
| | `base_min_wall` | 2.0 | Minimum distance from a base to the outer walls. |
| | `base_min_sep` | 4.0 | Minimum distance between same-team bases. |
| | `base_layout` | `None` | `None` = random. Otherwise an explicit list of blue base positions. |
| Init | `placement` | `"zone"` | `"zone"` or `"bases"`. |
| | `zone_mid_min`, `zone_mid_max` | 3.0, 8.0 *(tune)* | The deployment band's distance from the midline (the curriculum knob). |
| | `zone_wall_margin` | 2.0 | Minimum spawn distance from the top and bottom walls. |
| | `bases_spawn_radius` | 2.0 | Spawn radius around a base for `"bases"` placement. |
| | `drone_min_sep` | 1.5 | Minimum spacing between same-team drones at spawn. |
| | `mirror_init` | `True` | Red's spawn is the mirror image of blue's. |
| | `random_headings` | `True` | Otherwise drones start facing the enemy half. |
| Rewards | `drone_value` | 0.1 | Value of one drone: reward per enemy drone killed, minus the same per own drone lost. Low on purpose (§16). |
| | `elimination_bonus` | `True` | §6.7. |
| Episode | `max_steps` | 300 | Truncation cap, several times a typical game length. |

`Config.validate()` checks, at minimum, that:
- `len(base_values) == bases_per_team`;
- `turn_delta ≤ min(weapon_cone)` (if not, some bearings can never be brought into the cone);
- the deployment zone and the base constraints fit inside the team's half;
- every probability is in [0, 1].

---

## 4. State

| Field | Shape / type | Notes |
|---|---|---|
| `pos` | (2n, 2) float64 | Blue rows first. |
| `heading` | (2n,) float64 | Wrapped. |
| `alive` | (2n,) bool | |
| `weapon_range`, `weapon_cone` | (2n,) float64 | Per drone, copied from Config at reset. |
| `base_pos` | (2B, 2) float64 | Blue bases first. |
| `base_hp` | (2B,) int | A base is alive iff hp > 0. |
| `base_value` | (2B,) float64 | |
| `t` | int | Bookkeeping only. **Not part of the game state:** policies must not use it. |
| `terminated` | bool | Calling `step` on a terminated state raises. |

- `State.copy()` deep-copies every array.
- The team of drone i is `i // n` (0 = blue). The team of base b is `b // B`.

The positions, headings, alive flags and base HP are a Markov state for v0. Velocities are not needed, because hit probabilities depend only on the moves made in the *current* step.

---

## 5. Actions

Each drone's action is a pair **(move, turn)**:
- `move ∈ {0, …, move_dirs}`. `0` = hover. `k ≥ 1` = a world-frame direction at angle `(k−1)·2π/move_dirs`, so 1 = +x (east) and, with 8 directions, 3 = +y (north). **Movement is independent of heading** (quadcopter), and every move has length `step_len`.
- `turn ∈ {0, 1, 2}` = {−Δ (clockwise), 0, +Δ (counterclockwise)}.
- Each team's action is an int array of shape (n, 2). The input to `step` has shape (2n, 2), blue rows first.
- `actions.encode/decode` provides a flat index `a = move·3 + turn`.
- In `"exclusive"` mode, the legal actions are `(0, t)` for every t, plus `(k, 1)` for k ≥ 1, giving 11 in total. `legal_actions(cfg)` returns the list. `step` raises on an illegal action.
- Actions of dead drones are ignored.

---

## 6. Step

`step(cfg, state, actions, rng) -> (new_state, reward_blue: float, terminated: bool, truncated: bool, info: dict)`

Every phase computes from the arrays produced by the previous phase. Never apply partial updates within a phase.

**6.1 Validate.** Check shapes, value ranges and mode legality. Mask out dead drones.

**6.2 Move and turn.**
- `heading ← wrap(heading + (turn − 1)·turn_delta)`.
- `pos ← clip(pos + step_len·dir(move), [0, W] × [0, H])`.
- `disp ← new_pos − old_pos` (the *actual* displacement, after clipping). `moved_i ← ‖disp_i‖ > 1e-9`.

**6.3 Geometry.** Using post-move positions and post-turn headings, compute:
- the pairwise distances `d_ij`;
- the LOF unit vectors;
- the in-cone and in-range masks for drone → enemy drone and drone → enemy base.

Only alive shooters and alive enemy targets count.

**6.4 Hit probability.**

Against an enemy drone j:
```
P_ij = min(pmax, c / (θ_w,i · max(d_ij, EPS_DIST)))
       · (1 − α · |sin φ_j| · moved_j)
       · (1 − β · moved_i)
```
Here `|sin φ_j| = |cross(disp_j/‖disp_j‖, LOF_ij/‖LOF_ij‖)|`, and it is 0 if j did not move. `P_ij = 0` if j is out of range or out of the cone.

Against an enemy base: `P = base_hit_prob` if the base is in range and in the cone, otherwise 0.

*Why this model:* `c/(θ·d)` is the expected outcome of firing pellets with angular spread θ, so no projectile simulation is needed. The α term means a target crossing the line of fire is harder to hit, while a target moving straight toward or away from the shooter doesn't dodge anything. The β term means a hovering shooter is a stable gun platform but also a sitting duck.

**6.5 Targeting (auto-fire).** Each alive drone with at least one valid target fires **exactly one** shot.
- With `drones_first`: if any enemy drone has P > 0, pick the one with the highest P. Otherwise pick a base with P > 0.
- Ties go to the smaller distance, then to the lower index within the target team. Tie-breaking is deterministic and uses no RNG.

*Why auto-fire:* firing costs nothing in v0, so there's never a reason to hold fire, and making it an action would only add branching. Firing-as-action only becomes worthwhile once firing has a cost (ammo, cooldown, revealing position), which is a planned extension. Keep the fire phase separate so that `fire_mode` can be extended.

**6.6 Resolve.**
- Draw `u = rng.random(2n)` **every step, exactly 2n draws**, whether or not anyone fires.
- Drone i's shot hits if `u[i] < P_i,target(i)`.
- Apply all outcomes at once. A drone that is hit gets `alive = False`. A drone killed this step still fires this step, so mutual kills are possible. Each base loses `hp` equal to its number of hits, and is **destroyed** this step if hp drops from > 0 to ≤ 0.

*Why fixed draws:* they keep the RNG streams aligned across different policies (common random numbers) and make the mirror test exact. Implement the logic as `_step_with_uniforms(cfg, state, actions, u)` so tests can inject `u`.

*Why simultaneous resolution:* resolving shots in sequence would create a hidden turn-order advantage.

**6.7 Reward** (blue's perspective, for this step only):
```
r = Σ value(red bases destroyed this step) − Σ value(blue bases destroyed this step)
  + drone_value · (red drones killed − blue drones killed)
```
Summed over an episode (undiscounted), the drone term equals `drone_value · (blue survivors − red survivors)`, a terminal value per surviving drone.
**Elimination bonus:** if, after resolution, *exactly one* team has no alive drones, the surviving team also receives the total value of the eliminated team's bases that are still alive. The reasoning is that those bases would fall anyway, so elimination and base destruction are valued consistently. If both teams are eliminated in the same step, there is no bonus.

**6.8 Termination.**
- `terminated` = some team has 0 alive drones, or some team has 0 alive bases.
- Then `t ← t + 1`, and `truncated = (not terminated) and t ≥ max_steps`.

Truncation is a simulator safeguard, not a draw rule. The fraction of truncated games is the stalemate detector.

**6.9 Info.** Contains:
- `shots`: a list of (shooter, target_kind, target_idx, P, hit);
- `drones_killed`, `bases_hit`, `bases_destroyed`, `disp`, `moved`.

It is used for rendering and statistics.

---

## 7. Reset

`reset(cfg, rng) -> State`

**Bases.**
- If `base_layout` is given, use it for blue.
- Otherwise rejection-sample B blue positions uniformly subject to:
  - `x ≤ W/2 − base_min_mid` and `x ≥ base_min_wall`;
  - `y ∈ [base_min_wall, H − base_min_wall]`;
  - pairwise separation `≥ base_min_sep`.

  Raise an error after 10,000 failed tries.
- Red bases are the mirror images of blue's (§8), with the same values and `hp = base_hp`.

**Drones.**
- `"zone"`: blue spawns uniformly in `x ∈ [W/2 − zone_mid_max, W/2 − zone_mid_min]`, `y ∈ [zone_wall_margin, H − zone_wall_margin]`, with pairwise separation `≥ drone_min_sep` (rejection sampling).
- `"bases"`: each drone spawns uniformly within `bases_spawn_radius` of one of its team's bases (assigned round-robin), clipped to the arena, with the same separation rule.
- Headings are uniform on [−π, π) if `random_headings`. Otherwise blue starts at 0 and red at π.
- If `mirror_init`, red drone i is the mirror image of blue drone i. Otherwise red is sampled independently by the same rules in its own half.

*Why:* the zone's distance from the midline controls whether rushing works, which makes it a curriculum knob. Mirrored spawns make every episode fair.

---

## 8. Mirror transform (canonicalization)

**`mirror_state(s, cfg)`:**
- x → W − x, and ψ → wrap(π − ψ).
- Swap the team blocks (blue rows ↔ red rows, blue bases ↔ red bases), keeping the order within each team.
- Mirror the per-drone arrays the same way.

**`mirror_action(a, cfg)`:**
- Swap the team blocks.
- Moves: hover stays hover. Direction index `j = k−1` maps to `j' = (move_dirs/2 − j) mod move_dirs`.
- Turns: `0 ↔ 2`, because a reflection reverses the sense of rotation.

**Reward:** negated under the mirror.

**`observe(state, team, cfg)`** in v0 returns `state` for blue and `mirror_state(state, cfg)` for red. Policies always see themselves as blue and output (n, 2) actions in that frame. The runner maps red's actions back with `mirror_action`. `observe` is the *only* path from state to policy input, which is where fog of war will plug in later.

**Equivariance requirement:** for any s, a and u, if `_step_with_uniforms(cfg, s, a, u)` returns `s′` with reward r, then
`_step_with_uniforms(cfg, mirror_state(s, cfg), mirror_action(a, cfg), swap_blocks(u, n))` returns `mirror_state(s′, cfg)` with reward −r. Also, `mirror_state(mirror_state(s, cfg), cfg) == s`.

---

## 9. Observations

v0 is fully observable via `observe` (§8).

Reserved for v1 (do not implement now):
- sensing range and cone (`sense_range > weapon_range`);
- home-turf cameras;
- binned observations (range band × bearing sector) instead of exact positions;
- particle-filter beliefs.

---

## 10. Rendering

`render(state, cfg, info=None, ax=None)` draws:
- the arena with the midline, and the two halves lightly tinted blue and red;
- **bases** as squares in team color, with opacity scaled by HP and labeled `hp/value`; destroyed bases as a grey ×;
- **drones** as triangles in team color pointing along their heading, each with a translucent weapon-cone wedge (radius R_w, angle θ_w); dead drones as a small grey ×;
- **shots** (from `info`) as lines from shooter to target, solid for hits and dashed for misses;
- the step counter and the alive counts.

`animate(trajectory, cfg, path)` writes an mp4 via ffmpeg, falling back to a gif.

---

## 11. Scripted policies

Signature: `policy(obs, cfg, rng) -> (n, 2) int array`, in the canonical frame. These serve as baselines, opponents, and later as rollout policies, so keep them simple and fast.

- **random:** uniform over legal actions.
- **rusher:** moves toward the nearest alive enemy base and turns toward it.
- **hunter:** moves toward the nearest alive enemy drone and turns toward it.
- **defender:** stays within a radius `r_def` (default 4) of its nearest own base, turns toward the nearest enemy drone, and moves to intercept enemies that enter its own half.
- **raid** (`r_group` = 2.5, `engage_margin` = 2): the whole team attacks one enemy base together.
  - Target: the alive enemy base nearest the centroid of the team's alive drones, recomputed every step.
  - Cohesion: let D_i be the distance from drone i to the target and D_max the maximum over alive teammates. Drone i advances toward the target iff `D_i ≥ D_max − r_group`; otherwise it hovers. The pack advances at the pace of its rearmost drone.
  - Facing: toward the nearest enemy drone if it is within `weapon_range + engage_margin`; otherwise toward the target.
  - In weapon range of the target with no enemy drone within `weapon_range + engage_margin`: hover.
- **raid_strafe:** as raid, but when an enemy drone is within `weapon_range + engage_margin`, an advancing drone moves in the legal direction closest to perpendicular to its line of fire to the nearest enemy drone, on the side with the larger component toward the target (this exploits α). On an exact tie between the sides, it takes the counter-clockwise one; with the enemy drone at distance 0 (no line of fire), it moves as raid.
- **split** (`k` = 1): drones with canonical index < k run defender; the rest run raid as a pack of their own (the centroid and D_max are over the raiders only). Roles are fixed by index.

In `exclusive` mode, every scripted policy turns when its aim point is within weapon range + one step and outside its cone, and moves otherwise.

Helpers:
- *Turn toward X:* pick the turn that minimizes `|wrap(bearing_X − ψ)|` after turning.
- *Move toward X:* pick the direction with the largest projection onto the direction of X. Hover if X is closer than `step_len/2`.

---

## 12. Match runner

`scripts/run_match.py`:
- Arguments: `--blue`, `--red`, `--games`, `--seed`, `--render PATH` (renders the first game), `--hist PATH` (saves a histogram of episode lengths) and `--config` (overrides).

**Returns.** The environment is undiscounted. A game's return is the plain sum of its per-step rewards (§6.7), from blue's perspective; red's return is the negative.

**Knockout outcome** (from the final state). A team is *knocked out* when it has no alive drones or no alive bases.
- Exactly one team knocked out: the other team wins, by *elimination* (the loser has no drones) or *bases destroyed* (the loser still has drones but no bases).
- Both teams knocked out in the same step: a draw, split into *mutual elimination* (both teams lost all their drones) and *simultaneous knockout* (any other same-step double knockout, e.g. both teams' last bases fall together).
- Truncated games are reported separately, not as draws (§6.8).

**Return-based outcome:** the sign of blue's return: *blue ahead* (> 0), *even* (= 0) or *red ahead* (< 0). |return| ≤ 1e-9 counts as even, to absorb float rounding. The two outcomes agree when win ↔ blue ahead, loss ↔ red ahead, and draw or truncated ↔ even.

A knockout winner keeps at least one base and has collected the loser's whole base value (destroyed, or through the elimination bonus). It also keeps at least one drone, while the loser has at most n. So with `elimination_bonus = True` and `drone_value ≥ 0`, a knockout winner's return is strictly positive whenever `drone_value · (n − 1) < min(base_values)`, which holds at the defaults. Draws can have either sign: a mutual elimination is decided by the bases destroyed earlier, and a simultaneous knockout returns `drone_value · (blue survivors − red survivors)`. Truncated games can have either sign too.

Report:
- the knockout outcome: wins, losses and draws by cause, and truncated games;
- the return-based outcome, overall and as a cross-tab against the knockout categories, with the number of disagreements;
- episode length: mean ± std, median and interquartile range, and the truncation fraction;
- blue's mean return ± standard error, decomposed into its base-value term (bases destroyed), elimination-bonus term and drone-value term (`drone_value` × survivor difference), each ± SE;
- drone kills and base kills per team;
- shots fired and hit rate.

**Meta-game (`scripts/metagame.py`, needs scipy).**
- Policies (default): rusher, hunter, defender, raid, raid_strafe, split. Default config, seed 0, 2000 games per unordered pair i < j with i as blue (15 pairs). Every pair uses the same game seeds.
- Payoff matrix: `A[i, j]` = blue's mean return, `A[j, i] = −A[i, j]`, diagonal 0. Exact mirror symmetry makes the reversed order and the mirror pairs redundant, so they are not simulated.
- Equilibrium: the symmetric zero-sum matrix game is solved by linear programming (maximize v subject to `Aᵀx ≥ v`, x a probability vector). Its value is 0 because A is skew-symmetric. When several equilibria exist, the LP returns one of them.
- Bootstrap: 200 resamples of games within each cell. Each resample recomputes A and the equilibrium. The report gives each policy's support frequency (weight > 0.01) and its mean weight.
- Per-game results (return, length, truncated, cause, winner) are saved to `out/metagame_<timestamp>.json`. `--load PATH` recomputes the tables from that file without re-running the games.
- Report: A ± SE; the truncation fraction and median length (IQR) per pair; the equilibrium mixture; the bootstrap support frequencies and mean weights; each policy's worst-case payoff (the row minimum of A, which includes the 0 against itself).

**Sweep (`scripts/sweep.py`).** Runs the meta-game over a grid of configs and prints one summary table.
- Arena scale s (default 1.0, 1.33, 1.67): width = 30s, height = 20s, base_min_mid = 5s, base_min_wall = 2s, base_min_sep = 4s, zone_mid_min = 3s, zone_mid_max = 8s, zone_wall_margin = 2s, max_steps = round(300s). Drone parameters (step length, turning, weapon range and cone, hit model) are unchanged.
- `base_hp` (default 3, 5, 8). The 9 configs use the same 6 policies, seed 0 and 1000 games per pair. Each keeps its own `out/metagame_<timestamp>_s<s>_hp<hp>.json`; `--load` rebuilds the summary from saved runs.
- Summary row per config: the equilibrium mixture (weights ≥ 0.01) and its support size; the bootstrap support frequency per policy; the median game length over all non-mirror pairs; the largest truncation fraction over pairs and its pair; the share of non-truncated games ending by base knockout, elimination or double knockout; the runtime.

---

## 13. Tests (pytest)

1. **wrap:** ±π, multiples of 2π, negative inputs.
2. **In-cone:** on-axis; just inside and just outside θ/2; exactly on the edge (in); directly behind; d ≈ 0.
3. **Hit model:** the formula at several distances; the pmax cap; zero beyond range and outside the cone; a crossing target vs. a radially moving target (α); a moving shooter (β).
4. **Movement:** a diagonal step has length 1; clipping at walls, with `disp` reflecting the clip; dead drones don't move; in exclusive mode, illegal actions raise.
5. **Duel:** two drones facing each other at distance d, both hovering, over 20k trials. The per-step kill rate should match P(d) within 3 SE, and mutual kills should occur at about P².
6. **Base:** one hovering drone with the base in range and `base_hit_prob = 1`. The base should be destroyed on exactly step `base_hp`, with reward = +value on that step.
7. **Rewards and termination:** the elimination bonus; mutual elimination (no bonus); termination when all bases are destroyed; truncation (`truncated=True`, `terminated=False`); `step` on a terminated state raises.
8. **Invariants** over 1,000 random-policy episodes: `alive` never flips from False to True; positions stay in bounds; HP never increases; the sum of rewards equals the sum implied by the events in `info`.
9. **Mirror equivariance** with injected `u`, on random states and actions; `mirror ∘ mirror = id`; the action mapping round-trips.
10. **Symmetry:** rusher vs rusher over 2,000 seeds gives a blue win rate ≈ 0.5 (within 3 SE).
11. **Purity:** `step` does not mutate its input.
12. **Determinism:** the same seed produces an identical trajectory.

---

## 14. Performance

`scripts/bench.py` measures steps/sec with random actions for n ∈ {1, 2, 4, 8}.

Target: **≤ ~100 μs per step at n = 4** in numpy. Planners will call `step` 10⁴–10⁵ times per decision. Don't micro-optimize yet, but avoid Python loops over pairs. numba may come later behind a flag.

---

## 15. Out of scope for v0 (hooks only)

- Fog of war, home cameras, binned observations (reserved: `sense_range`, `sense_cone`, `observe`).
- Firing as an action with a cost (ammo, cooldown, revealing position) (reserved: `fire_mode`).
- Heterogeneous drones with different θ_w and R_w (the per-drone arrays already exist; requires `turn_delta ≤ min θ_w`).
- Limited communication or decentralized execution.
- Bases that shoot back.
- A Gym/PettingZoo wrapper (the parallel API fits simultaneous moves; easy to add later because the core API is functional).
- All planning and learning algorithms.

---

## 16. Decision log

- **2026-11-13:**
  - Continuous positions and headings; discrete actions; coarse time step.
  - Quadcopter dynamics: world-frame movement in any direction, independent of heading; turning is ±Δ per step and may happen together with a move.
  - Drones have 1 HP, with probabilistic hits that depend on distance, cone width and motion. Bases have HP and are hit deterministically.
  - Auto-fire, with the targeting rule drones before bases.
  - Simultaneous resolution of moves and shots.
  - Zero-sum rewards from base values, plus the elimination bonus; drone kills are worth 0 by default.
  - The time cap counts as truncation, not as a game outcome.
  - Mirrored bases and spawns, with a deployment zone placed back from the midline.
  - (v1) Sensing range will be larger than weapon range.
  - `kill_reward` renamed to `drone_value`; default 0 → 0.1. drone_value = 0.1 (low). Drones are cheap and attritable: a full team (n=3) is worth less than one base. It stands in for the value surviving drones keep after the episode (limited supply). In an undiscounted zero-sum game, a per-kill reward equals a terminal reward of drone_value per surviving drone.

## 17. Open questions (settle through balance runs)

- `base_hp` relative to the defenders' response time, which sets the rush vs. turtle balance. Watch game length and the truncation fraction.
- The deployment zone's distance from the midline (curriculum).
- The sizes of α and β.
- `n_drones` and the arena size.
- Whether mutual elimination should stay a draw.
- `drone_value` relative to base value, as a later sweep axis. It only matters once policies optimize reward; the scripted policies ignore it.
