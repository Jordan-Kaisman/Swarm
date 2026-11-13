# Drone Swarm Battle (AA228 Final Project)

## What this is
A two-team (blue vs red) drone swarm game that serves as the environment for an AA228 (Decision Making Under Uncertainty) final project. Formally it is a two-player zero-sum stochastic game, and each team is controlled by a central controller. If the opponent is fixed, it becomes an MDP; it becomes a POMDP once fog of war is added later.

**Current phase: build the environment only.** Solution methods (MCTS, multiagent rollout, policy gradient, POMDP solvers) come later. Don't implement them unless asked.

**`docs/ENV_SPEC.md` is the source of truth.** Read it before changing anything in `drone_battle/`. If code and spec disagree, or the spec is silent on a *game-design* question, ask. Don't invent mechanics. Implementation details are your call. Record every design change in the spec's Decision log.

## Conventions
- Python ≥ 3.11, with numpy, matplotlib and pytest. Ask before adding other dependencies.
- **Units:** lengths are measured in drone step lengths. Angles are in radians, CCW from +x, and are always wrapped to [−π, π). Use `geometry.wrap()` for every angle difference.
- **Randomness** comes only from an `np.random.Generator` passed in as an argument. Never use global RNG state or the `np.random.*` module functions.
- **`step()` is pure.** It never mutates its input state and returns a new `State`.
- **State is a struct of arrays,** never per-drone objects. Drone arrays have length 2n, with blue at indices `0..n-1` and red at `n..2n-1`. Base arrays are laid out the same way (blue first).
- **Every mechanic and constant lives in `Config`.** No magic numbers in the logic. A new mechanic gets a config switch whose default preserves the current behavior.
- **Per-drone parameters** (weapon range, weapon cone) are stored as length-2n arrays even while they are uniform.
- **Vectorize pairwise geometry.** No Python loops over drone pairs inside `step()`.
- **Rewards** are returned from blue's perspective. The game is zero-sum, so red's reward is the negative. The environment never discounts.
- **Policies** always act in the *canonical frame*: they see themselves as blue (rows `0..n-1`, left half). Red is handled through `mirror_state` / `mirror_action`.

## Commands (create these per the spec)
- `pytest -q`: fast tests only (~12 s); this is the default via `addopts` in pyproject.toml
- `pytest -q -m slow`: the statistical and long tests (duel rates, invariants, symmetry, outcome checks; ~2 min)
- `pytest -q -m ""`: everything
- `python scripts/run_match.py --blue rusher --red hunter --games 1000 --seed 0`
- `python scripts/run_match.py --blue rusher --red hunter --games 1 --render out.mp4`
- `python scripts/bench.py`

## Working rules
- Add tests together with each mechanic (spec §13). The mirror-equivariance and purity tests must always pass.
- Build in the order given in spec §1. Render something before moving on to the next phase.
