# Booster K1 BeyondMimic Tasks

This repository reuses the shared `robot_lab.tasks.manager_based.beyondmimic` MDP implementation for Booster K1 motion tracking tasks. Booster-specific code is limited to the K1 asset configuration, actuator model, motion/body mapping, and task registrations.

## Prerequisites

Copy or mount the official `booster_assets` directory next to `robot_lab`, or set `BOOSTER_ASSETS_DIR` to its absolute path. Installing the `booster_assets` Python package is not required for training.

- `robots/K1/K1_22dof.urdf`
- `motions/K1/k1_fight_001.npz`
- `motions/K1/k1_mj2_seg1.npz`

Recommended layout:

```text
DVP/
  robot_lab/
  booster_assets/
```

If the assets live elsewhere, set the environment variable manually:

```bash
export BOOSTER_ASSETS_DIR=/path/to/booster_assets
```

## Registered Tasks

- `RobotLab-Isaac-BeyondMimic-Rough-Booster-K1-Fight-001-v0`
- `RobotLab-Isaac-BeyondMimic-Rough-Booster-K1-MJ-Dance-002-v0`

Play variants use flat terrain and reset the reference motion from frame 0:

- `RobotLab-Isaac-BeyondMimic-Flat-Booster-K1-Fight-001-Play-v0`
- `RobotLab-Isaac-BeyondMimic-Flat-Booster-K1-MJ-Dance-002-Play-v0`

## Training

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-BeyondMimic-Rough-Booster-K1-MJ-Dance-002-v0 \
  --headless --num_envs 4096
```

For a quick smoke test:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-BeyondMimic-Rough-Booster-K1-MJ-Dance-002-v0 \
  --headless --num_envs 1 --max_iterations 1
```
