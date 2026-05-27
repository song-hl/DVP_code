# DVP Robot Lab

This repository contains the Isaac Lab code for the DVP paper experiments on
deployable proprioceptive representation learning for humanoid whole-body
control.

## Included

SRL methods:

- DVP
- SPR
- VAE
- SimSiam
- PPO baseline

Tasks:

- LimX Oli BeyondMimic
- Unitree G1 BeyondMimic
- Booster K1 BeyondMimic
- Noetix E1 BeyondMimic
- Unitree G1 velocity tracking

## Installation

Use an Isaac Lab Python environment. The experiments were developed with Isaac
Sim 5.1.0, Isaac Lab 2.3.2, Python 3.11, and a CUDA-enabled PyTorch setup.

```bash
python -m pip install -e source/robot_lab
git lfs pull
```

List registered environments:

```bash
python scripts/tools/list_envs.py
```

## Motion Data

Motion datasets are not included in this repository. Place compatible motion
files at the paths expected by the task configs before training.

| Task | Source | Expected path |
| --- | --- | --- |
| LimX Oli | private converted BeyondMimic motion | `source/robot_lab/robot_lab/tasks/manager_based/beyondmimic/config/limx_oli/motion/limx_oli_motion.npz` |
| Unitree G1 Dance 102 | `unitree_rl_lab` | `source/robot_lab/robot_lab/tasks/manager_based/beyondmimic/config/g1/motion/G1_Take_102.bvh_60hz.npz` |
| Unitree G1 Gangnam | `unitree_rl_lab` | `source/robot_lab/robot_lab/tasks/manager_based/beyondmimic/config/g1/motion/G1_gangnam_style_V01.bvh_60hz.npz` |
| Noetix E1 | `noetix_e1_lab` | `source/robot_lab/robot_lab/tasks/manager_based/beyondmimic/config/noetix_e1/motion/dance1.npz` |

Booster K1 reads motions from `booster_assets` instead of this repository:

```bash
export BOOSTER_ASSETS_DIR=/path/to/booster_assets
```

Expected files under that directory:

```text
motions/K1/k1_fight_001.npz
motions/K1/k1_mj2_seg1.npz
```

## Training

Use the RSL-RL entry point:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task <TASK_NAME> \
  --srl_algo_name <METHOD> \
  --headless
```

Methods:

```text
ppo
ppo_dvp
ppo_spr
ppo_vae
ppo_simsiam
```

Task IDs:

```text
RobotLab-Isaac-BeyondMimic-Flat-LimX-Oli-v0
RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-Dance-102-v0
RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-Gangnam-Style-v0
RobotLab-Isaac-BeyondMimic-Rough-Booster-K1-Fight-001-v0
RobotLab-Isaac-BeyondMimic-Rough-Booster-K1-MJ-Dance-002-v0
RobotLab-Isaac-BeyondMimic-Flat-Noetix-E1-v0
RobotLab-Isaac-Velocity-Rough-Unitree-G1-v0
```

## Evaluation

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task <TASK_NAME> \
  --srl_algo_name <METHOD> \
  --checkpoint <CHECKPOINT_PATH>
```

## Third-Party Sources

See `THIRD_PARTY_NOTICES.md` for upstream repositories and license notes. This
release uses RobotLab/Isaac Lab extension patterns, Unitree G1 references from
`unitree_rl_lab`, Booster K1 references from `booster_assets` and
`booster_train`, and Noetix E1 references from `noetix_e1_lab`.

## Citation

If you use this code, cite the corresponding DVP paper. The BibTeX entry will be
added after publication.

## License

The repository code is released under Apache-2.0 unless a file states otherwise.
Third-party assets and adapted components remain subject to their upstream
licenses.
