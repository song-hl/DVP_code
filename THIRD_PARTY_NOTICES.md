# Third-Party Notices

This repository is a focused research-code release built on top of open-source
robot learning and robot asset repositories. The notes below document the
upstream projects used while preparing the released tasks.

## Upstream Projects

| Upstream project | License observed locally | Usage in this repository |
| --- | --- | --- |
| [RobotLab](https://github.com/fan-ziqi/robot_lab) | Apache-2.0 | Isaac Lab extension structure, task registration patterns, base locomotion environment organization. |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | BSD-3-Clause | Environment manager patterns, simulation configuration APIs, and RSL-RL integration conventions. |
| [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) | Upstream repository includes its own notices under `doc/licenses` | Unitree G1 robot/task implementation references and G1 velocity/mimic setup. |
| [booster_assets](https://github.com/BoosterRobotics/booster_assets) | BSD-3-Clause | Booster K1 robot assets and reference motion file layout. |
| [booster_train](https://github.com/BoosterRobotics/booster_train) | Apache-2.0 | Booster K1 training configuration and actuator implementation references. |
| [noetix_e1_lab](https://github.com/Noetix-Robotics/noetix_e1_lab) | BSD-3-Clause | Noetix E1 robot model and whole-body motion tracking task references. |

## Released Embodiments

- LimX Oli: retained for the main whole-body motion tracking experiments
  described in the paper. The release keeps only a sanitized BeyondMimic task
  configuration; private LimX mimic code and local processed motion data are not
  distributed.
- Unitree G1: retained for whole-body motion tracking and optional velocity
  tracking.
- Booster K1: retained for cross-platform whole-body motion tracking. K1 assets
  are resolved from `BOOSTER_ASSETS_DIR`, an installed `booster_assets` package,
  or a sibling `../booster_assets` checkout.
- Noetix E1: retained for cross-platform whole-body motion tracking.

## Distribution Notes

Third-party robot descriptions, meshes, USD files, and motion file paths are
included or referenced only for reproducing the paper experiments. Third-party
`*.npz` motion datasets are not distributed in this release; see `README.md`
for the upstream source locations and expected local paths. When redistributing
or modifying this repository, check the upstream licenses for each asset source
and preserve required notices.

Large motion/robot files may be stored through Git LFS. If a file is an LFS
pointer after cloning, run:

```bash
git lfs pull
```
