# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents, flat_env_cfg

_ENTRY_POINT = "isaaclab.envs:ManagerBasedRLEnv"

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Rough-Booster-K1-Fight-001-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{flat_env_cfg.__name__}:BoosterK1Fight001EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BoosterK1Fight001PPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-Booster-K1-Fight-001-Play-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{flat_env_cfg.__name__}:BoosterK1Fight001PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BoosterK1Fight001PPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Rough-Booster-K1-MJ-Dance-002-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{flat_env_cfg.__name__}:BoosterK1MJDance002EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BoosterK1MJDance002PPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-Booster-K1-MJ-Dance-002-Play-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{flat_env_cfg.__name__}:BoosterK1MJDance002PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BoosterK1MJDance002PPORunnerCfg",
    },
)
