# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents, flat_env_cfg

_ENTRY_POINT = "isaaclab.envs:ManagerBasedRLEnv"

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-LimX-Oli-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{flat_env_cfg.__name__}:LimXOliBeyondMimicFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LimXOliBeyondMimicPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-LimX-Oli-Play-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{flat_env_cfg.__name__}:LimXOliBeyondMimicPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LimXOliBeyondMimicPPORunnerCfg",
    },
)
