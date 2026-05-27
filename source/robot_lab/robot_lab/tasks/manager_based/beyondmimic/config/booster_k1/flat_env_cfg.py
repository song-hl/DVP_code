# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.terrains as terrain_gen
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from robot_lab.assets.booster import BOOSTER_K1_ACTION_SCALE, BOOSTER_K1_CFG, booster_assets_path
from robot_lab.tasks.manager_based.beyondmimic.tracking_env_cfg import BeyondMimicEnvCfg


def _motion_file(name: str) -> str:
    return booster_assets_path(f"motions/K1/{name}")


BOOSTER_K1_BODY_NAMES = [
    "Trunk",
    "Head_2",
    "Left_Hip_Roll",
    "Left_Shank",
    "left_foot_link",
    "Right_Hip_Roll",
    "Right_Shank",
    "right_foot_link",
    "Left_Arm_2",
    "Left_Arm_3",
    "left_hand_link",
    "Right_Arm_2",
    "Right_Arm_3",
    "right_hand_link",
]


@configclass
class BoosterK1BeyondMimicFlatEnvCfg(BeyondMimicEnvCfg):
    motion_name = "k1_mj2_seg1.npz"

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = BOOSTER_K1_ACTION_SCALE
        self.commands.motion.motion_file = _motion_file(self.motion_name)
        self.commands.motion.anchor_body_name = "Trunk"
        self.commands.motion.body_names = BOOSTER_K1_BODY_NAMES

        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
        self.events.randomize_com_positions.params["asset_cfg"] = SceneEntityCfg("robot", body_names="Trunk")
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[r"^(?!left_hand_link$)(?!right_hand_link$)(?!left_foot_link$)(?!right_foot_link$).+$"],
        )
        self.terminations.ee_body_pos.params["body_names"] = [
            "left_hand_link",
            "right_hand_link",
            "left_foot_link",
            "right_foot_link",
        ]
        self.episode_length_s = 10.0


@configclass
class BoosterK1BeyondMimicRoughEnvCfg(BoosterK1BeyondMimicFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.debug_vis = False
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(10.0, 10.0),
            border_width=20.0,
            num_rows=5,
            num_cols=10,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=False,
            sub_terrains={
                "nearly_flat": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.8,
                    noise_range=(0.0, 0.005),
                    noise_step=0.005,
                    border_width=0.25,
                ),
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.2,
                    noise_range=(-0.015, 0.015),
                    noise_step=0.005,
                    border_width=0.25,
                ),
            },
        )


@configclass
class BoosterK1BeyondMimicPlayEnvCfg(BoosterK1BeyondMimicFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.play = True
        self.events.randomize_push_robot = None


@configclass
class BoosterK1Fight001EnvCfg(BoosterK1BeyondMimicRoughEnvCfg):
    motion_name = "k1_fight_001.npz"


@configclass
class BoosterK1MJDance002EnvCfg(BoosterK1BeyondMimicRoughEnvCfg):
    motion_name = "k1_mj2_seg1.npz"


@configclass
class BoosterK1Fight001PlayEnvCfg(BoosterK1BeyondMimicPlayEnvCfg):
    motion_name = "k1_fight_001.npz"


@configclass
class BoosterK1MJDance002PlayEnvCfg(BoosterK1BeyondMimicPlayEnvCfg):
    motion_name = "k1_mj2_seg1.npz"
