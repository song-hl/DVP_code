# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robot_lab.assets.limx import HU_D03_03
from robot_lab.tasks.manager_based.beyondmimic.tracking_env_cfg import BeyondMimicEnvCfg


LIMX_OLI_BODY_NAMES = [
    "base_link",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "waist_pitch_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_hand_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_hand_yaw_link",
]

LIMX_OLI_ACTION_SCALE = {
    ".*_hip_pitch.*": 0.25,
    ".*_hip_roll.*": 0.25,
    ".*_hip_yaw.*": 0.5,
    ".*_knee.*": 0.25,
    ".*_ankle.*": 0.5,
    "waist_.*": 0.5,
    "head_.*": 0.1,
    ".*_shoulder.*": 0.1,
    ".*_elbow.*": 0.1,
    ".*_wrist.*": 0.1,
    ".*_hand.*": 0.1,
}


@configclass
class LimXOliBeyondMimicFlatEnvCfg(BeyondMimicEnvCfg):
    motion_file = f"{os.path.dirname(__file__)}/motion/limx_oli_motion.npz"

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = HU_D03_03.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = LIMX_OLI_ACTION_SCALE
        self.commands.motion.motion_file = self.motion_file
        self.commands.motion.anchor_body_name = "waist_pitch_link"
        self.commands.motion.body_names = LIMX_OLI_BODY_NAMES

        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None

        self.events.randomize_com_positions.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base_link")
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[
                r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                r"(?!left_hand_yaw_link$)(?!right_hand_yaw_link$).+$"
            ],
        )
        self.terminations.ee_body_pos.params["body_names"] = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_hand_yaw_link",
            "right_hand_yaw_link",
        ]
        self.episode_length_s = 30.0


@configclass
class LimXOliBeyondMimicPlayEnvCfg(LimXOliBeyondMimicFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.play = True
        self.events.randomize_push_robot = None
