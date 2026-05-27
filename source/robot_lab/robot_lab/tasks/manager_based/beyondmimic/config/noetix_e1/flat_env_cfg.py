# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robot_lab.assets.noetix import NOETIX_E1_24DOF_ACTION_SCALE, NOETIX_E1_24DOF_CFG
from robot_lab.tasks.manager_based.beyondmimic.tracking_env_cfg import BeyondMimicEnvCfg


NOETIX_E1_BODY_NAMES = [
    "base_link",
    "waist_roll_link",
    "l_leg_hip_pitch_link",
    "r_leg_hip_pitch_link",
    "l_leg_knee_link",
    "r_leg_knee_link",
    "l_arm_shoulder_roll_link",
    "r_arm_shoulder_roll_link",
    "l_leg_ankle_roll_link",
    "r_leg_ankle_roll_link",
    "l_arm_elbow_pitch_link",
    "r_arm_elbow_pitch_link",
    "l_arm_elbow_yaw_link",
    "r_arm_elbow_yaw_link",
]


@configclass
class NoetixE1BeyondMimicFlatEnvCfg(BeyondMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = NOETIX_E1_24DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = NOETIX_E1_24DOF_ACTION_SCALE
        self.commands.motion.motion_file = f"{os.path.dirname(__file__)}/motion/dance1.npz"
        self.commands.motion.anchor_body_name = "waist_roll_link"
        self.commands.motion.body_names = NOETIX_E1_BODY_NAMES
        self.commands.motion.pose_range["z"] = (0.01, 0.03)

        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None

        self.events.randomize_com_positions.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base_link")
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[
                r"^(?!l_leg_ankle_roll_link$)(?!r_leg_ankle_roll_link$)"
                r"(?!l_arm_elbow_yaw_link$)(?!r_arm_elbow_yaw_link$).+$"
            ],
        )
        self.terminations.ee_body_pos.params["body_names"] = [
            "l_leg_ankle_roll_link",
            "r_leg_ankle_roll_link",
            "l_arm_elbow_yaw_link",
            "r_arm_elbow_yaw_link",
        ]
        self.episode_length_s = 30.0


@configclass
class NoetixE1BeyondMimicPlayEnvCfg(NoetixE1BeyondMimicFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.play = True
        self.events.randomize_push_robot = None
