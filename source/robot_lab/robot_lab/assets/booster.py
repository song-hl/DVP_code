# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import booster_actuator as actuator


def _resolve_booster_assets_dir() -> str | None:
    env_dir = os.environ.get("BOOSTER_ASSETS_DIR")
    if env_dir:
        return env_dir

    try:
        from booster_assets import BOOSTER_ASSETS_DIR as package_assets_dir

        return package_assets_dir
    except ImportError:
        pass

    repo_sibling = Path(__file__).resolve().parents[5] / "booster_assets"
    if repo_sibling.is_dir():
        return str(repo_sibling)

    return None


BOOSTER_ASSETS_DIR = _resolve_booster_assets_dir()


def booster_assets_path(path: str) -> str:
    if BOOSTER_ASSETS_DIR is None:
        return f"${{BOOSTER_ASSETS_DIR}}/{path}"
    return f"{BOOSTER_ASSETS_DIR}/{path}"


BOOSTER_K1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        asset_path=booster_assets_path("robots/K1/K1_22dof.urdf"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            "Left_Shoulder_Roll": -1.3,
            "Right_Shoulder_Roll": 1.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": actuator.BoosterDelayedPDActuatorCfg(
            min_delay=2,
            max_delay=8,
            joint_names_expr=[".*_Hip_Pitch", ".*_Hip_Roll", ".*_Hip_Yaw", ".*_Knee_Pitch"],
            booster_joint_cfgs={
                ".*_Hip_Pitch": actuator.BoosterJointE6408(natural_freq=4.0, damping_ratio=1.5),
                ".*_Hip_Roll": actuator.BoosterJointE4315(natural_freq=4.0, damping_ratio=1.5),
                ".*_Hip_Yaw": actuator.BoosterJointE4310(natural_freq=4.0, damping_ratio=1.5),
                ".*_Knee_Pitch": actuator.BoosterJointE6416(natural_freq=4.0, damping_ratio=1.0),
            },
        ),
        "feet": actuator.BoosterDelayedPDActuatorCfg(
            min_delay=2,
            max_delay=8,
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            booster_joint_cfgs={
                ".*_Ankle_Pitch": actuator.BoosterK1AnkleParaWrapperCfg(
                    base_joint_cfg=actuator.BoosterJointE4310(),
                    serial_index=0,
                    natural_freq=4.0,
                    damping_ratio=1.5,
                ),
                ".*_Ankle_Roll": actuator.BoosterK1AnkleParaWrapperCfg(
                    base_joint_cfg=actuator.BoosterJointE4310(),
                    serial_index=1,
                    natural_freq=4.0,
                    damping_ratio=1.5,
                ),
            },
        ),
        "arms": actuator.BoosterDelayedPDActuatorCfg(
            min_delay=2,
            max_delay=8,
            joint_names_expr=[".*_Shoulder_Pitch", ".*_Shoulder_Roll", ".*_Elbow_Pitch", ".*_Elbow_Yaw"],
            booster_joint_cfgs=actuator.BoosterJointR14(),
        ),
        "head": actuator.BoosterDelayedPDActuatorCfg(
            min_delay=2,
            max_delay=8,
            joint_names_expr=[".*Head.*"],
            booster_joint_cfgs=actuator.BoosterJointHT4438(),
        ),
    },
)

BOOSTER_K1_ACTION_SCALE = {}
for _actuator_cfg in BOOSTER_K1_CFG.actuators.values():
    _effort_limits = _actuator_cfg.effort_limit_sim
    _stiffness = _actuator_cfg.stiffness
    if not isinstance(_effort_limits, dict):
        _effort_limits = {name: _effort_limits for name in _actuator_cfg.joint_names_expr}
    if not isinstance(_stiffness, dict):
        _stiffness = {name: _stiffness for name in _actuator_cfg.joint_names_expr}
    for _name in _actuator_cfg.joint_names_expr:
        if _name in _effort_limits and _name in _stiffness and _stiffness[_name]:
            BOOSTER_K1_ACTION_SCALE[_name] = 0.25 * _effort_limits[_name] / _stiffness[_name]
