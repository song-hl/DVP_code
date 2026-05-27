# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import MISSING

import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class BoosterDelayedPDActuator(DelayedPDActuator):
    """Delayed PD actuator with Booster-style speed-dependent torque clipping."""

    cfg: "BoosterDelayedPDActuatorCfg"

    def __init__(self, cfg: "BoosterDelayedPDActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.knee_point_velocity = self._parse_joint_parameter(cfg.knee_point_velocity, self.velocity_limit)
        self.knee_point_velocity = torch.minimum(torch.clamp(self.knee_point_velocity, min=0.0), self.velocity_limit)
        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._denom = (self.velocity_limit - self.knee_point_velocity).clamp(min=1e-6)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        joint_vel_abs = self._joint_vel.abs()
        tau_linear = self.effort_limit * (self.velocity_limit - joint_vel_abs) / self._denom
        max_effort = tau_linear.clamp(min=0.0).clamp(max=self.effort_limit)
        max_effort = torch.where(torch.isfinite(self.velocity_limit), max_effort, self.effort_limit)
        max_effort = torch.where(self.velocity_limit <= 0.0, torch.zeros_like(max_effort), max_effort)
        return torch.clip(effort, min=-max_effort, max=max_effort)


@configclass
class BoosterJointCfg:
    joint_model_name: str = MISSING
    effort_limit: float = MISSING
    velocity_limit: float = MISSING
    knee_point_velocity: float = MISSING
    armature: float = MISSING
    stiffness: float | None = None
    damping: float | None = None
    natural_freq: float = 10.0
    damping_ratio: float = 2.0

    def __post_init__(self):
        if self.stiffness is None:
            self.stiffness = self.armature * (2 * torch.pi * self.natural_freq) ** 2
        if self.damping is None:
            self.damping = 2 * self.damping_ratio * self.armature * (2 * torch.pi * self.natural_freq)


@configclass
class BoosterDelayedPDActuatorCfg(DelayedPDActuatorCfg):
    class_type: type = BoosterDelayedPDActuator
    knee_point_velocity: dict[str, float] | float | None = None
    stiffness: dict[str, float] | float | None = None
    damping: dict[str, float] | float | None = None
    booster_joint_cfgs: dict[str, BoosterJointCfg] | BoosterJointCfg | None = None

    def __post_init__(self):
        if isinstance(self.booster_joint_cfgs, BoosterJointCfg):
            joint_cfg = self.booster_joint_cfgs
            self.effort_limit_sim = joint_cfg.effort_limit
            self.velocity_limit_sim = joint_cfg.velocity_limit
            self.knee_point_velocity = joint_cfg.knee_point_velocity
            self.armature = joint_cfg.armature
            self.stiffness = joint_cfg.stiffness if self.stiffness is None else self.stiffness
            self.damping = joint_cfg.damping if self.damping is None else self.damping
        elif isinstance(self.booster_joint_cfgs, dict):
            self.effort_limit_sim = {name: cfg.effort_limit for name, cfg in self.booster_joint_cfgs.items()}
            self.velocity_limit_sim = {name: cfg.velocity_limit for name, cfg in self.booster_joint_cfgs.items()}
            self.knee_point_velocity = {name: cfg.knee_point_velocity for name, cfg in self.booster_joint_cfgs.items()}
            self.armature = {name: cfg.armature for name, cfg in self.booster_joint_cfgs.items()}
            if self.stiffness is None:
                self.stiffness = {name: cfg.stiffness for name, cfg in self.booster_joint_cfgs.items()}
            if self.damping is None:
                self.damping = {name: cfg.damping for name, cfg in self.booster_joint_cfgs.items()}


@configclass
class ParallelJointWrapperCfg(BoosterJointCfg):
    joint_model_name: str = "ParallelJointWrapper"
    effort_ratio: tuple[float, float] = MISSING
    velocity_ratio: tuple[float, float] = MISSING
    armature_ratio: tuple[float, float] = MISSING
    knee_point_velocity_ratio: tuple[float, float] = (1.0, 1.0)
    base_joint_cfg: BoosterJointCfg = MISSING
    serial_index: int = MISSING

    def __post_init__(self):
        self.effort_limit = self.effort_ratio[self.serial_index] * self.base_joint_cfg.effort_limit
        self.velocity_limit = self.velocity_ratio[self.serial_index] * self.base_joint_cfg.velocity_limit
        self.knee_point_velocity = (
            self.knee_point_velocity_ratio[self.serial_index] * self.base_joint_cfg.knee_point_velocity
        )
        self.armature = self.armature_ratio[self.serial_index] * self.base_joint_cfg.armature
        self.joint_model_name = f"{self.joint_model_name}({self.base_joint_cfg.joint_model_name})[{self.serial_index}]"
        super().__post_init__()


@configclass
class BoosterK1AnkleParaWrapperCfg(ParallelJointWrapperCfg):
    joint_model_name: str = "BoosterK1AnkleParaWrapper"
    effort_ratio: tuple[float, float] = (1.0, 1.0)
    velocity_ratio: tuple[float, float] = (1.0, 1.0)
    armature_ratio: tuple[float, float] = (2.0, 2.0)


@configclass
class BoosterJointE6408(BoosterJointCfg):
    joint_model_name: str = "E6408"
    effort_limit: float = 68.0
    velocity_limit: float = 14.66
    knee_point_velocity: float = 1.88
    armature: float = 0.0478125


@configclass
class BoosterJointE4315(BoosterJointCfg):
    joint_model_name: str = "E4315"
    effort_limit: float = 76.0
    velocity_limit: float = 12.57
    knee_point_velocity: float = 2.62
    armature: float = 0.0339552


@configclass
class BoosterJointE4310(BoosterJointCfg):
    joint_model_name: str = "E4310"
    effort_limit: float = 38.3
    velocity_limit: float = 17.59
    knee_point_velocity: float = 7.85
    armature: float = 0.0282528


@configclass
class BoosterJointE6416(BoosterJointCfg):
    joint_model_name: str = "E6416"
    effort_limit: float = 112.0
    velocity_limit: float = 12.57
    knee_point_velocity: float = 2.09
    armature: float = 0.095625


@configclass
class BoosterJointR14(BoosterJointCfg):
    joint_model_name: str = "R14"
    effort_limit: float = 14.0
    velocity_limit: float = 33.51
    knee_point_velocity: float = 5.24
    armature: float = 0.001


@configclass
class BoosterJointHT4438(BoosterJointCfg):
    joint_model_name: str = "HT4438"
    effort_limit: float = 6.0
    velocity_limit: float = 7.85
    knee_point_velocity: float = 10.47
    armature: float = 0.001
