# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import warnings
from dataclasses import asdict

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner
from torch.utils.tensorboard import SummaryWriter

from .actor_critic import ActorCriticWithEncoder
from .ppo import SrlPPO


class WandbSummaryWriterWithRunName(SummaryWriter):
    """Wandb writer that can decouple the wandb run name from the local log directory basename."""

    def __init__(self, log_dir: str, flush_secs: int, cfg):
        super().__init__(log_dir, flush_secs)

        try:
            import wandb
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")

        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.")

        entity = os.environ.get("WANDB_USERNAME")
        run_name = cfg.get("wandb_run_name") or os.path.split(log_dir)[-1]
        wandb.init(project=project, entity=entity, name=run_name)
        wandb.config.update({"log_dir": log_dir})

        self._wandb = wandb
        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

    def store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        self._wandb.config.update({"runner_cfg": runner_cfg})
        self._wandb.config.update({"policy_cfg": policy_cfg})
        self._wandb.config.update({"alg_cfg": alg_cfg})
        try:
            self._wandb.config.update({"env_cfg": env_cfg.to_dict()})
        except Exception:
            self._wandb.config.update({"env_cfg": asdict(env_cfg)})

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        self._wandb.log({self._map_path(tag): scalar_value}, step=global_step)

    def stop(self):
        self._wandb.finish()

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    def save_model(self, model_path, iter):
        self._wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path, iter=None):
        self._wandb.save(path, base_path=os.path.dirname(path))

    def _map_path(self, path):
        if path in self.name_map:
            return self.name_map[path]
        return path


class SrlOnPolicyRunner(OnPolicyRunner):
    """RSL-RL runner using the SRL-compatible actor encoder for PPO and PPO+SRL."""

    def _prepare_logging_writer(self):
        if self.cfg.get("logger", "tensorboard").lower() != "wandb":
            return super()._prepare_logging_writer()

        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.logger_type = "wandb"
            self.writer = WandbSummaryWriterWithRunName(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)

    def _construct_algorithm(self, obs) -> PPO | SrlPPO:
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        policy_class_name = self.policy_cfg.pop("class_name")
        if policy_class_name not in {"ActorCritic", "ActorCriticWithEncoder"}:
            raise ValueError(f"SRL training only supports non-recurrent actor-critic policies, got {policy_class_name}.")
        actor_critic = ActorCriticWithEncoder(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        alg_class_name = self.alg_cfg.pop("class_name")
        if alg_class_name not in {"PPO", "SrlPPO"}:
            raise ValueError(f"SRL training only supports PPO algorithms, got {alg_class_name}.")
        srl_cfg = self.cfg.get("srl", {})
        if srl_cfg.get("srl_algo_name", "ppo") == "ppo":
            alg = PPO(actor_critic, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)
        else:
            alg = SrlPPO(
                actor_critic,
                srl_cfg=srl_cfg,
                device=self.device,
                **self.alg_cfg,
                multi_gpu_cfg=self.multi_gpu_cfg,
            )

        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )
        return alg

    def save(self, path: str, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if hasattr(self.alg, "srl"):
            saved_dict["srl_state_dict"] = self.alg.srl.state_dict()
            saved_dict["srl_step"] = self.alg.srl_step
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None):
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if hasattr(self.alg, "srl") and "srl_state_dict" in loaded_dict:
            self.alg.srl.load_state_dict(loaded_dict["srl_state_dict"])
            self.alg.srl_step = loaded_dict.get("srl_step", self.alg.srl_step)
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def train_mode(self):
        self.alg.policy.train()
        if hasattr(self.alg, "srl"):
            self.alg.srl.train()
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.train()

    def eval_mode(self):
        self.alg.policy.eval()
        if hasattr(self.alg, "srl"):
            self.alg.srl.eval()
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.eval()
