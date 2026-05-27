# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
    # -- SRL auxiliary losses
    arg_group.add_argument(
        "--srl_algo_name",
        type=str,
        default="ppo",
        choices={
            "ppo",
            "ppo_simsiam",
            "ppo_vae",
            "ppo_spr",
            "ppo_dvp",
        },
        help="Optional paper SRL objective to add to PPO.",
    )
    arg_group.add_argument("--srl_loss_coef", type=float, default=1.0, help="Global SRL auxiliary loss scale.")
    arg_group.add_argument("--srl_time_prop", type=int, default=10**12, help="Number of PPO updates using SRL loss.")
    arg_group.add_argument("--srl_data_prop", type=float, default=1.0, help="Fraction of each minibatch used for SRL.")
    arg_group.add_argument("--srl_interval", type=int, default=1, help="Apply SRL loss every N PPO updates.")
    arg_group.add_argument(
        "--srl_aug_q",
        type=str,
        default="gaussian_noise",
        choices={"none", "gaussian_noise", "random_masking", "random_amplitude_scaling"},
        help="Default first SRL observation augmentation.",
    )
    arg_group.add_argument(
        "--srl_aug_k",
        type=str,
        default="random_masking",
        choices={"none", "gaussian_noise", "random_masking", "random_amplitude_scaling"},
        help="Default second SRL observation augmentation.",
    )
    arg_group.add_argument("--aug_noise_sigma", type=float, default=0.05, help="Gaussian noise std for SRL augments.")
    arg_group.add_argument("--aug_mask_ratio", type=float, default=0.1, help="Element mask ratio for SRL augments.")
    arg_group.add_argument("--aug_scale_low", type=float, default=0.8, help="Low amplitude scale for SRL augments.")
    arg_group.add_argument("--aug_scale_high", type=float, default=1.2, help="High amplitude scale for SRL augments.")
    arg_group.add_argument("--simsiam_loss_coef", type=float, default=None, help="SimSiam loss scale.")
    arg_group.add_argument("--simsiam_hidden_dim", type=int, default=512, help="SimSiam predictor hidden dim.")
    arg_group.add_argument("--vae_loss_coef", type=float, default=None, help="VAE loss scale.")
    arg_group.add_argument("--vae_latent_dim", type=int, default=64, help="VAE latent dim.")
    arg_group.add_argument("--vae_hidden_dim", type=int, default=512, help="VAE decoder hidden dim.")
    arg_group.add_argument("--vae_kl_weight", type=float, default=1.0e-3, help="VAE KL loss weight.")
    arg_group.add_argument("--spr_loss_coef", type=float, default=None, help="SPR loss scale.")
    arg_group.add_argument("--spr_k", type=int, default=3, help="SPR prediction horizon.")
    arg_group.add_argument("--spr_skip", type=int, default=1, help="SPR prediction stride.")
    arg_group.add_argument("--spr_tau", type=float, default=0.99, help="SPR target-network EMA coefficient.")
    arg_group.add_argument("--spr_hidden_dim", type=int, default=512, help="SPR transition hidden dim.")
    arg_group.add_argument("--spr_projector_dim", type=int, default=64, help="SPR projector output dim.")
    arg_group.add_argument("--spr_avg_loss", action="store_true", default=True, help="Average SPR loss over horizons.")
    arg_group.add_argument("--spr_loss_decay", action="store_true", default=False, help="Decay SPR loss coefficient.")
    arg_group.add_argument(
        "--spr_aug",
        type=str,
        default="random_masking",
        choices={"none", "gaussian_noise", "random_masking", "random_amplitude_scaling"},
        help="SPR sequence augmentation.",
    )
    arg_group.add_argument("--dvp_loss_coef", type=float, default=None, help="DVP loss scale.")
    arg_group.add_argument("--dvp_hidden_dim", type=int, default=512, help="DVP predictor hidden dim.")
    arg_group.add_argument(
        "--dvp_view_mode",
        type=str,
        default="privileged_encoder",
        choices={"privileged_encoder", "replace_tail"},
        help="DVP target view construction. Use replace_tail only when actor obs has privileged placeholders.",
    )
    arg_group.add_argument(
        "--dvp_privileged_position",
        type=str,
        default="prefix",
        choices={"prefix", "suffix", "auto"},
        help="Where critic-only privileged observations are located relative to policy observations.",
    )
    arg_group.add_argument(
        "--srl_temporal_batch_size", type=int, default=1024, help="Max transition pairs used by temporal SRL losses."
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlBaseRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlBaseRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlBaseRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg


def parse_srl_cfg(args_cli: argparse.Namespace) -> dict:
    """Parse SRL-specific CLI arguments into a runner config dictionary."""
    cfg = {
        "srl_algo_name": args_cli.srl_algo_name,
        "srl_loss_coef": args_cli.srl_loss_coef,
        "srl_time_prop": args_cli.srl_time_prop,
        "srl_data_prop": args_cli.srl_data_prop,
        "srl_interval": args_cli.srl_interval,
        "srl_aug_q": args_cli.srl_aug_q,
        "srl_aug_k": args_cli.srl_aug_k,
        "aug_noise_sigma": args_cli.aug_noise_sigma,
        "aug_mask_ratio": args_cli.aug_mask_ratio,
        "aug_scale_low": args_cli.aug_scale_low,
        "aug_scale_high": args_cli.aug_scale_high,
        "simsiam_hidden_dim": args_cli.simsiam_hidden_dim,
        "vae_latent_dim": args_cli.vae_latent_dim,
        "vae_hidden_dim": args_cli.vae_hidden_dim,
        "vae_kl_weight": args_cli.vae_kl_weight,
        "spr_k": args_cli.spr_k,
        "spr_skip": args_cli.spr_skip,
        "spr_tau": args_cli.spr_tau,
        "spr_hidden_dim": args_cli.spr_hidden_dim,
        "spr_projector_dim": args_cli.spr_projector_dim,
        "spr_avg_loss": args_cli.spr_avg_loss,
        "spr_loss_decay": args_cli.spr_loss_decay,
        "spr_aug": args_cli.spr_aug,
        "dvp_hidden_dim": args_cli.dvp_hidden_dim,
        "dvp_view_mode": args_cli.dvp_view_mode,
        "dvp_privileged_position": args_cli.dvp_privileged_position,
        "srl_temporal_batch_size": args_cli.srl_temporal_batch_size,
    }
    optional_keys = [
        "simsiam_loss_coef",
        "vae_loss_coef",
        "spr_loss_coef",
        "dvp_loss_coef",
    ]
    for key in optional_keys:
        value = getattr(args_cli, key)
        if value is not None:
            cfg[key] = value
    return cfg
