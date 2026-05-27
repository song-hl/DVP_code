# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_augs import apply_augmentation


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
    )


class SrlObjective(nn.Module):
    requires_next_obs = False
    requires_sequence = False
    requires_privileged_obs = False

    def __init__(self, encoder_online: nn.Module, cfg: dict):
        super().__init__()
        self.__dict__["encoder_online"] = encoder_online
        self.cfg = cfg
        self.loss_coef = cfg.get("srl_loss_coef", 1.0)

    def _views(self, obs: torch.Tensor, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
        aug_q = self.cfg.get(f"{prefix}_aug_q", self.cfg.get("srl_aug_q", "gaussian_noise"))
        aug_k = self.cfg.get(f"{prefix}_aug_k", self.cfg.get("srl_aug_k", "random_masking"))
        return apply_augmentation(obs, aug_q, self.cfg), apply_augmentation(obs, aug_k, self.cfg)

    def update_misc(self):
        pass


class _PrivilegedMixin:
    def _privileged_dim(self) -> int:
        actor_dim = self.cfg.get("actor_obs_dim")
        critic_dim = self.cfg.get("critic_obs_dim")
        if actor_dim is None or critic_dim is None or critic_dim <= actor_dim:
            raise ValueError("This SRL objective requires critic observations with critic_dim > actor_dim.")
        return critic_dim - actor_dim


class SimSiam(SrlObjective):
    def __init__(self, encoder_online: nn.Module, cfg: dict):
        super().__init__(encoder_online, cfg)
        self.loss_coef = cfg.get("simsiam_loss_coef", self.loss_coef)
        dim = getattr(encoder_online, "output_dim", None)
        if dim is None:
            raise ValueError("SimSiam requires an actor encoder with an output_dim attribute.")
        hidden_dim = cfg.get("simsiam_hidden_dim", max(256, dim))
        self.predictor = _mlp(dim, hidden_dim, dim)

    def _loss(self, p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        p = F.normalize(p, dim=-1)
        z = F.normalize(z.detach(), dim=-1)
        return -F.cosine_similarity(p, z, dim=-1).mean()

    def compute_loss(self, obs: torch.Tensor, next_obs: torch.Tensor | None = None, actions: torch.Tensor | None = None):
        x1, x2 = self._views(obs, "simsiam")
        z1 = self.encoder_online(x1)
        z2 = self.encoder_online(x2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return self.loss_coef * 0.5 * (self._loss(p1, z2) + self._loss(p2, z1))


class VAE(SrlObjective):
    def __init__(self, encoder_online: nn.Module, cfg: dict):
        super().__init__(encoder_online, cfg)
        self.loss_coef = cfg.get("vae_loss_coef", self.loss_coef)
        dim = getattr(encoder_online, "output_dim", None)
        if dim is None:
            raise ValueError("VAE requires an actor encoder with an output_dim attribute.")
        latent_dim = cfg.get("vae_latent_dim", max(16, dim // 2))
        hidden_dim = cfg.get("vae_hidden_dim", max(256, dim))
        self.kl_weight = cfg.get("vae_kl_weight", 1.0e-3)
        self.fc_mu = nn.Linear(dim, latent_dim)
        self.fc_logvar = nn.Linear(dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, dim),
        )

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def compute_loss(self, obs: torch.Tensor, next_obs: torch.Tensor | None = None, actions: torch.Tensor | None = None):
        features = self.encoder_online(obs)
        mu = self.fc_mu(features)
        logvar = torch.clamp(self.fc_logvar(features), min=-10.0, max=10.0)
        latent = self._reparameterize(mu, logvar)
        recon = self.decoder(latent)
        recon_loss = F.mse_loss(recon, features.detach())
        kl_loss = torch.mean(-0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
        return self.loss_coef * (recon_loss + self.kl_weight * kl_loss)


class DVP(SrlObjective, _PrivilegedMixin):
    requires_privileged_obs = True

    def __init__(self, encoder_online: nn.Module, cfg: dict):
        super().__init__(encoder_online, cfg)
        self.loss_coef = cfg.get("dvp_loss_coef", self.loss_coef)
        dim = getattr(encoder_online, "output_dim", None)
        if dim is None:
            raise ValueError("DVP requires an actor encoder with an output_dim attribute.")
        hidden_dim = cfg.get("dvp_hidden_dim", max(256, dim))
        self.view_mode = cfg.get("dvp_view_mode", "privileged_encoder")
        self.predictor = _mlp(dim, hidden_dim, dim)
        if self.view_mode == "privileged_encoder":
            self.privileged_encoder = _mlp(self._privileged_dim(), hidden_dim, dim)
        elif self.view_mode != "replace_tail":
            raise ValueError(f"Unknown dvp_view_mode: {self.view_mode}")

    def _loss(self, p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        p = F.normalize(p, dim=-1)
        z = F.normalize(z.detach(), dim=-1)
        return -F.cosine_similarity(p, z, dim=-1).mean()

    def compute_loss(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        privileged_obs: torch.Tensor | None = None,
    ):
        if privileged_obs is None:
            raise ValueError("DVP requires privileged observations.")
        z1 = self.encoder_online(obs)
        if self.view_mode == "replace_tail":
            priv_dim = privileged_obs.shape[-1]
            if priv_dim <= 0 or priv_dim >= obs.shape[-1]:
                raise ValueError(f"Invalid DVP privileged dimension: {priv_dim} for obs dim {obs.shape[-1]}.")
            z2 = self.encoder_online(torch.cat([obs[:, :-priv_dim], privileged_obs], dim=-1))
        else:
            z2 = self.privileged_encoder(privileged_obs)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return self.loss_coef * 0.5 * (self._loss(p1, z2) + self._loss(p2, z1))


class SPR(SrlObjective):
    requires_sequence = True

    def __init__(self, encoder_online: nn.Module, cfg: dict, action_dim: int):
        super().__init__(encoder_online, cfg)
        self.loss_coef = cfg.get("spr_loss_coef", self.loss_coef)
        self.k = cfg.get("spr_k", 3)
        self.skip = max(1, cfg.get("spr_skip", 1))
        self.tau = cfg.get("spr_tau", 0.99)
        self.avg_loss = cfg.get("spr_avg_loss", True)
        self.loss_decay = cfg.get("spr_loss_decay", False)
        dim = getattr(encoder_online, "output_dim", None)
        if dim is None:
            raise ValueError("SPR requires an actor encoder with an output_dim attribute.")
        hidden_dim = cfg.get("spr_hidden_dim", max(512, dim))
        proj_dim = cfg.get("spr_projector_dim", max(64, dim // 2))
        self.encoder_target = copy.deepcopy(encoder_online)
        self.proj_online = nn.Linear(dim, proj_dim)
        self.proj_target = copy.deepcopy(self.proj_online)
        for module in (self.encoder_target, self.proj_target):
            module.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.transition = nn.Sequential(
            nn.Linear(dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def compute_loss(self, obs: torch.Tensor, next_obs: torch.Tensor | None = None, actions: torch.Tensor | None = None):
        if actions is None:
            raise ValueError("SPR requires state sequences and action sequences.")
        states = apply_augmentation(obs, self.cfg.get("spr_aug", "random_masking"), self.cfg)
        z = self.encoder_online(states[:, 0])
        loss = torch.zeros((), device=states.device)
        num_predictions = 0
        for step in range(self.skip, self.k + 1, self.skip):
            action = actions[:, step - 1]
            z = self.transition(torch.cat([z, action], dim=-1))
            y_pred = F.normalize(self.predictor(self.proj_online(z)), dim=-1)
            with torch.no_grad():
                z_target = self.encoder_target(states[:, step])
                y_target = F.normalize(self.proj_target(z_target), dim=-1)
            loss = loss + F.mse_loss(y_pred, y_target, reduction="none").sum(dim=-1).mean()
            num_predictions += 1
        if self.avg_loss and num_predictions > 0:
            loss = loss / num_predictions
        return self.loss_coef * loss

    @torch.no_grad()
    def update_misc(self):
        for online, target in zip(self.encoder_online.parameters(), self.encoder_target.parameters(), strict=True):
            target.data.mul_(self.tau).add_(online.data, alpha=1.0 - self.tau)
        for online, target in zip(self.proj_online.parameters(), self.proj_target.parameters(), strict=True):
            target.data.mul_(self.tau).add_(online.data, alpha=1.0 - self.tau)
        if self.loss_decay:
            self.loss_coef *= 0.999


def build_srl_objective(
    name: str,
    encoder_online: nn.Module,
    cfg: dict,
    action_dim: int | None = None,
) -> SrlObjective:
    objective_by_name = {
        "ppo_simsiam": SimSiam,
        "ppo_vae": VAE,
        "ppo_dvp": DVP,
        "ppo_spr": SPR,
    }
    if name not in objective_by_name:
        raise ValueError(f"Unsupported SRL algorithm: {name}")
    if name == "ppo_spr":
        if action_dim is None:
            raise ValueError(f"{name} requires action_dim.")
        return objective_by_name[name](encoder_online, cfg, action_dim)
    return objective_by_name[name](encoder_online, cfg)
