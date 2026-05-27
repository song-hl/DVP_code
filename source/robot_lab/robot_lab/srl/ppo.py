# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.algorithms import PPO

from .objectives import build_srl_objective


class SrlPPO(PPO):
    """PPO with an auxiliary state representation learning loss."""

    def __init__(self, policy, srl_cfg: dict, *args, **kwargs):
        self.srl_cfg = srl_cfg
        self.srl_algo_name = srl_cfg["srl_algo_name"]
        super().__init__(policy, *args, **kwargs)
        self.srl_cfg.setdefault("actor_obs_dim", getattr(self.policy, "actor_obs_dim", None))
        self.srl_cfg.setdefault("critic_obs_dim", getattr(self.policy, "critic_obs_dim", None))
        self.srl = build_srl_objective(
            self.srl_algo_name,
            self.policy.actor_encoder,
            self.srl_cfg,
            action_dim=self.policy.action_dim,
        ).to(self.device)
        self.srl_step = 0
        self._warned_missing_privileged_obs = False
        self.optimizer = optim.Adam(
            [
                {"params": self.policy.parameters()},
                {"params": self.srl.parameters()},
            ],
            lr=self.learning_rate,
        )

    def _sample_temporal_srl_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.storage.num_transitions_per_env < 2:
            empty = torch.empty(0, device=self.device)
            return empty, empty, empty

        obs = self.storage.observations
        obs_t = self.policy.get_actor_obs(obs[:-1]).flatten(0, 1)
        obs_next = self.policy.get_actor_obs(obs[1:]).flatten(0, 1)
        actions = self.storage.actions[:-1].flatten(0, 1)
        valid = ~self.storage.dones[:-1].flatten(0, 1).squeeze(-1).bool()
        obs_t = obs_t[valid]
        obs_next = obs_next[valid]
        actions = actions[valid]

        data_prop = float(self.srl_cfg.get("srl_data_prop", 1.0))
        if data_prop <= 0.0 or obs_t.shape[0] == 0:
            empty = torch.empty(0, device=self.device)
            return empty, empty, empty
        max_batch_size = int(self.srl_cfg.get("srl_temporal_batch_size", 1024))
        target_size = obs_t.shape[0] if data_prop >= 1.0 else max(1, int(obs_t.shape[0] * data_prop))
        target_size = min(target_size, max_batch_size, obs_t.shape[0])
        indices = torch.randperm(obs_t.shape[0], device=self.device)[:target_size]
        return obs_t[indices], obs_next[indices], actions[indices]

    def _sample_temporal_privileged_srl_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.storage.num_transitions_per_env < 2:
            empty = torch.empty(0, device=self.device)
            return empty, empty, empty, empty

        obs = self.storage.observations
        actor_obs = self.policy.get_actor_obs(obs[:-1]).flatten(0, 1)
        next_actor_obs = self.policy.get_actor_obs(obs[1:]).flatten(0, 1)
        critic_obs = self.policy.get_critic_obs(obs[:-1]).flatten(0, 1)
        actions = self.storage.actions[:-1].flatten(0, 1)
        valid = ~self.storage.dones[:-1].flatten(0, 1).squeeze(-1).bool()
        actor_obs = actor_obs[valid]
        next_actor_obs = next_actor_obs[valid]
        critic_obs = critic_obs[valid]
        actions = actions[valid]

        privileged_obs = self._extract_privileged_obs(actor_obs, critic_obs)
        if privileged_obs is None:
            empty = torch.empty(0, device=self.device)
            return empty, empty, empty, empty

        data_prop = float(self.srl_cfg.get("srl_data_prop", 1.0))
        if data_prop <= 0.0 or actor_obs.shape[0] == 0:
            empty = torch.empty(0, device=self.device)
            return empty, empty, empty, empty
        max_batch_size = int(self.srl_cfg.get("srl_temporal_batch_size", 1024))
        target_size = actor_obs.shape[0] if data_prop >= 1.0 else max(1, int(actor_obs.shape[0] * data_prop))
        target_size = min(target_size, max_batch_size, actor_obs.shape[0])
        indices = torch.randperm(actor_obs.shape[0], device=self.device)[:target_size]
        return actor_obs[indices], next_actor_obs[indices], actions[indices], privileged_obs[indices]

    def _sample_sequence_srl_batch(self, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.storage.num_transitions_per_env <= horizon:
            return torch.empty(0, device=self.device), torch.empty(0, device=self.device)

        actor_obs = self.policy.get_actor_obs(self.storage.observations)
        state_sequences = []
        action_sequences = []
        for step in range(self.storage.num_transitions_per_env - horizon):
            valid = ~self.storage.dones[step : step + horizon].squeeze(-1).any(dim=0).bool()
            if valid.any():
                state_sequences.append(actor_obs[step : step + horizon + 1, valid].transpose(0, 1))
                action_sequences.append(self.storage.actions[step : step + horizon, valid].transpose(0, 1))
        if not state_sequences:
            return torch.empty(0, device=self.device), torch.empty(0, device=self.device)

        states = torch.cat(state_sequences, dim=0)
        actions = torch.cat(action_sequences, dim=0)
        data_prop = float(self.srl_cfg.get("srl_data_prop", 1.0))
        if data_prop <= 0.0:
            return torch.empty(0, device=self.device), torch.empty(0, device=self.device)
        max_batch_size = int(self.srl_cfg.get("srl_temporal_batch_size", 1024))
        target_size = states.shape[0] if data_prop >= 1.0 else max(1, int(states.shape[0] * data_prop))
        target_size = min(target_size, max_batch_size, states.shape[0])
        indices = torch.randperm(states.shape[0], device=self.device)[:target_size]
        return states[indices], actions[indices]

    def _extract_privileged_obs(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor | None:
        actor_dim = actor_obs.shape[-1]
        critic_dim = critic_obs.shape[-1]
        if critic_dim <= actor_dim:
            return None

        priv_dim = critic_dim - actor_dim
        position = self.srl_cfg.get("dvp_privileged_position", "prefix")
        if position == "prefix":
            return critic_obs[:, :priv_dim]
        if position == "suffix":
            return critic_obs[:, actor_dim:]
        if position != "auto":
            raise ValueError(f"Unknown dvp_privileged_position: {position}")

        atol = float(self.srl_cfg.get("dvp_match_atol", 1.0e-5))
        rtol = float(self.srl_cfg.get("dvp_match_rtol", 1.0e-5))
        if torch.allclose(critic_obs[:, -actor_dim:], actor_obs, atol=atol, rtol=rtol):
            return critic_obs[:, :priv_dim]
        if torch.allclose(critic_obs[:, :actor_dim], actor_obs, atol=atol, rtol=rtol):
            return critic_obs[:, actor_dim:]

        if not self._warned_missing_privileged_obs:
            print(
                "[WARN] SRL could not infer privileged observations: critic observations do not contain policy "
                "observations as a prefix or suffix. Skipping privileged SRL loss for this task."
            )
            self._warned_missing_privileged_obs = True
        return None

    def _compute_srl_loss(self, obs_batch) -> torch.Tensor:
        if self.srl_step > self.srl_cfg.get("srl_time_prop", 10**12):
            return torch.zeros((), device=self.device)
        if self.srl_step % max(1, self.srl_cfg.get("srl_interval", 1)) != 0:
            return torch.zeros((), device=self.device)

        if getattr(self.srl, "requires_next_privileged_obs", False):
            actor_obs, next_actor_obs, actions, privileged_obs = self._sample_temporal_privileged_srl_batch()
            if actor_obs.shape[0] < 2:
                return torch.zeros((), device=self.device)
            return self.srl.compute_loss(actor_obs, next_actor_obs, actions, privileged_obs=privileged_obs)

        if getattr(self.srl, "requires_next_obs", False):
            actor_obs, next_actor_obs, actions = self._sample_temporal_srl_batch()
            if actor_obs.shape[0] < 2:
                return torch.zeros((), device=self.device)
            return self.srl.compute_loss(actor_obs, next_actor_obs, actions)

        if getattr(self.srl, "requires_sequence", False):
            states, actions = self._sample_sequence_srl_batch(getattr(self.srl, "k", 1))
            if states.shape[0] < 2:
                return torch.zeros((), device=self.device)
            return self.srl.compute_loss(states, actions=actions)

        actor_obs = self.policy.get_actor_obs(obs_batch)
        critic_obs = None
        if getattr(self.srl, "requires_privileged_obs", False):
            critic_obs = self.policy.get_critic_obs(obs_batch)
        data_prop = float(self.srl_cfg.get("srl_data_prop", 1.0))
        if data_prop <= 0.0:
            return torch.zeros((), device=self.device)
        if data_prop < 1.0:
            mask = torch.rand(actor_obs.shape[0], device=actor_obs.device) < data_prop
            actor_obs = actor_obs[mask]
            if critic_obs is not None:
                critic_obs = critic_obs[mask]
        if actor_obs.shape[0] < 2:
            return torch.zeros((), device=self.device)
        if getattr(self.srl, "requires_privileged_obs", False):
            privileged_obs = self._extract_privileged_obs(actor_obs, critic_obs)
            if privileged_obs is None:
                return torch.zeros((), device=self.device)
            return self.srl.compute_loss(actor_obs, privileged_obs=privileged_obs)
        return self.srl.compute_loss(actor_obs)

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_srl_loss = 0
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            num_aug = 1
            original_batch_size = obs_batch.batch_size[0]
            srl_obs_batch = obs_batch[:original_batch_size]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    num_aug = int(obs_batch.shape[0] / original_batch_size)
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                rnd_loss = torch.nn.MSELoss()(predicted_embedding, target_embedding)

            srl_loss = self._compute_srl_loss(srl_obs_batch)
            loss = loss + srl_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(chain(self.policy.parameters(), self.srl.parameters()), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()
            self.srl.update_misc()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_srl_loss += srl_loss.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_srl_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        self.storage.clear()
        self.srl_step += 1

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "srl": mean_srl_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        return loss_dict

    def broadcast_parameters(self):
        model_params = [self.policy.state_dict(), self.srl.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])
        self.srl.load_state_dict(model_params[1])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[2])

    def reduce_parameters(self):
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        grads += [param.grad.view(-1) for param in self.srl.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        all_params = chain(self.policy.parameters(), self.srl.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
