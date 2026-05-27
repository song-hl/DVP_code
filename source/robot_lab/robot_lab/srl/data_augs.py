# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


def gaussian_noise(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    return x + torch.randn_like(x) * sigma


def random_amplitude_scaling(x: torch.Tensor, low: float = 0.8, high: float = 1.2) -> torch.Tensor:
    scale = torch.empty(x.shape[0], 1, device=x.device, dtype=x.dtype).uniform_(low, high)
    return x * scale


def random_masking(x: torch.Tensor, mask_ratio: float = 0.1, mask_value: float = 0.0) -> torch.Tensor:
    mask = torch.rand_like(x) >= mask_ratio
    return torch.where(mask, x, torch.full_like(x, mask_value))


def apply_augmentation(x: torch.Tensor, name: str, cfg: dict) -> torch.Tensor:
    if name == "none":
        return x
    if name == "gaussian_noise":
        return gaussian_noise(x, sigma=cfg.get("aug_noise_sigma", 0.05))
    if name == "random_masking":
        return random_masking(
            x,
            mask_ratio=cfg.get("aug_mask_ratio", 0.1),
            mask_value=cfg.get("aug_mask_value", 0.0),
        )
    if name == "random_amplitude_scaling":
        return random_amplitude_scaling(
            x,
            low=cfg.get("aug_scale_low", 0.8),
            high=cfg.get("aug_scale_high", 1.2),
        )
    raise ValueError(f"Unknown SRL augmentation: {name}")
