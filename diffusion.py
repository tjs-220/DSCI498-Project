# -*- coding: utf-8 -*-
"""
DSCI 498 Project

diffusion.py

Taylor Schultz
"""

import torch
import torch.nn.functional as F


class DiffusionSchedule:
    def __init__(self, cfg):
        self.num_timesteps = cfg.num_timesteps
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_timesteps)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), self.alphas_cumprod[:-1]], dim=0
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_om = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ac * x0 + sqrt_om * noise

    def _extract(self, a, t, x_shape):
        out = a.gather(-1, t.cpu()).to(t.device)
        return out.view(-1, *([1] * (len(x_shape) - 1)))


def diffusion_loss(model, diffusion, x0, t, cond_flag):
    noise = torch.randn_like(x0)
    x_noisy = diffusion.q_sample(x0, t, noise)
    noise_pred = model(x_noisy, t, cond_flag)
    return F.mse_loss(noise_pred, noise)
