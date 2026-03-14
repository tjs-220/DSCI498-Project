# -*- coding: utf-8 -*-
"""
DSCI 498 Project

sample.py

Taylor Schultz
"""

import torch
from tqdm import tqdm
from torchvision import utils as vutils
import os
import math

from diffusion import DiffusionSchedule
from unet import UNet


@torch.no_grad()
def sample_images(cfg, ckpt_path, out_file="samples.png", n=64, guidance_scale=3.0):
    device = cfg.device
    model = UNet(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    diffusion = DiffusionSchedule(cfg)
    x = torch.randn(n, cfg.channels, cfg.img_size, cfg.img_size, device=device)

    for i in tqdm(reversed(range(diffusion.num_timesteps)), total=diffusion.num_timesteps):
        t = torch.full((n,), i, device=device, dtype=torch.long)

        cond = torch.ones(n, device=device, dtype=torch.long)
        uncond = torch.zeros(n, device=device, dtype=torch.long)

        eps_cond = model(x, t, cond)
        eps_uncond = model(x, t, uncond)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        beta = diffusion.betas[t].view(-1, 1, 1, 1).to(device)
        sqrt_one_minus = diffusion.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(device)
        sqrt_recip = diffusion.sqrt_recip_alphas[t].view(-1, 1, 1, 1).to(device)

        model_mean = sqrt_recip * (x - beta / sqrt_one_minus * eps)

        if i > 0:
            var = diffusion.posterior_variance[t].view(-1, 1, 1, 1).to(device)
            x = model_mean + torch.sqrt(var) * torch.randn_like(x)
        else:
            x = model_mean

    x = (x.clamp(-1, 1) + 1) / 2
    grid = vutils.make_grid(x, nrow=int(math.sqrt(n)))
    os.makedirs(cfg.output_dir, exist_ok=True)
    vutils.save_image(grid, os.path.join(cfg.output_dir, out_file))
