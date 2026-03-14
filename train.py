# -*- coding: utf-8 -*-
"""
DSCI 498 Project

train.py

Taylor Schultz
"""

import os
import math
import torch
from tqdm import tqdm
from torchvision import utils as vutils

from dataset import get_omniglot_few_shot_loader
from diffusion import DiffusionSchedule, diffusion_loss
from unet import UNet


def train(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = cfg.device

    loader = get_omniglot_few_shot_loader(cfg)
    diffusion = DiffusionSchedule(cfg)
    model = UNet(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for x, _ in pbar:
            x = x.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (x.size(0),), device=device)
            cond_flag = torch.ones(x.size(0), device=device, dtype=torch.long)
            cond_flag[torch.rand(x.size(0), device=device) < cfg.p_uncond] = 0

            loss = diffusion_loss(model, diffusion, x, t, cond_flag)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})

        if epoch % cfg.ckpt_every == 0:
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, f"model_epoch_{epoch}.pt"))
