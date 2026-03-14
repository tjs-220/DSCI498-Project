# -*- coding: utf-8 -*-
"""
DSCI 498 Project

unet.py

Taylor Schultz
"""

import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch)
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.res = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res(x)


class UNet(nn.Module):
    def __init__(self, cfg, time_dim=128, cond_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        self.cond_emb = nn.Embedding(2, cond_dim)
        self.cond_proj = nn.Linear(cond_dim, time_dim)

        ch = 32
        self.conv_in = nn.Conv2d(cfg.channels, ch, 3, padding=1)

        self.down1 = ResidualBlock(ch, ch, time_dim)
        self.down2 = ResidualBlock(ch, ch * 2, time_dim)
        self.down3 = ResidualBlock(ch * 2, ch * 4, time_dim)
        self.pool = nn.AvgPool2d(2)

        self.mid1 = ResidualBlock(ch * 4, ch * 4, time_dim)
        self.mid2 = ResidualBlock(ch * 4, ch * 4, time_dim)

        self.up1 = ResidualBlock(ch * 4 + ch * 4, ch * 2, time_dim)
        self.up2 = ResidualBlock(ch * 2 + ch * 2, ch, time_dim)
        self.up3 = ResidualBlock(ch + ch, ch, time_dim)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_out = nn.Conv2d(ch, cfg.channels, 3, padding=1)

    def forward(self, x, t, cond_flag):
        t_emb = self.time_mlp(t)
        c_emb = self.cond_proj(self.cond_emb(cond_flag))
        t_emb = t_emb + c_emb

        x = self.conv_in(x)
        d1 = self.down1(x, t_emb)
        x = self.pool(d1)
        d2 = self.down2(x, t_emb)
        x = self.pool(d2)
        d3 = self.down3(x, t_emb)
        x = self.pool(d3)

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        x = self.upsample(x)
        x = self.up1(torch.cat([x, d3], dim=1), t_emb)

        x = self.upsample(x)
        x = self.up2(torch.cat([x, d2], dim=1), t_emb)

        x = self.upsample(x)
        x = self.up3(torch.cat([x, d1], dim=1), t_emb)

        return self.conv_out(x)
