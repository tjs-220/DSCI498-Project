# -*- coding: utf-8 -*-
"""
DSCI 498 Project

config.py

Taylor Schultz
"""

from dataclasses import dataclass
import torch


@dataclass
class Config:
    data_root: str = "./data"
    output_dir: str = "./outputs"
    img_size: int = 28
    channels: int = 1

    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    batch_size: int = 64
    num_epochs: int = 20
    lr: float = 2e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    seed: int = 42

    num_classes_few_shot: int = 20
    shots_per_class: int = 5

    p_uncond: float = 0.1

    num_samples: int = 64
    sample_every: int = 2

    ckpt_every: int = 5
    ckpt_path: str = "model_last.pt"
