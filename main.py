# -*- coding: utf-8 -*-
"""
DSCI 498 Project

main.py

Taylor Schultz
"""

from config import Config
from train import train
from sample import sample_images

if __name__ == "__main__":
    cfg = Config()

    # Train
    train(cfg)

    # Sample from final checkpoint
    sample_images(cfg, ckpt_path=f"{cfg.output_dir}/model_epoch_{cfg.num_epochs}.pt")
