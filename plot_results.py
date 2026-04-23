"""
DSCI 498 Project

plot_results.py

Generates training loss curves and sample visualizations.
"""

import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

from diffusion import DiffusionSchedule
from unet import UNet
from config import Config


def plot_loss_curve(loss_path, out_path="loss_curve.png"):
    loss_history = torch.load(loss_path)

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, linewidth=1.5)
    plt.title("Training Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved loss curve to {out_path}")


@torch.no_grad()
def visualize_denoising(cfg, ckpt_path, out_path="denoising_steps.png"):
    device = cfg.device
    model = UNet(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    diffusion = DiffusionSchedule(cfg)

    # Start from random noise
    x = torch.randn(1, cfg.channels, cfg.img_size, cfg.img_size, device=device)

    frames = []
    timesteps_to_show = [999, 750, 500, 250, 100, 50, 10, 0]

    for t in timesteps_to_show:
        t_tensor = torch.tensor([t], device=device, dtype=torch.long)
        cond = torch.ones(1, device=device, dtype=torch.long)

        eps = model(x, t_tensor, cond)

        beta = diffusion.betas[t].view(1, 1, 1, 1)
        sqrt_one_minus = diffusion.sqrt_one_minus_alphas_cumprod[t].view(1, 1, 1, 1)
        sqrt_recip = diffusion.sqrt_recip_alphas[t].view(1, 1, 1, 1)

        model_mean = sqrt_recip * (x - beta / sqrt_one_minus * eps)

        x = model_mean

        frames.append(x.clone())

    # Convert frames to grid
    frames = torch.cat(frames, dim=0)
    frames = (frames.clamp(-1, 1) + 1) / 2
    grid = make_grid(frames, nrow=len(timesteps_to_show))

    save_image(grid, out_path)
    print(f"Saved denoising visualization to {out_path}")


if __name__ == "__main__":
    cfg = Config()
    out_dir = cfg.output_dir

    # 1. Loss curve
    plot_loss_curve(
        loss_path=os.path.join(out_dir, "loss_history.pt"),
        out_path=os.path.join(out_dir, "loss_curve.png")
    )

    # 2. Denoising visualization
    visualize_denoising(
        cfg,
        ckpt_path=os.path.join(out_dir, f"model_epoch_{cfg.num_epochs}.pt"),
        out_path=os.path.join(out_dir, "denoising_steps.png")
    )
