# -*- coding: utf-8 -*-
"""
DSCI 498 Project

dataset.py

Taylor Schultz
"""

import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import Omniglot
import torchvision.transforms as T


def get_omniglot_few_shot_loader(cfg):
    """
    Loads Omniglot using torchvision (no TensorFlow) and constructs a few-shot subset.
    """

    transform = T.Compose([
        T.Resize((cfg.img_size, cfg.img_size)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])

    # Load Omniglot (background = training set)
    full_dataset = Omniglot(
        root=cfg.data_root,
        background=True,
        download=True,
        transform=transform
    )

    # Build class → indices mapping
    class_to_indices = {}
    for idx in range(len(full_dataset)):
        _, label = full_dataset[idx]
        class_to_indices.setdefault(label, []).append(idx)

    # Select few-shot classes
    selected_classes = random.sample(
        list(class_to_indices.keys()),
        k=min(cfg.num_classes_few_shot, len(class_to_indices))
    )

    few_shot_indices = []
    for c in selected_classes:
        indices = class_to_indices[c]
        random.shuffle(indices)
        few_shot_indices.extend(indices[:cfg.shots_per_class])

    few_shot_dataset = Subset(full_dataset, few_shot_indices)

    return DataLoader(
        few_shot_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True
    )
