# -*- coding: utf-8 -*-
"""
DSCI 498 Project

dataset.py

Taylor Schultz
"""

import random
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_omniglot_few_shot_loader(cfg):
    transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.Omniglot(
        root=cfg.data_root,
        background=True,
        download=True,
        transform=transform
    )

    class_to_indices = {}
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        class_to_indices.setdefault(target, []).append(idx)

    selected_classes = random.sample(
        list(class_to_indices.keys()),
        k=min(cfg.num_classes_few_shot, len(class_to_indices))
    )

    few_shot_indices = []
    for c in selected_classes:
        indices = class_to_indices[c]
        random.shuffle(indices)
        few_shot_indices.extend(indices[:cfg.shots_per_class])

    few_shot_dataset = Subset(dataset, few_shot_indices)

    return DataLoader(
        few_shot_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True
    )
