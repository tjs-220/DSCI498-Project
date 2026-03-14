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
import tensorflow_datasets as tfds
from omniglot_tfds import Omniglot


class OmniglotTorchWrapper(Dataset):
    """
    Wraps the TFDS Omniglot dataset so it behaves like a PyTorch Dataset.
    Converts TF tensors → NumPy → PyTorch tensors.
    """

    def __init__(self, tfds_split, cfg, transform=None):
        self.cfg = cfg
        self.transform = transform

        # Load TFDS dataset
        builder = Omniglot()
        builder.download_and_prepare()
        self.ds = builder.as_dataset(split=tfds_split)

        # Convert TFDS dataset to a list for indexing
        self.samples = list(tfds.as_numpy(self.ds))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = sample["image"]  # shape (105,105,3)
        label = sample["label"]

        # Convert to float32 and normalize to [0,1]
        img = img.astype(np.float32) / 255.0

        # Convert to grayscale (Omniglot is grayscale originally)
        img = np.mean(img, axis=2, keepdims=True)  # (105,105,1)

        # Resize to model size
        img = torch.tensor(img).permute(2, 0, 1)  # (1,105,105)

        if self.transform:
            img = self.transform(img)

        return img, label


def get_omniglot_few_shot_loader(cfg):
    """
    Loads Omniglot using TFDS and constructs a few-shot subset.
    """

    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((cfg.img_size, cfg.img_size)),
        T.Normalize((0.5,), (0.5,))
    ])

    # Load TRAIN split from TFDS
    full_dataset = OmniglotTorchWrapper("train", cfg, transform=transform)

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
