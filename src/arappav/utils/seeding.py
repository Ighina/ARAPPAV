"""Reproducible seeding utilities."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic ops have a performance cost; only enable if needed.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
