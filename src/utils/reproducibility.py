"""Seed management for reproducibility."""

import random

import numpy as np

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set seeds for random, numpy, and torch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Return a numpy Generator for explicit RNG control."""
    return np.random.default_rng(seed)
