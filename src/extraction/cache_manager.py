"""Save/load feature matrices as .npz files."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ..utils.config_loader import get_results_dir

logger = logging.getLogger(__name__)


def save_features(
    per_input: np.ndarray,
    aggregated: np.ndarray,
    tag: str = "base",
    metadata: Optional[dict] = None,
) -> Path:
    """Save feature matrices to disk.

    Parameters
    ----------
    per_input : np.ndarray
        Shape (n_prompts, n_layers, n_heads, n_features).
    aggregated : np.ndarray
        Shape (n_layers * n_heads, n_features).
    tag : str
        Identifier for this extraction run (e.g. "base", "sft", "dpo").
    metadata : dict, optional
        Additional metadata to store.

    Returns
    -------
    Path
        Path to saved .npz file.
    """
    out_dir = get_results_dir("features")
    path = out_dir / f"features_{tag}.npz"

    save_dict = {
        "per_input": per_input,
        "aggregated": aggregated,
    }
    if metadata:
        for k, v in metadata.items():
            save_dict[f"meta_{k}"] = np.array(v) if not isinstance(v, np.ndarray) else v

    np.savez_compressed(path, **save_dict)
    logger.info(f"Saved features to {path} (per_input: {per_input.shape}, aggregated: {aggregated.shape})")
    return path


def load_features(tag: str = "base") -> dict[str, np.ndarray]:
    """Load feature matrices from disk.

    Parameters
    ----------
    tag : str
        Identifier for the extraction run.

    Returns
    -------
    dict[str, np.ndarray]
        Dict with keys "per_input", "aggregated", and any metadata keys.
    """
    out_dir = get_results_dir("features")
    path = out_dir / f"features_{tag}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Feature cache not found: {path}")

    data = dict(np.load(path, allow_pickle=True))
    logger.info(f"Loaded features from {path}")
    return data


def features_exist(tag: str = "base") -> bool:
    """Check if cached features exist for a given tag."""
    return (get_results_dir("features") / f"features_{tag}.npz").exists()
