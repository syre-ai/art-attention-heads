"""Unified interface for FuzzyART, GaussianART, and HypersphereART."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from artlib import FuzzyART, GaussianART, HypersphereART

from ..utils.config_loader import load_config

logger = logging.getLogger(__name__)

ART_VARIANTS = {"fuzzy_art": FuzzyART, "gaussian_art": GaussianART, "hypersphere_art": HypersphereART}


@dataclass
class ClusteringResult:
    """Result of a single ART clustering run."""
    variant: str
    rho: float
    n_clusters: int
    labels: np.ndarray
    centers: list[np.ndarray]
    model: object  # Fitted ART model
    metadata: dict = field(default_factory=dict)


def create_art_model(variant: str, rho: float, n_features: int = 14, model_params: Optional[dict] = None) -> object:
    """Create an ART model instance.

    Parameters
    ----------
    variant : str
        One of "fuzzy_art", "gaussian_art", "hypersphere_art".
    rho : float
        Vigilance parameter.
    n_features : int
        Number of input features (before complement coding).
    model_params : dict, optional
        Override config-based params (e.g. {"sigma_init_scale": 0.5}).

    Returns
    -------
    ART model instance.
    """
    cfg = load_config("clustering")
    variant_cfg = cfg["variants"][variant]["params"].copy()
    if model_params:
        variant_cfg.update(model_params)

    if variant == "fuzzy_art":
        return FuzzyART(
            rho=rho,
            alpha=variant_cfg.get("alpha", 0.001),
            beta=variant_cfg.get("beta", 1.0),
        )
    elif variant == "gaussian_art":
        sigma_scale = variant_cfg.get("sigma_init_scale", 0.1)
        return GaussianART(
            rho=rho,
            sigma_init=np.ones(n_features) * sigma_scale,
        )
    elif variant == "hypersphere_art":
        return HypersphereART(
            rho=rho,
            alpha=variant_cfg.get("alpha", 0.001),
            beta=variant_cfg.get("beta", 1.0),
            r_hat=variant_cfg.get("r_hat", 0.5),
        )
    else:
        raise ValueError(f"Unknown ART variant: {variant}")


def run_art(
    X: np.ndarray,
    variant: str,
    rho: float,
    shuffle_order: Optional[np.ndarray] = None,
    max_iter: int = 1,
    model_params: Optional[dict] = None,
) -> ClusteringResult:
    """Run a single ART clustering.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features), values in [0, 1].
    variant : str
        ART variant name.
    rho : float
        Vigilance parameter.
    shuffle_order : np.ndarray, optional
        Permutation of sample indices. If None, uses original order.
    max_iter : int
        Number of passes over the data.

    Returns
    -------
    ClusteringResult
        Clustering outcome.
    """
    n_features = X.shape[1]
    model = create_art_model(variant, rho, n_features, model_params=model_params)

    # Apply ordering
    if shuffle_order is not None:
        X_ordered = X[shuffle_order]
    else:
        X_ordered = X

    # Prepare data (complement coding for FuzzyART, normalization for others)
    X_prepared = model.prepare_data(X_ordered)

    # Fit
    model.fit(X_prepared, max_iter=max_iter)

    # Get labels in original order
    labels = model.labels_
    if shuffle_order is not None:
        # Map back to original order
        labels_original = np.empty_like(labels)
        labels_original[shuffle_order] = labels
        labels = labels_original

    centers = model.get_cluster_centers()

    result = ClusteringResult(
        variant=variant,
        rho=rho,
        n_clusters=model.n_clusters,
        labels=labels,
        centers=centers,
        model=model,
        metadata={"max_iter": max_iter, "shuffled": shuffle_order is not None},
    )

    logger.debug(f"{variant} rho={rho}: {result.n_clusters} clusters")
    return result
