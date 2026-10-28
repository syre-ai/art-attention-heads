"""Sweep vigilance parameter rho, record n_clusters, store fitted models."""

import logging
from dataclasses import dataclass, field

import numpy as np

from ..utils.config_loader import load_config
from ..utils.reproducibility import get_rng
from .art_runner import ClusteringResult, run_art

logger = logging.getLogger(__name__)


@dataclass
class SweepResult:
    """Result of a full vigilance sweep for one ART variant."""
    variant: str
    rho_values: list[float]
    n_clusters: list[int]  # Primary run n_clusters per rho
    results: list[ClusteringResult]  # Primary run results per rho
    ordering_runs: dict[float, list[int]] = field(default_factory=dict)
    # rho -> list of n_clusters across ordering shuffles


def sweep_vigilance(
    X: np.ndarray,
    variant: str,
    rho_values: list[float] | None = None,
    n_ordering_runs: int | None = None,
    seed: int = 42,
) -> SweepResult:
    """Run vigilance sweep for one ART variant.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features), values in [0, 1].
    variant : str
        ART variant name.
    rho_values : list[float], optional
        Vigilance values to sweep. Defaults to config.
    n_ordering_runs : int, optional
        Number of shuffled re-runs per rho. Defaults to config.
    seed : int
        Random seed.

    Returns
    -------
    SweepResult
        Full sweep results.
    """
    cfg = load_config("clustering")
    rho_values = rho_values or cfg["vigilance_sweep"]["rho_values"]
    n_ordering_runs = n_ordering_runs if n_ordering_runs is not None else cfg["vigilance_sweep"]["n_ordering_runs"]
    max_iter = cfg["fit_params"]["max_iter"]
    rng = get_rng(seed)

    primary_results = []
    primary_n_clusters = []
    ordering_runs: dict[float, list[int]] = {}

    for rho in rho_values:
        logger.info(f"Sweep {variant} rho={rho}")

        # Primary run (deterministic order)
        result = run_art(X, variant, rho, max_iter=max_iter)
        primary_results.append(result)
        primary_n_clusters.append(result.n_clusters)

        # Ordering sensitivity runs
        ordering_n = []
        for run_idx in range(n_ordering_runs):
            shuffle = rng.permutation(len(X))
            r = run_art(X, variant, rho, shuffle_order=shuffle, max_iter=max_iter)
            ordering_n.append(r.n_clusters)

        ordering_runs[rho] = ordering_n
        logger.info(
            f"  rho={rho}: primary={result.n_clusters}, "
            f"ordering range=[{min(ordering_n)}, {max(ordering_n)}]"
        )

    return SweepResult(
        variant=variant,
        rho_values=rho_values,
        n_clusters=primary_n_clusters,
        results=primary_results,
        ordering_runs=ordering_runs,
    )


def sweep_all_variants(
    X: np.ndarray,
    seed: int = 42,
) -> dict[str, SweepResult]:
    """Run vigilance sweep for all 3 ART variants.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features), values in [0, 1].
    seed : int
        Random seed.

    Returns
    -------
    dict[str, SweepResult]
        Mapping variant name to sweep result.
    """
    cfg = load_config("clustering")
    results = {}
    for variant in cfg["variants"]:
        logger.info(f"=== Sweeping {variant} ===")
        results[variant] = sweep_vigilance(X, variant, seed=seed)
    return results
