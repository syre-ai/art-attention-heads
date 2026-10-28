"""Stability validation: bootstrap resampling, split-half, ordering sensitivity, feature ablation."""

import logging

import numpy as np
from sklearn.metrics import adjusted_rand_score

from ..clustering.art_runner import run_art
from ..utils.reproducibility import get_rng

logger = logging.getLogger(__name__)


def bootstrap_stability(
    X: np.ndarray,
    per_input_features: np.ndarray,
    variant: str,
    rho: float,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap resampling stability.

    Resample prompts → recompute aggregated features → re-cluster → measure ARI.

    Parameters
    ----------
    X : np.ndarray
        Aggregated feature matrix (n_heads, n_features).
    per_input_features : np.ndarray
        Per-input features (n_prompts, n_layers, n_heads, n_features).
    variant : str
        ART variant.
    rho : float
        Vigilance parameter.
    n_bootstrap : int
        Number of bootstrap samples.
    seed : int
        Random seed.

    Returns
    -------
    dict with "mean_ari", "std_ari", "min_ari", "max_ari".
    """
    rng = get_rng(seed)
    n_prompts = per_input_features.shape[0]
    n_layers = per_input_features.shape[1]
    n_heads = per_input_features.shape[2]

    # Reference clustering on full data
    ref_result = run_art(X, variant, rho)
    ref_labels = ref_result.labels

    aris = []
    for i in range(n_bootstrap):
        # Bootstrap sample of prompts
        sample_idx = rng.choice(n_prompts, size=n_prompts, replace=True)
        sampled = per_input_features[sample_idx]

        # Re-aggregate
        agg = np.mean(sampled, axis=0).reshape(n_layers * n_heads, -1)

        # Normalize per feature
        for f in range(agg.shape[1]):
            col = agg[:, f]
            fmin, fmax = col.min(), col.max()
            if fmax - fmin > 1e-8:
                agg[:, f] = (col - fmin) / (fmax - fmin)
            else:
                agg[:, f] = 0.5

        # Re-cluster
        result = run_art(agg, variant, rho)
        ari = adjusted_rand_score(ref_labels, result.labels)
        aris.append(ari)

    aris = np.array(aris)
    summary = {
        "mean_ari": float(aris.mean()),
        "std_ari": float(aris.std()),
        "min_ari": float(aris.min()),
        "max_ari": float(aris.max()),
    }
    logger.info(f"Bootstrap stability ({variant}, rho={rho}): ARI={summary['mean_ari']:.3f}±{summary['std_ari']:.3f}")
    return summary


def split_half_stability(
    per_input_features: np.ndarray,
    variant: str,
    rho: float,
    n_splits: int = 10,
    seed: int = 42,
) -> dict[str, float]:
    """Split-half stability: split prompts, cluster each half, measure ARI.

    Parameters
    ----------
    per_input_features : np.ndarray
        Shape (n_prompts, n_layers, n_heads, n_features).
    variant : str
        ART variant.
    rho : float
        Vigilance parameter.
    n_splits : int
        Number of random splits.
    seed : int
        Random seed.

    Returns
    -------
    dict with "mean_ari", "std_ari".
    """
    rng = get_rng(seed)
    n_prompts = per_input_features.shape[0]
    n_layers = per_input_features.shape[1]
    n_heads = per_input_features.shape[2]

    aris = []
    for _ in range(n_splits):
        perm = rng.permutation(n_prompts)
        half = n_prompts // 2
        idx_a, idx_b = perm[:half], perm[half:2 * half]

        def _aggregate(idx):
            agg = np.mean(per_input_features[idx], axis=0).reshape(n_layers * n_heads, -1)
            for f in range(agg.shape[1]):
                col = agg[:, f]
                fmin, fmax = col.min(), col.max()
                if fmax - fmin > 1e-8:
                    agg[:, f] = (col - fmin) / (fmax - fmin)
                else:
                    agg[:, f] = 0.5
            return agg

        agg_a = _aggregate(idx_a)
        agg_b = _aggregate(idx_b)

        result_a = run_art(agg_a, variant, rho)
        result_b = run_art(agg_b, variant, rho)
        ari = adjusted_rand_score(result_a.labels, result_b.labels)
        aris.append(ari)

    aris = np.array(aris)
    summary = {"mean_ari": float(aris.mean()), "std_ari": float(aris.std())}
    logger.info(f"Split-half stability ({variant}, rho={rho}): ARI={summary['mean_ari']:.3f}±{summary['std_ari']:.3f}")
    return summary


def ordering_sensitivity(
    X: np.ndarray,
    variant: str,
    rho: float,
    n_runs: int = 10,
    seed: int = 42,
) -> dict[str, float]:
    """Measure sensitivity to presentation order.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    variant : str
        ART variant.
    rho : float
        Vigilance parameter.
    n_runs : int
        Number of shuffled runs.
    seed : int
        Random seed.

    Returns
    -------
    dict with "n_clusters_mean", "n_clusters_std", "cv" (coefficient of variation),
    "pairwise_ari_mean".
    """
    rng = get_rng(seed)
    all_labels = []
    all_n_clusters = []

    for _ in range(n_runs):
        shuffle = rng.permutation(len(X))
        result = run_art(X, variant, rho, shuffle_order=shuffle)
        all_labels.append(result.labels)
        all_n_clusters.append(result.n_clusters)

    all_n_clusters = np.array(all_n_clusters)

    # Pairwise ARI
    aris = []
    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            aris.append(adjusted_rand_score(all_labels[i], all_labels[j]))

    mean_nc = float(all_n_clusters.mean())
    std_nc = float(all_n_clusters.std())
    cv = std_nc / mean_nc if mean_nc > 0 else 0.0

    summary = {
        "n_clusters_mean": mean_nc,
        "n_clusters_std": std_nc,
        "cv": cv,
        "pairwise_ari_mean": float(np.mean(aris)) if aris else 0.0,
    }
    logger.info(
        f"Ordering sensitivity ({variant}, rho={rho}): "
        f"n_clusters={mean_nc:.1f}±{std_nc:.1f} (CV={cv:.3f}), "
        f"pairwise ARI={summary['pairwise_ari_mean']:.3f}"
    )
    return summary


def feature_ablation(
    X: np.ndarray,
    variant: str,
    rho: float,
    feature_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Remove one feature at a time and measure ARI drop.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    variant : str
        ART variant.
    rho : float
        Vigilance parameter.
    feature_names : list[str], optional
        Names for each feature column.

    Returns
    -------
    dict mapping feature name/index to {"ari_drop": float, "n_clusters": int}.
    """
    from ..extraction.feature_computer import FEATURE_NAMES

    feature_names = feature_names or FEATURE_NAMES
    n_features = X.shape[1]

    # Reference clustering
    ref_result = run_art(X, variant, rho)
    ref_labels = ref_result.labels

    results = {}
    for f_idx in range(n_features):
        # Remove feature f_idx
        X_ablated = np.delete(X, f_idx, axis=1)

        # Re-normalize ablated features
        for col_idx in range(X_ablated.shape[1]):
            col = X_ablated[:, col_idx]
            fmin, fmax = col.min(), col.max()
            if fmax - fmin > 1e-8:
                X_ablated[:, col_idx] = (col - fmin) / (fmax - fmin)
            else:
                X_ablated[:, col_idx] = 0.5

        result = run_art(X_ablated, variant, rho)
        ari = adjusted_rand_score(ref_labels, result.labels)

        name = feature_names[f_idx] if f_idx < len(feature_names) else str(f_idx)
        results[name] = {
            "ari_vs_full": float(ari),
            "ari_drop": float(1.0 - ari),
            "n_clusters": result.n_clusters,
            "n_clusters_diff": result.n_clusters - ref_result.n_clusters,
        }
        logger.info(f"  Ablate {name}: ARI={ari:.3f}, n_clusters={result.n_clusters}")

    return results
