"""PCA/UMAP of feature space colored by ART category."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from ..utils.config_loader import get_results_dir

logger = logging.getLogger(__name__)


def plot_pca(
    features: np.ndarray,
    labels: np.ndarray,
    title: str = "PCA of Attention Head Features",
    variant: str = "",
    rho: float = 0.0,
    save: bool = True,
    n_layers: int = 16,
    n_heads: int = 16,
) -> Path | None:
    """Plot PCA of feature space colored by ART cluster labels.

    Parameters
    ----------
    features : np.ndarray
        Shape (n_heads_total, n_features).
    labels : np.ndarray
        Cluster labels, shape (n_heads_total,).
    title : str
        Plot title.
    variant : str
        ART variant name (for filename).
    rho : float
        Vigilance value (for filename).
    save : bool
        Whether to save the figure.
    n_layers, n_heads : int
        For annotation.

    Returns
    -------
    Path or None
        Path to saved figure.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    n_clusters = len(np.unique(labels))
    cmap = plt.cm.get_cmap("tab20", max(n_clusters, 2))

    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=labels, cmap=cmap, s=30, alpha=0.7, edgecolors="k", linewidth=0.3,
    )

    # Annotate with layer.head
    for idx in range(len(features)):
        layer = idx // n_heads
        head = idx % n_heads
        ax.annotate(
            f"{layer}.{head}", (X_pca[idx, 0], X_pca[idx, 1]),
            fontsize=4, alpha=0.5, ha="center", va="bottom",
        )

    plt.colorbar(scatter, ax=ax, label="ART Category")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(f"{title}\n{variant} rho={rho}, {n_clusters} clusters")

    plt.tight_layout()

    if save:
        out_dir = get_results_dir("figures")
        fname = f"pca_{variant}_rho{rho:.2f}.png"
        path = out_dir / fname
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved PCA plot to {path}")
        return path
    else:
        plt.show()
        return None


def plot_umap(
    features: np.ndarray,
    labels: np.ndarray,
    title: str = "UMAP of Attention Head Features",
    variant: str = "",
    rho: float = 0.0,
    save: bool = True,
    n_layers: int = 16,
    n_heads: int = 16,
) -> Path | None:
    """Plot UMAP of feature space colored by ART cluster labels.

    Falls back to PCA if umap-learn is not installed.
    """
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15)
        X_2d = reducer.fit_transform(features)
        method = "UMAP"
    except ImportError:
        logger.warning("umap-learn not installed, falling back to PCA")
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(features)
        method = "PCA (umap unavailable)"

    fig, ax = plt.subplots(figsize=(10, 8))
    n_clusters = len(np.unique(labels))
    cmap = plt.cm.get_cmap("tab20", max(n_clusters, 2))

    scatter = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=labels, cmap=cmap, s=30, alpha=0.7, edgecolors="k", linewidth=0.3,
    )

    plt.colorbar(scatter, ax=ax, label="ART Category")
    ax.set_xlabel(f"{method} dim 1")
    ax.set_ylabel(f"{method} dim 2")
    ax.set_title(f"{title}\n{variant} rho={rho}, {n_clusters} clusters")

    plt.tight_layout()

    if save:
        out_dir = get_results_dir("figures")
        fname = f"umap_{variant}_rho{rho:.2f}.png"
        path = out_dir / fname
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved UMAP plot to {path}")
        return path
    else:
        plt.show()
        return None
