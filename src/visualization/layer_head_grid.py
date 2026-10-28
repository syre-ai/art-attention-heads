"""16x16 grid of attention heads colored by ART category."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from ..utils.config_loader import get_results_dir

logger = logging.getLogger(__name__)


def plot_layer_head_grid(
    labels: np.ndarray,
    n_layers: int = 16,
    n_heads: int = 16,
    title: str = "Attention Head Taxonomy",
    variant: str = "",
    rho: float = 0.0,
    save: bool = True,
) -> Path | None:
    """Plot a layer x head grid colored by ART cluster assignment.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels of shape (n_layers * n_heads,).
    n_layers, n_heads : int
        Grid dimensions.
    title : str
        Plot title.
    variant : str
        ART variant name (for filename).
    rho : float
        Vigilance value (for filename).
    save : bool
        Whether to save.

    Returns
    -------
    Path or None
    """
    grid = labels.reshape(n_layers, n_heads)
    n_clusters = len(np.unique(labels))

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.cm.get_cmap("tab20", max(n_clusters, 2))

    im = ax.imshow(grid, cmap=cmap, aspect="equal", interpolation="nearest")

    # Annotate each cell with cluster label
    for i in range(n_layers):
        for j in range(n_heads):
            ax.text(j, i, str(grid[i, j]), ha="center", va="center",
                    fontsize=7, fontweight="bold",
                    color="white" if grid[i, j] > n_clusters / 2 else "black")

    ax.set_xlabel("Head", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_title(f"{title}\n{variant} rho={rho}, {n_clusters} categories", fontsize=14)

    plt.colorbar(im, ax=ax, label="ART Category", shrink=0.8)
    plt.tight_layout()

    if save:
        path = get_results_dir("figures") / f"layer_head_grid_{variant}_rho{rho:.2f}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved layer-head grid to {path}")
        return path
    else:
        plt.show()
        return None


def plot_multi_variant_grid(
    all_labels: dict[str, np.ndarray],
    n_layers: int = 16,
    n_heads: int = 16,
    rho: float = 0.0,
    save: bool = True,
) -> Path | None:
    """Plot side-by-side grids for multiple ART variants.

    Parameters
    ----------
    all_labels : dict[str, np.ndarray]
        Mapping variant name to label arrays.
    """
    n_variants = len(all_labels)
    fig, axes = plt.subplots(1, n_variants, figsize=(6 * n_variants, 10))
    if n_variants == 1:
        axes = [axes]

    for ax, (variant, labels) in zip(axes, all_labels.items()):
        grid = labels.reshape(n_layers, n_heads)
        n_clusters = len(np.unique(labels))
        cmap = plt.cm.get_cmap("tab20", max(n_clusters, 2))

        im = ax.imshow(grid, cmap=cmap, aspect="equal", interpolation="nearest")
        for i in range(n_layers):
            for j in range(n_heads):
                ax.text(j, i, str(grid[i, j]), ha="center", va="center",
                        fontsize=5, fontweight="bold")

        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(f"{variant}\nrho={rho}, {n_clusters} cats")
        plt.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle(f"Cross-Variant Comparison at rho={rho}", fontsize=14, y=1.02)
    plt.tight_layout()

    if save:
        path = get_results_dir("figures") / f"multi_variant_grid_rho{rho:.2f}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved multi-variant grid to {path}")
        return path
    else:
        plt.show()
        return None
