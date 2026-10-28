"""Radar charts of ART category prototypes."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..extraction.feature_computer import FEATURE_NAMES
from ..utils.config_loader import get_results_dir

logger = logging.getLogger(__name__)


def plot_radar_chart(
    centers: list[np.ndarray],
    labels: np.ndarray,
    feature_names: list[str] | None = None,
    variant: str = "",
    rho: float = 0.0,
    max_categories: int = 12,
    save: bool = True,
) -> Path | None:
    """Plot radar charts for ART category prototypes.

    Parameters
    ----------
    centers : list[np.ndarray]
        Cluster centers, each of shape (n_features,).
    labels : np.ndarray
        Cluster labels (to compute category sizes).
    feature_names : list[str], optional
        Names for each feature dimension.
    variant : str
        ART variant name.
    rho : float
        Vigilance parameter.
    max_categories : int
        Maximum categories to show (largest by member count).
    save : bool
        Whether to save.

    Returns
    -------
    Path or None
    """
    feature_names = feature_names or FEATURE_NAMES
    n_features = len(feature_names)

    # Count members per category and pick top ones
    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_idx = np.argsort(-counts)
    top_cats = sorted_idx[:max_categories]

    n_cats = len(top_cats)
    if n_cats == 0:
        logger.warning("No categories to plot")
        return None

    # Radar chart setup
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    cols = min(4, n_cats)
    rows = (n_cats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows),
                              subplot_kw=dict(polar=True))
    if n_cats == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    for idx, cat_idx in enumerate(top_cats):
        row, col = divmod(idx, cols)
        ax = axes[row, col]

        cat_label = unique_labels[cat_idx]
        center = np.array(centers[cat_label])
        values = center.tolist() + center[:1].tolist()

        ax.plot(angles, values, "o-", linewidth=2, markersize=4)
        ax.fill(angles, values, alpha=0.2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([n[:6] for n in feature_names], fontsize=6)
        ax.set_ylim(0, 1)
        ax.set_title(f"Cat {cat_label} (n={counts[cat_idx]})", fontsize=10)

    # Hide unused axes
    for idx in range(n_cats, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"Category Prototypes: {variant} rho={rho}\n({n_cats} largest categories)",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()

    if save:
        path = get_results_dir("figures") / f"prototypes_{variant}_rho{rho:.2f}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved prototype radar chart to {path}")
        return path
    else:
        plt.show()
        return None


def prototype_summary_table(
    centers: list[np.ndarray],
    labels: np.ndarray,
    feature_names: list[str] | None = None,
) -> str:
    """Generate a text summary of category prototypes.

    Returns
    -------
    str
        Formatted table of prototype feature values.
    """
    feature_names = feature_names or FEATURE_NAMES
    unique_labels, counts = np.unique(labels, return_counts=True)

    lines = []
    header = f"{'Cat':>4} {'Size':>5} " + " ".join(f"{n[:8]:>8}" for n in feature_names)
    lines.append(header)
    lines.append("-" * len(header))

    for label, count in zip(unique_labels, counts):
        center = np.array(centers[label])
        vals = " ".join(f"{v:8.3f}" for v in center)
        lines.append(f"{label:4d} {count:5d} {vals}")

    return "\n".join(lines)
