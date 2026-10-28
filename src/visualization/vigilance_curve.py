"""Plots of n_clusters vs vigilance parameter rho."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..clustering.vigilance_sweep import SweepResult
from ..utils.config_loader import get_results_dir

logger = logging.getLogger(__name__)

VARIANT_STYLES = {
    "fuzzy_art": {"color": "#1f77b4", "marker": "o", "label": "FuzzyART"},
    "gaussian_art": {"color": "#ff7f0e", "marker": "s", "label": "GaussianART"},
    "hypersphere_art": {"color": "#2ca02c", "marker": "^", "label": "HypersphereART"},
}


def plot_vigilance_curve(
    sweep_results: dict[str, SweepResult],
    save: bool = True,
) -> Path | None:
    """Plot n_clusters vs rho for all variants with ordering sensitivity bands.

    Parameters
    ----------
    sweep_results : dict[str, SweepResult]
        Mapping variant name to sweep results.
    save : bool
        Whether to save the figure.

    Returns
    -------
    Path or None
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for variant, sweep in sweep_results.items():
        style = VARIANT_STYLES.get(variant, {"color": "gray", "marker": "x", "label": variant})
        rhos = np.array(sweep.rho_values)
        n_clusters = np.array(sweep.n_clusters)

        # Main line
        ax.plot(rhos, n_clusters, color=style["color"], marker=style["marker"],
                label=style["label"], linewidth=2, markersize=6)

        # Ordering sensitivity band
        if sweep.ordering_runs:
            lows, highs = [], []
            for rho in sweep.rho_values:
                runs = sweep.ordering_runs.get(rho, [])
                if runs:
                    lows.append(min(runs))
                    highs.append(max(runs))
                else:
                    lows.append(n_clusters[list(sweep.rho_values).index(rho)])
                    highs.append(n_clusters[list(sweep.rho_values).index(rho)])

            ax.fill_between(rhos, lows, highs, color=style["color"], alpha=0.15)

    ax.set_xlabel("Vigilance parameter (rho)", fontsize=12)
    ax.set_ylabel("Number of clusters", fontsize=12)
    ax.set_title("ART Vigilance Sweep: Number of Clusters vs Vigilance", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(np.arange(0, 1.05, 0.1))

    plt.tight_layout()

    if save:
        path = get_results_dir("figures") / "vigilance_curve.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved vigilance curve to {path}")
        return path
    else:
        plt.show()
        return None


def plot_ordering_sensitivity(
    sweep_results: dict[str, SweepResult],
    save: bool = True,
) -> Path | None:
    """Plot coefficient of variation of n_clusters across orderings.

    Parameters
    ----------
    sweep_results : dict[str, SweepResult]
        Mapping variant name to sweep results.
    save : bool
        Whether to save the figure.

    Returns
    -------
    Path or None
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for variant, sweep in sweep_results.items():
        style = VARIANT_STYLES.get(variant, {"color": "gray", "marker": "x", "label": variant})
        rhos = []
        cvs = []
        for rho in sweep.rho_values:
            runs = sweep.ordering_runs.get(rho, [])
            if runs:
                mean_n = np.mean(runs)
                std_n = np.std(runs)
                cv = std_n / mean_n if mean_n > 0 else 0
                rhos.append(rho)
                cvs.append(cv)

        ax.plot(rhos, cvs, color=style["color"], marker=style["marker"],
                label=style["label"], linewidth=2, markersize=6)

    ax.axhline(y=0.2, color="red", linestyle="--", alpha=0.5, label="CV threshold (0.2)")
    ax.set_xlabel("Vigilance parameter (rho)", fontsize=12)
    ax.set_ylabel("Coefficient of Variation (n_clusters)", fontsize=12)
    ax.set_title("Ordering Sensitivity: CV of Cluster Count", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        path = get_results_dir("figures") / "ordering_sensitivity.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved ordering sensitivity plot to {path}")
        return path
    else:
        plt.show()
        return None
