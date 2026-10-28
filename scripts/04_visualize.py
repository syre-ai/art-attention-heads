#!/usr/bin/env python3
"""Step 4: Generate all figures.

Usage:
    python scripts/04_visualize.py [--tag base] [--rho 0.5]
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config, get_results_dir
from src.extraction.cache_manager import load_features
from src.clustering.art_runner import run_art
from src.visualization.feature_space import plot_pca, plot_umap
from src.visualization.vigilance_curve import plot_vigilance_curve, plot_ordering_sensitivity
from src.visualization.layer_head_grid import plot_layer_head_grid, plot_multi_variant_grid
from src.visualization.prototype_analysis import plot_radar_chart, prototype_summary_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations")
    parser.add_argument("--tag", default="base", help="Feature cache tag")
    parser.add_argument("--rho", type=float, default=0.5, help="Vigilance for detailed plots")
    args = parser.parse_args()

    model_cfg = load_config("model")
    clustering_cfg = load_config("clustering")
    n_layers = model_cfg["n_layers"]
    n_heads = model_cfg["n_heads"]

    # Load features
    data = load_features(args.tag)
    X = data["aggregated"]
    logger.info(f"Loaded features: {X.shape}")

    # Load sweep results if available
    sweep_path = get_results_dir("clustering") / f"sweep_results_{args.tag}.pkl"
    sweep_results = None
    if sweep_path.exists():
        with open(sweep_path, "rb") as f:
            sweep_results = pickle.load(f)
        logger.info("Loaded sweep results")

    # 1. Vigilance curves
    if sweep_results:
        print("Generating vigilance curves...")
        plot_vigilance_curve(sweep_results)
        plot_ordering_sensitivity(sweep_results)

    # 2. Per-variant detailed plots at specified rho
    variants = list(clustering_cfg["variants"].keys())
    all_labels = {}

    for variant in variants:
        print(f"Generating plots for {variant} at rho={args.rho}...")
        result = run_art(X, variant, args.rho)
        all_labels[variant] = result.labels

        # PCA
        plot_pca(X, result.labels, variant=variant, rho=args.rho,
                 n_layers=n_layers, n_heads=n_heads)

        # UMAP
        plot_umap(X, result.labels, variant=variant, rho=args.rho,
                  n_layers=n_layers, n_heads=n_heads)

        # Layer-head grid
        plot_layer_head_grid(result.labels, n_layers=n_layers, n_heads=n_heads,
                            variant=variant, rho=args.rho)

        # Prototype radar charts
        plot_radar_chart(result.centers, result.labels, variant=variant, rho=args.rho)

        # Print prototype summary
        print(f"\n{variant} Prototype Summary (rho={args.rho}):")
        print(prototype_summary_table(result.centers, result.labels))
        print()

    # 3. Multi-variant comparison grid
    if len(all_labels) > 1:
        print("Generating multi-variant comparison grid...")
        plot_multi_variant_grid(all_labels, n_layers=n_layers, n_heads=n_heads, rho=args.rho)

    # 4. Feature-only PCA (no clustering, just structure)
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    # Color by layer
    layers = np.repeat(np.arange(n_layers), n_heads)
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=layers, cmap="viridis",
                         s=30, alpha=0.7, edgecolors="k", linewidth=0.3)
    plt.colorbar(scatter, ax=ax, label="Layer")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("Feature Space PCA (colored by layer)")
    plt.tight_layout()
    path = get_results_dir("figures") / "pca_by_layer.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"All visualizations saved to {get_results_dir('figures')}")
    print(f"\nAll figures saved to {get_results_dir('figures')}")


if __name__ == "__main__":
    main()
