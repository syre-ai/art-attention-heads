#!/usr/bin/env python3
"""E3: K-Means and Agglomerative baselines for comparison with ART.

Runs standard clustering algorithms on the same 256x14 feature matrix.
Both are order-independent, setting the stability floor that ART
(with consensus) needs to match.

Usage:
    python scripts/06_baselines.py [--tag base] [--k 3 8 15 24]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import get_results_dir
from src.utils.reproducibility import set_seed
from src.extraction.cache_manager import load_features
from src.validation.known_heads import classify_heads_heuristic, compute_heuristic_agreement

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_kmeans(X, k, n_init=50, seed=42):
    """Run K-Means clustering with stability measurement."""
    # Primary run
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed, max_iter=300)
    labels = km.fit_predict(X)
    inertia = km.inertia_

    # Stability: run multiple times with different seeds, measure pairwise ARI
    stability_labels = []
    for i in range(10):
        km_i = KMeans(n_clusters=k, n_init=10, random_state=seed + i + 100)
        stability_labels.append(km_i.fit_predict(X))

    pairwise_aris = []
    for i in range(len(stability_labels)):
        for j in range(i + 1, len(stability_labels)):
            pairwise_aris.append(adjusted_rand_score(stability_labels[i], stability_labels[j]))

    silhouette = silhouette_score(X, labels) if k > 1 and k < len(X) else 0.0

    return {
        "labels": labels,
        "n_clusters": k,
        "inertia": float(inertia),
        "silhouette": float(silhouette),
        "pairwise_ari_mean": float(np.mean(pairwise_aris)),
        "pairwise_ari_std": float(np.std(pairwise_aris)),
        "centroids": km.cluster_centers_,
    }


def run_agglomerative(X, k, linkage="ward"):
    """Run agglomerative clustering."""
    agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = agg.fit_predict(X)

    silhouette = silhouette_score(X, labels) if k > 1 and k < len(X) else 0.0

    # Agglomerative is deterministic, so stability = 1.0
    return {
        "labels": labels,
        "n_clusters": k,
        "linkage": linkage,
        "silhouette": float(silhouette),
        "pairwise_ari_mean": 1.0,  # deterministic
        "pairwise_ari_std": 0.0,
    }


def bootstrap_stability(X, per_input_features, method, k, n_bootstrap=100, seed=42):
    """Bootstrap stability for any clustering method.

    Resample prompts → recompute aggregated features → re-cluster → measure ARI vs full-data clustering.
    """
    rng = np.random.default_rng(seed)
    n_prompts = per_input_features.shape[0]
    n_layers = per_input_features.shape[1]
    n_heads = per_input_features.shape[2]

    # Reference clustering
    if method == "kmeans":
        ref_labels = KMeans(n_clusters=k, n_init=50, random_state=seed).fit_predict(X)
    else:
        ref_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)

    aris = []
    for i in range(n_bootstrap):
        sample_idx = rng.choice(n_prompts, size=n_prompts, replace=True)
        sampled = per_input_features[sample_idx]
        agg = np.mean(sampled, axis=0).reshape(n_layers * n_heads, -1)

        # Normalize
        for f in range(agg.shape[1]):
            col = agg[:, f]
            fmin, fmax = col.min(), col.max()
            if fmax - fmin > 1e-8:
                agg[:, f] = (col - fmin) / (fmax - fmin)
            else:
                agg[:, f] = 0.5

        if method == "kmeans":
            boot_labels = KMeans(n_clusters=k, n_init=10, random_state=seed + i).fit_predict(agg)
        else:
            boot_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(agg)

        aris.append(adjusted_rand_score(ref_labels, boot_labels))

    aris = np.array(aris)
    return {"mean_ari": float(aris.mean()), "std_ari": float(aris.std())}


def main():
    parser = argparse.ArgumentParser(description="E3: Baseline clustering comparison")
    parser.add_argument("--tag", default="base", help="Feature tag to load")
    parser.add_argument("--k", type=int, nargs="+", default=[3, 8, 15, 24],
                        help="Number of clusters to test")
    parser.add_argument("--n-bootstrap", type=int, default=100,
                        help="Number of bootstrap samples for stability")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load features
    features = load_features(args.tag)
    X = features["aggregated"]
    per_input = features["per_input"]
    logger.info(f"Loaded features: aggregated={X.shape}, per_input={per_input.shape}")

    # Heuristic labels
    heuristic_labels, heuristic_groups = classify_heads_heuristic(X)
    logger.info(f"Heuristic classification: {dict((k, len(v)) for k, v in heuristic_groups.items())}")

    out_dir = get_results_dir("baselines")
    all_results = {}

    for k in args.k:
        logger.info(f"\n{'='*60}")
        logger.info(f"E3: Baselines at k={k}")
        logger.info(f"{'='*60}")

        results_k = {}

        # --- K-Means ---
        logger.info(f"\nK-Means (k={k})...")
        km_result = run_kmeans(X, k, seed=args.seed)
        km_heuristic = compute_heuristic_agreement(km_result["labels"], heuristic_labels)
        km_bootstrap = bootstrap_stability(X, per_input, "kmeans", k,
                                           n_bootstrap=args.n_bootstrap, seed=args.seed)

        results_k["kmeans"] = {
            "n_clusters": k,
            "silhouette": km_result["silhouette"],
            "inertia": km_result["inertia"],
            "heuristic_ari": km_heuristic["ari"],
            "heuristic_nmi": km_heuristic["nmi"],
            "ordering_pairwise_ari": km_result["pairwise_ari_mean"],
            "ordering_pairwise_ari_std": km_result["pairwise_ari_std"],
            "bootstrap_ari": km_bootstrap["mean_ari"],
            "bootstrap_ari_std": km_bootstrap["std_ari"],
            "cluster_sizes": {int(l): int((km_result["labels"] == l).sum())
                              for l in np.unique(km_result["labels"])},
        }

        np.save(out_dir / f"kmeans_labels_k{k}.npy", km_result["labels"])
        np.save(out_dir / f"kmeans_centroids_k{k}.npy", km_result["centroids"])

        logger.info(f"  Silhouette: {km_result['silhouette']:.3f}")
        logger.info(f"  Heuristic ARI: {km_heuristic['ari']:.3f}")
        logger.info(f"  Ordering stability: {km_result['pairwise_ari_mean']:.3f}")
        logger.info(f"  Bootstrap ARI: {km_bootstrap['mean_ari']:.3f}")
        logger.info(f"  Sizes: {results_k['kmeans']['cluster_sizes']}")

        # --- Agglomerative (Ward) ---
        logger.info(f"\nAgglomerative Ward (k={k})...")
        agg_result = run_agglomerative(X, k, linkage="ward")
        agg_heuristic = compute_heuristic_agreement(agg_result["labels"], heuristic_labels)
        agg_bootstrap = bootstrap_stability(X, per_input, "agglomerative", k,
                                            n_bootstrap=args.n_bootstrap, seed=args.seed)

        results_k["agglomerative_ward"] = {
            "n_clusters": k,
            "silhouette": agg_result["silhouette"],
            "heuristic_ari": agg_heuristic["ari"],
            "heuristic_nmi": agg_heuristic["nmi"],
            "ordering_pairwise_ari": 1.0,  # deterministic
            "bootstrap_ari": agg_bootstrap["mean_ari"],
            "bootstrap_ari_std": agg_bootstrap["std_ari"],
            "cluster_sizes": {int(l): int((agg_result["labels"] == l).sum())
                              for l in np.unique(agg_result["labels"])},
        }

        np.save(out_dir / f"agglomerative_labels_k{k}.npy", agg_result["labels"])

        logger.info(f"  Silhouette: {agg_result['silhouette']:.3f}")
        logger.info(f"  Heuristic ARI: {agg_heuristic['ari']:.3f}")
        logger.info(f"  Bootstrap ARI: {agg_bootstrap['mean_ari']:.3f}")
        logger.info(f"  Sizes: {results_k['agglomerative_ward']['cluster_sizes']}")

        # --- Cross-method agreement ---
        km_vs_agg = adjusted_rand_score(km_result["labels"], agg_result["labels"])
        results_k["cross_method_ari"] = float(km_vs_agg)
        logger.info(f"\n  K-Means vs Agglomerative ARI: {km_vs_agg:.3f}")

        all_results[f"k_{k}"] = results_k

    # Save all results
    results_path = out_dir / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nAll results saved to {results_path}")

    # Final summary table
    logger.info(f"\n{'='*80}")
    logger.info(f"BASELINE COMPARISON TABLE")
    logger.info(f"{'='*80}")
    logger.info(f"{'Method':>20} | {'k':>3} | {'Silh':>6} | {'H-ARI':>6} | {'Ord-ARI':>7} | {'B-ARI':>6}")
    logger.info(f"{'-'*20}-+-{'-'*3}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}")
    for k_key, k_results in all_results.items():
        for method, r in k_results.items():
            if method == "cross_method_ari":
                continue
            logger.info(
                f"{method:>20} | {r['n_clusters']:>3} | "
                f"{r['silhouette']:>6.3f} | "
                f"{r['heuristic_ari']:>6.3f} | "
                f"{r['ordering_pairwise_ari']:>7.3f} | "
                f"{r['bootstrap_ari']:>6.3f}"
            )
    logger.info(f"\nFor comparison — Round 1 FuzzyART single-run at rho=0.3:")
    logger.info(f"  Heuristic ARI=0.535, Bootstrap ARI=0.684, Ordering ARI=0.080")


if __name__ == "__main__":
    main()
