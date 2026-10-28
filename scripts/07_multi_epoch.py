#!/usr/bin/env python3
"""E2: Multi-epoch ART ablation — does multiple passes stabilize clustering?

Tests whether running ART with max_iter > 1 reduces ordering sensitivity.
Round 1 used max_iter=1 (single pass). Multiple passes refine prototypes
which may reduce dependence on initial presentation order.

Usage:
    python scripts/07_multi_epoch.py [--tag base] [--rho 0.5]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import get_results_dir
from src.utils.reproducibility import set_seed, get_rng
from src.extraction.cache_manager import load_features
from src.clustering.art_runner import run_art
from src.validation.known_heads import classify_heads_heuristic, compute_heuristic_agreement

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def measure_ordering_sensitivity(X, variant, rho, max_iter, n_runs=10, seed=42):
    """Run ART n_runs times with different orderings, measure stability."""
    rng = get_rng(seed)
    all_labels = []
    all_n_clusters = []

    for i in range(n_runs):
        shuffle = rng.permutation(len(X))
        result = run_art(X, variant, rho, shuffle_order=shuffle, max_iter=max_iter)
        all_labels.append(result.labels)
        all_n_clusters.append(result.n_clusters)

    # Pairwise ARI
    pairwise_aris = []
    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            pairwise_aris.append(adjusted_rand_score(all_labels[i], all_labels[j]))

    n_clusters_arr = np.array(all_n_clusters)
    mean_nc = float(n_clusters_arr.mean())
    cv = float(n_clusters_arr.std() / mean_nc) if mean_nc > 0 else 0.0

    return {
        "n_clusters_mean": mean_nc,
        "n_clusters_std": float(n_clusters_arr.std()),
        "n_clusters_range": [int(n_clusters_arr.min()), int(n_clusters_arr.max())],
        "cv": cv,
        "pairwise_ari_mean": float(np.mean(pairwise_aris)),
        "pairwise_ari_std": float(np.std(pairwise_aris)),
        "pairwise_ari_min": float(np.min(pairwise_aris)),
        "pairwise_ari_max": float(np.max(pairwise_aris)),
    }


def main():
    parser = argparse.ArgumentParser(description="E2: Multi-epoch ART ablation")
    parser.add_argument("--tag", default="base", help="Feature tag to load")
    parser.add_argument("--rho", type=float, nargs="+", default=[0.3, 0.5],
                        help="Vigilance values to test")
    parser.add_argument("--max-iter", type=int, nargs="+", default=[1, 3, 5, 10],
                        help="Number of ART epochs to test")
    parser.add_argument("--variant", default="fuzzy_art", help="ART variant")
    parser.add_argument("--n-ordering-runs", type=int, default=10,
                        help="Number of shuffled runs per setting")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load features
    features = load_features(args.tag)
    X = features["aggregated"]
    logger.info(f"Loaded features: {X.shape}")

    # Heuristic labels
    heuristic_labels, heuristic_groups = classify_heads_heuristic(X)

    out_dir = get_results_dir("multi_epoch")
    all_results = {}

    for rho in args.rho:
        logger.info(f"\n{'='*60}")
        logger.info(f"E2: Multi-epoch ablation — {args.variant} rho={rho}")
        logger.info(f"{'='*60}")

        rho_results = {}

        for max_iter in args.max_iter:
            logger.info(f"\n--- max_iter={max_iter} ---")

            # Ordering sensitivity
            sensitivity = measure_ordering_sensitivity(
                X, args.variant, rho, max_iter,
                n_runs=args.n_ordering_runs, seed=args.seed
            )

            # Heuristic agreement (use first shuffled run)
            rng = get_rng(args.seed)
            shuffle = rng.permutation(len(X))
            result = run_art(X, args.variant, rho, shuffle_order=shuffle, max_iter=max_iter)
            heuristic_ari = compute_heuristic_agreement(result.labels, heuristic_labels)

            rho_results[f"max_iter_{max_iter}"] = {
                "max_iter": max_iter,
                "ordering_sensitivity": sensitivity,
                "heuristic_ari": heuristic_ari["ari"],
                "heuristic_nmi": heuristic_ari["nmi"],
                "n_clusters": result.n_clusters,
            }

            logger.info(f"  n_clusters: {sensitivity['n_clusters_mean']:.1f}±{sensitivity['n_clusters_std']:.1f} "
                         f"(range {sensitivity['n_clusters_range']})")
            logger.info(f"  Ordering pairwise ARI: {sensitivity['pairwise_ari_mean']:.3f}±{sensitivity['pairwise_ari_std']:.3f}")
            logger.info(f"  CV: {sensitivity['cv']:.3f}")
            logger.info(f"  Heuristic ARI: {heuristic_ari['ari']:.3f}")

        all_results[f"rho_{rho}"] = rho_results

    # Save results
    results_path = out_dir / f"multi_epoch_results_{args.variant}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Summary table
    logger.info(f"\n{'='*80}")
    logger.info(f"MULTI-EPOCH ABLATION TABLE")
    logger.info(f"{'='*80}")
    logger.info(f"{'rho':>6} | {'epochs':>6} | {'k':>10} | {'Ord-ARI':>9} | {'CV':>6} | {'H-ARI':>6}")
    logger.info(f"{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*9}-+-{'-'*6}-+-{'-'*6}")
    for rho_key, rho_results in all_results.items():
        for iter_key, r in rho_results.items():
            s = r["ordering_sensitivity"]
            k_str = f"{s['n_clusters_mean']:.0f}±{s['n_clusters_std']:.0f}"
            logger.info(
                f"{float(rho_key.split('_')[1]):>6.2f} | {r['max_iter']:>6} | "
                f"{k_str:>10} | "
                f"{s['pairwise_ari_mean']:>9.3f} | "
                f"{s['cv']:>6.3f} | "
                f"{r['heuristic_ari']:>6.3f}"
            )
    logger.info(f"\nRound 1 reference (max_iter=1, rho=0.3): pairwise ARI=0.080, CV=0.125")


if __name__ == "__main__":
    main()
