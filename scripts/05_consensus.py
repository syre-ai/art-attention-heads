#!/usr/bin/env python3
"""E1: Consensus ART clustering — fixes ordering sensitivity.

Runs FuzzyART N times with random orderings, builds co-association matrix,
applies agglomerative clustering. Measures whether consensus eliminates
the ordering sensitivity problem (Round 1 pairwise ARI = 0.07-0.11).

Usage:
    python scripts/05_consensus.py [--tag base] [--n-runs 50] [--rho 0.3 0.5 0.7]
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import get_results_dir
from src.utils.reproducibility import set_seed
from src.extraction.cache_manager import load_features
from src.clustering.consensus import (
    consensus_art,
    consensus_stability,
    compare_consensus_vs_single,
)
from src.validation.known_heads import classify_heads_heuristic, compute_heuristic_agreement

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="E1: Consensus ART clustering")
    parser.add_argument("--tag", default="base", help="Feature tag to load")
    parser.add_argument("--n-runs", type=int, default=50, help="Number of ART runs per consensus")
    parser.add_argument("--rho", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                        help="Vigilance values to test")
    parser.add_argument("--linkage", default="average", choices=["average", "complete", "ward"],
                        help="Linkage method for agglomerative clustering")
    parser.add_argument("--n-stability-repeats", type=int, default=5,
                        help="Number of independent consensus runs to measure stability")
    parser.add_argument("--variant", default="fuzzy_art", help="ART variant")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load features
    features = load_features(args.tag)
    X = features["aggregated"]
    logger.info(f"Loaded features: {X.shape}")

    # Heuristic labels for comparison
    heuristic_labels, heuristic_groups = classify_heads_heuristic(X)
    logger.info(f"Heuristic classification: {dict((k, len(v)) for k, v in heuristic_groups.items())}")

    # Output directory
    out_dir = get_results_dir("consensus")
    logger.info(f"Results will be saved to {out_dir}")

    all_results = {}

    for rho in args.rho:
        logger.info(f"\n{'='*60}")
        logger.info(f"E1: Consensus ART — {args.variant} rho={rho}, N={args.n_runs}")
        logger.info(f"{'='*60}")

        # --- Step 1: Run consensus clustering ---
        consensus_result = consensus_art(
            X, args.variant, rho,
            n_runs=args.n_runs,
            linkage_method=args.linkage,
            seed=args.seed,
        )

        # --- Step 2: Heuristic agreement ---
        heuristic_ari = compute_heuristic_agreement(consensus_result.labels, heuristic_labels)
        logger.info(f"Heuristic agreement: ARI={heuristic_ari['ari']:.3f}, NMI={heuristic_ari['nmi']:.3f}")

        # --- Step 3: Consensus stability (repeat consensus with different seeds) ---
        logger.info(f"\nMeasuring consensus stability ({args.n_stability_repeats} repeats)...")
        stability = consensus_stability(
            X, args.variant, rho,
            n_runs=args.n_runs,
            n_repeats=args.n_stability_repeats,
            linkage_method=args.linkage,
            seed=args.seed + 1000,  # different seed for stability test
        )

        # --- Step 4: Compare consensus vs single-run ART ---
        logger.info(f"\nComparing consensus vs single-run ART...")
        comparison = compare_consensus_vs_single(
            X, args.variant, rho,
            consensus_labels=consensus_result.labels,
            n_single_runs=10,
            seed=args.seed + 2000,
        )

        # --- Step 5: Co-association matrix diagnostics ---
        C = consensus_result.coassoc_matrix
        # Block structure: for each consensus cluster, compute within-cluster mean C
        unique_labels = np.unique(consensus_result.labels)
        cluster_cohesion = {}
        for label in unique_labels:
            mask = consensus_result.labels == label
            within = C[np.ix_(mask, mask)]
            # Exclude diagonal
            n = mask.sum()
            if n > 1:
                off_diag = within[np.triu_indices(n, k=1)]
                cluster_cohesion[int(label)] = {
                    "size": int(n),
                    "within_mean": float(off_diag.mean()),
                    "within_min": float(off_diag.min()),
                    "within_std": float(off_diag.std()),
                }
            else:
                cluster_cohesion[int(label)] = {"size": 1, "within_mean": 1.0, "within_min": 1.0, "within_std": 0.0}

        # Between-cluster separation
        between_vals = []
        for i_label in unique_labels:
            for j_label in unique_labels:
                if i_label >= j_label:
                    continue
                mask_i = consensus_result.labels == i_label
                mask_j = consensus_result.labels == j_label
                between = C[np.ix_(mask_i, mask_j)]
                between_vals.extend(between.ravel().tolist())

        between_summary = {
            "between_mean": float(np.mean(between_vals)) if between_vals else 0.0,
            "between_max": float(np.max(between_vals)) if between_vals else 0.0,
        }

        # --- Compile results ---
        rho_results = {
            "rho": rho,
            "variant": args.variant,
            "n_runs": args.n_runs,
            "linkage_method": args.linkage,
            "n_clusters": consensus_result.n_clusters,
            "n_clusters_per_run": {
                "median": int(np.median(consensus_result.per_run_n_clusters)),
                "range": [int(min(consensus_result.per_run_n_clusters)),
                          int(max(consensus_result.per_run_n_clusters))],
                "mean": float(np.mean(consensus_result.per_run_n_clusters)),
            },
            "heuristic": heuristic_ari,
            "consensus_stability": stability,
            "vs_single_run": comparison,
            "cluster_cohesion": cluster_cohesion,
            "between_cluster": between_summary,
            "cluster_sizes": {int(l): int((consensus_result.labels == l).sum()) for l in unique_labels},
        }
        all_results[f"rho_{rho}"] = rho_results

        # Save consensus labels
        np.save(out_dir / f"consensus_labels_{args.variant}_rho{rho:.2f}.npy", consensus_result.labels)

        # Save co-association matrix (diagnostic)
        np.save(out_dir / f"coassoc_matrix_{args.variant}_rho{rho:.2f}.npy", C)

        # Save linkage matrix for dendrogram
        np.save(out_dir / f"linkage_{args.variant}_rho{rho:.2f}.npy", consensus_result.linkage_matrix)

        # --- Print summary ---
        logger.info(f"\n--- SUMMARY rho={rho} ---")
        logger.info(f"Consensus clusters: {consensus_result.n_clusters}")
        logger.info(f"Individual run clusters: median={rho_results['n_clusters_per_run']['median']}, "
                     f"range={rho_results['n_clusters_per_run']['range']}")
        logger.info(f"Heuristic ARI: {heuristic_ari['ari']:.3f} (threshold: >0.3)")
        logger.info(f"Consensus stability pairwise ARI: {stability['pairwise_ari_mean']:.3f} (threshold: >0.9)")
        logger.info(f"Single-run vs consensus ARI: {comparison['single_vs_consensus_mean']:.3f}")
        logger.info(f"Single-run vs single-run ARI: {comparison['single_vs_single_mean']:.3f} (Round 1 baseline)")
        logger.info(f"Cluster sizes: {rho_results['cluster_sizes']}")
        for label, cohesion in cluster_cohesion.items():
            logger.info(f"  Cluster {label}: size={cohesion['size']}, "
                         f"cohesion={cohesion['within_mean']:.3f} (min={cohesion['within_min']:.3f})")
        logger.info(f"Between-cluster co-association: mean={between_summary['between_mean']:.3f}, "
                     f"max={between_summary['between_max']:.3f}")

        # Pass/fail check
        passes = []
        if stability['pairwise_ari_mean'] > 0.9:
            passes.append("consensus_stability PASS")
        else:
            passes.append(f"consensus_stability FAIL ({stability['pairwise_ari_mean']:.3f} < 0.9)")
        if heuristic_ari['ari'] > 0.3:
            passes.append("heuristic_ari PASS")
        else:
            passes.append(f"heuristic_ari FAIL ({heuristic_ari['ari']:.3f} < 0.3)")
        if comparison['single_vs_single_mean'] < comparison['single_vs_consensus_mean']:
            passes.append("consensus_improvement PASS (consensus more consistent than single-run)")
        else:
            passes.append("consensus_improvement FAIL (no improvement over single-run)")

        for p in passes:
            logger.info(f"  {p}")

    # Save all results
    results_path = out_dir / f"consensus_results_{args.variant}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nAll results saved to {results_path}")

    # Final summary table
    logger.info(f"\n{'='*80}")
    logger.info(f"FINAL COMPARISON: Consensus vs Single-Run ART")
    logger.info(f"{'='*80}")
    logger.info(f"{'rho':>6} | {'k':>3} | {'Heur ARI':>9} | {'Cons Stab':>9} | {'1-vs-cons':>9} | {'1-vs-1':>9}")
    logger.info(f"{'-'*6}-+-{'-'*3}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
    for rho_key, r in all_results.items():
        logger.info(
            f"{r['rho']:>6.2f} | {r['n_clusters']:>3} | "
            f"{r['heuristic']['ari']:>9.3f} | "
            f"{r['consensus_stability']['pairwise_ari_mean']:>9.3f} | "
            f"{r['vs_single_run']['single_vs_consensus_mean']:>9.3f} | "
            f"{r['vs_single_run']['single_vs_single_mean']:>9.3f}"
        )
    logger.info(f"\nRound 1 single-run pairwise ARI was 0.07-0.11. Target consensus stability: >0.9")


if __name__ == "__main__":
    main()
