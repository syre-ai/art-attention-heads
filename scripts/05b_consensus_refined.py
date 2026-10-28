#!/usr/bin/env python3
"""E1 Iter 2: Refined consensus — fix k, increase N, try ward linkage.

Addresses two sources of instability from E1 iter 1:
1. Co-association noise: N=50 leaves ~10% uncertainty → increase to N=100
2. Auto k-selection: median k varies between repeats → fix k explicitly

Also tests ward linkage (balanced clusters) vs average (co-association standard).

Usage:
    python scripts/05b_consensus_refined.py --tag base
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import get_results_dir
from src.utils.reproducibility import set_seed
from src.extraction.cache_manager import load_features
from src.clustering.consensus import consensus_art, consensus_stability
from src.validation.known_heads import classify_heads_heuristic, compute_heuristic_agreement

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="E1 refined consensus")
    parser.add_argument("--tag", default="base")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    features = load_features(args.tag)
    X = features["aggregated"]
    heuristic_labels, _ = classify_heads_heuristic(X)
    out_dir = get_results_dir("consensus")

    # Test matrix: {N} x {linkage} x {k} x {rho}
    configs = [
        # rho, k, N, linkage
        (0.3, 3, 100, "average"),
        (0.3, 5, 100, "average"),
        (0.3, 8, 100, "average"),
        (0.5, 8, 100, "average"),
        (0.5, 10, 100, "average"),
        (0.5, 15, 100, "average"),
        # Ward linkage variants
        (0.3, 5, 100, "ward"),
        (0.5, 10, 100, "ward"),
        (0.3, 3, 100, "ward"),
        (0.5, 8, 100, "ward"),
    ]

    all_results = {}

    for rho, k, n_runs, link in configs:
        key = f"rho{rho}_k{k}_N{n_runs}_{link}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Config: rho={rho}, k={k}, N={n_runs}, linkage={link}")
        logger.info(f"{'='*60}")

        # Primary consensus
        result = consensus_art(
            X, "fuzzy_art", rho,
            n_runs=n_runs,
            linkage_method=link,
            n_clusters_target=k,
            seed=args.seed,
        )

        # Heuristic agreement
        h = compute_heuristic_agreement(result.labels, heuristic_labels)

        # Stability: 5 repeats with different seeds
        stability = consensus_stability(
            X, "fuzzy_art", rho,
            n_runs=n_runs,
            n_repeats=5,
            linkage_method=link,
            seed=args.seed + 1000,
        )

        # Co-association diagnostics
        C = result.coassoc_matrix
        unique_labels = np.unique(result.labels)

        cluster_info = {}
        for label in unique_labels:
            mask = result.labels == label
            n = mask.sum()
            if n > 1:
                within = C[np.ix_(mask, mask)]
                off_diag = within[np.triu_indices(n, k=1)]
                cluster_info[int(label)] = {
                    "size": int(n),
                    "cohesion_mean": float(off_diag.mean()),
                    "cohesion_min": float(off_diag.min()),
                }
            else:
                cluster_info[int(label)] = {"size": 1, "cohesion_mean": 1.0, "cohesion_min": 1.0}

        entry = {
            "rho": rho, "k_target": k, "k_actual": result.n_clusters,
            "n_runs": n_runs, "linkage": link,
            "heuristic_ari": h["ari"], "heuristic_nmi": h["nmi"],
            "consensus_stability_ari": stability["pairwise_ari_mean"],
            "consensus_stability_std": stability["pairwise_ari_std"],
            "consensus_stability_min": stability["pairwise_ari_min"],
            "art_k_median": result.metadata["n_clusters_median"],
            "art_k_range": result.metadata["n_clusters_range"],
            "coassoc_mean": float(C.mean()),
            "cluster_info": cluster_info,
        }
        all_results[key] = entry

        # Save labels for best configs
        np.save(out_dir / f"refined_labels_{key}.npy", result.labels)

        logger.info(f"  k_actual={result.n_clusters}, ART median k={result.metadata['n_clusters_median']}")
        logger.info(f"  Heuristic ARI: {h['ari']:.3f}")
        logger.info(f"  Consensus stability: {stability['pairwise_ari_mean']:.3f}+/-{stability['pairwise_ari_std']:.3f} (min={stability['pairwise_ari_min']:.3f})")
        for label, ci in cluster_info.items():
            logger.info(f"    Cluster {label}: size={ci['size']}, cohesion={ci['cohesion_mean']:.3f}")

    # Save all
    results_path = out_dir / "consensus_refined_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    logger.info(f"\n{'='*80}")
    logger.info(f"REFINED CONSENSUS COMPARISON")
    logger.info(f"{'='*80}")
    logger.info(f"{'rho':>5} | {'k':>3} | {'link':>7} | {'H-ARI':>6} | {'Stab':>6} | {'Stab-min':>8} | {'ART-k':>5}")
    logger.info(f"{'-'*5}-+-{'-'*3}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*5}")
    for key, r in all_results.items():
        logger.info(
            f"{r['rho']:>5.1f} | {r['k_target']:>3} | {r['linkage']:>7} | "
            f"{r['heuristic_ari']:>6.3f} | {r['consensus_stability_ari']:>6.3f} | "
            f"{r['consensus_stability_min']:>8.3f} | {r['art_k_median']:>5}"
        )

    logger.info(f"\nTargets: Stability > 0.9, Heuristic ARI > 0.3")
    logger.info(f"Baseline reference: K-Means k=3 H-ARI=0.335 stability=0.992")
    logger.info(f"E1 iter 1 best: rho=0.5 k=10 H-ARI=0.585 stability=0.506")


if __name__ == "__main__":
    main()
