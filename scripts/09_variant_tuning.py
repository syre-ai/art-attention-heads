#!/usr/bin/env python3
"""E6: GaussianART / HypersphereART tuning + cross-variant comparison.

Round 1 found GaussianART over-clusters (150+ categories at rho=0.1 with sigma_init_scale=0.1)
and HypersphereART over-clusters (32+ at r_hat=0.5). Only FuzzyART produced useful results.

This script:
1. Tests corrected hyperparameters for GaussianART and HypersphereART
2. Runs consensus clustering on each working variant
3. Computes cross-variant ARI to check convergent validation

Usage:
    python scripts/09_variant_tuning.py --tag base
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import get_results_dir
from src.utils.reproducibility import set_seed
from src.extraction.cache_manager import load_features
from src.clustering.art_runner import run_art
from src.clustering.consensus import consensus_art
from src.validation.known_heads import classify_heads_heuristic, compute_heuristic_agreement

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def phase1_sweep(X, variant, rho_values, model_params, seed=42):
    """Quick sweep: how many clusters does each (variant, rho, params) produce?"""
    results = []
    rng = np.random.default_rng(seed)
    for rho in rho_values:
        n_clusters_list = []
        for _ in range(5):  # 5 random orderings for quick stats
            shuffle = rng.permutation(len(X))
            r = run_art(X, variant, rho, shuffle_order=shuffle, model_params=model_params)
            n_clusters_list.append(r.n_clusters)
        results.append({
            "rho": rho,
            "n_clusters_mean": float(np.mean(n_clusters_list)),
            "n_clusters_std": float(np.std(n_clusters_list)),
            "n_clusters_range": [int(min(n_clusters_list)), int(max(n_clusters_list))],
        })
        logger.info(f"  {variant} rho={rho}: k={np.mean(n_clusters_list):.1f} +/- {np.std(n_clusters_list):.1f} [{min(n_clusters_list)}-{max(n_clusters_list)}]")
    return results


def phase2_consensus(X, variant, rho, k, model_params, heuristic_labels, out_dir, seed=42):
    """Run consensus with best params for a variant."""
    logger.info(f"\nConsensus: {variant} rho={rho} k={k}")
    result = consensus_art(
        X, variant, rho,
        n_runs=100,
        linkage_method="average",
        n_clusters_target=k,
        seed=seed,
        model_params=model_params,
    )

    h = compute_heuristic_agreement(result.labels, heuristic_labels)

    # Silhouette on original feature space
    if result.n_clusters > 1:
        sil = silhouette_score(X, result.labels)
    else:
        sil = 0.0

    # Cluster sizes
    sizes = {int(l): int((result.labels == l).sum()) for l in np.unique(result.labels)}

    tag = f"{variant}_rho{rho}_k{k}"
    np.save(out_dir / f"variant_labels_{tag}.npy", result.labels)

    logger.info(f"  H-ARI: {h['ari']:.3f}, Silhouette: {sil:.3f}")
    logger.info(f"  Sizes: {sizes}")

    return {
        "variant": variant,
        "rho": rho,
        "k_target": k,
        "k_actual": result.n_clusters,
        "heuristic_ari": h["ari"],
        "heuristic_nmi": h["nmi"],
        "silhouette": float(sil),
        "sizes": sizes,
        "art_k_median": result.metadata["n_clusters_median"],
        "art_k_range": result.metadata["n_clusters_range"],
        "labels": result.labels,  # for cross-variant comparison
    }


def main():
    parser = argparse.ArgumentParser(description="E6: Variant tuning")
    parser.add_argument("--tag", default="base")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    features = load_features(args.tag)
    X = features["aggregated"]
    heuristic_labels, _ = classify_heads_heuristic(X)
    out_dir = get_results_dir("variant_tuning")

    rho_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # ==========================================
    # Phase 1: Parameter sweep for each variant
    # ==========================================
    logger.info("=" * 60)
    logger.info("PHASE 1: Parameter sweeps")
    logger.info("=" * 60)

    # GaussianART: test sigma_init_scale = {0.3, 0.5, 1.0}
    # Round 1 used 0.1 which was too tight (150+ clusters)
    # Feature stds range 0.11-0.27, so 0.3 should be a reasonable starting point
    gaussian_sweeps = {}
    for sigma in [0.3, 0.5, 1.0]:
        params = {"sigma_init_scale": sigma}
        logger.info(f"\n--- GaussianART sigma_init_scale={sigma} ---")
        sweep = phase1_sweep(X, "gaussian_art", rho_values, params, seed=args.seed)
        gaussian_sweeps[f"sigma_{sigma}"] = sweep

    # HypersphereART: test r_hat = {1.0, 2.0, 3.0}
    # Round 1 used 0.5 which was too small (32+ clusters)
    hypersphere_sweeps = {}
    for r_hat in [1.0, 2.0, 3.0]:
        params = {"r_hat": r_hat}
        logger.info(f"\n--- HypersphereART r_hat={r_hat} ---")
        sweep = phase1_sweep(X, "hypersphere_art", rho_values, params, seed=args.seed)
        hypersphere_sweeps[f"rhat_{r_hat}"] = sweep

    # FuzzyART baseline (for reference)
    logger.info(f"\n--- FuzzyART (reference) ---")
    fuzzy_sweep = phase1_sweep(X, "fuzzy_art", rho_values, {}, seed=args.seed)

    # Save phase 1 results
    phase1_results = {
        "gaussian_sweeps": gaussian_sweeps,
        "hypersphere_sweeps": hypersphere_sweeps,
        "fuzzy_sweep": fuzzy_sweep,
    }
    with open(out_dir / "phase1_sweeps.json", "w") as f:
        json.dump(phase1_results, f, indent=2)

    # ==========================================
    # Phase 2: Select best params, run consensus
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Consensus with best params")
    logger.info("=" * 60)

    # Find Gaussian config that produces ~8-15 clusters at moderate rho
    best_gaussian = _select_best_config(gaussian_sweeps, "gaussian_art")
    best_hypersphere = _select_best_config(hypersphere_sweeps, "hypersphere_art")

    consensus_results = {}

    # FuzzyART consensus (our known best: rho=0.5 k=8)
    fuzzy_result = phase2_consensus(
        X, "fuzzy_art", 0.5, 8, {},
        heuristic_labels, out_dir, seed=args.seed,
    )
    consensus_results["fuzzy_art"] = fuzzy_result

    # GaussianART consensus
    if best_gaussian:
        sigma, rho, k = best_gaussian
        gauss_result = phase2_consensus(
            X, "gaussian_art", rho, k,
            {"sigma_init_scale": sigma},
            heuristic_labels, out_dir, seed=args.seed,
        )
        consensus_results["gaussian_art"] = gauss_result

    # HypersphereART consensus
    if best_hypersphere:
        r_hat, rho, k = best_hypersphere
        hyper_result = phase2_consensus(
            X, "hypersphere_art", rho, k,
            {"r_hat": r_hat},
            heuristic_labels, out_dir, seed=args.seed,
        )
        consensus_results["hypersphere_art"] = hyper_result

    # ==========================================
    # Phase 3: Cross-variant comparison
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Cross-variant comparison")
    logger.info("=" * 60)

    variant_names = list(consensus_results.keys())
    cross_ari = {}
    for i, va in enumerate(variant_names):
        for vb in variant_names[i+1:]:
            ari = adjusted_rand_score(
                consensus_results[va]["labels"],
                consensus_results[vb]["labels"],
            )
            pair = f"{va}_vs_{vb}"
            cross_ari[pair] = float(ari)
            logger.info(f"  {pair}: ARI = {ari:.3f}")

    # ==========================================
    # Summary
    # ==========================================
    logger.info("\n" + "=" * 80)
    logger.info("E6 SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\n{'Variant':>20} | {'rho':>5} | {'k':>3} | {'H-ARI':>6} | {'Silh':>6} | {'Sizes'}")
    logger.info(f"{'-'*20}-+-{'-'*5}-+-{'-'*3}-+-{'-'*6}-+-{'-'*6}-+-{'-'*30}")
    for name, r in consensus_results.items():
        sizes_str = ", ".join(f"{s}" for s in sorted(r["sizes"].values(), reverse=True))
        logger.info(
            f"{name:>20} | {r['rho']:>5.1f} | {r['k_target']:>3} | "
            f"{r['heuristic_ari']:>6.3f} | {r['silhouette']:>6.3f} | [{sizes_str}]"
        )

    if cross_ari:
        logger.info(f"\nCross-variant ARI:")
        for pair, ari in cross_ari.items():
            logger.info(f"  {pair}: {ari:.3f}")
        logger.info(f"Interpretation: ARI > 0.5 = convergent validation (variants find same structure)")
        logger.info(f"               ARI < 0.3 = different structure (interesting disagreement)")

    # Save final results (without numpy arrays)
    final = {
        "consensus": {k: {kk: vv for kk, vv in v.items() if kk != "labels"}
                      for k, v in consensus_results.items()},
        "cross_variant_ari": cross_ari,
        "phase1": phase1_results,
    }
    with open(out_dir / "variant_tuning_results.json", "w") as f:
        json.dump(final, f, indent=2)
    logger.info(f"\nResults saved to {out_dir / 'variant_tuning_results.json'}")


def _select_best_config(sweeps, variant_name):
    """Select (param_value, rho, k) that gives ~8 clusters at moderate rho."""
    best = None
    best_score = float("inf")

    for param_key, sweep_results in sweeps.items():
        param_val = float(param_key.split("_")[1])
        for entry in sweep_results:
            rho = entry["rho"]
            k_mean = entry["n_clusters_mean"]
            # Score: prefer k near 8, rho near 0.5
            k_penalty = abs(k_mean - 8) / 8
            rho_penalty = abs(rho - 0.5) / 0.5
            # Only consider if k is in reasonable range
            if k_mean < 3 or k_mean > 30:
                continue
            score = k_penalty + 0.3 * rho_penalty
            if score < best_score:
                best_score = score
                best = (param_val, rho, round(k_mean))

    if best:
        logger.info(f"  Best {variant_name}: param={best[0]}, rho={best[1]}, k~{best[2]}")
    else:
        logger.warning(f"  No good config found for {variant_name}")
    return best


if __name__ == "__main__":
    main()
