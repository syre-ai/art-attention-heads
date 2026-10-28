#!/usr/bin/env python3
"""E4: Cross-checkpoint comparison — detect RLHF-induced attention shifts.

Compares feature matrices and consensus ART taxonomies across OLMo-2 checkpoints
(base, SFT, Instruct) to detect how alignment training modifies attention patterns.

Usage:
    python scripts/08_cross_checkpoint.py --tags base sft instruct
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics import adjusted_rand_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import get_results_dir
from src.utils.reproducibility import set_seed
from src.extraction.cache_manager import load_features
from src.extraction.feature_computer import FEATURE_NAMES
from src.clustering.consensus import consensus_art
from src.validation.known_heads import classify_heads_heuristic, compute_heuristic_agreement

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

N_LAYERS = 16
N_HEADS = 16


def compute_feature_drift(X_a, X_b, feature_names):
    """Compare two feature matrices per-feature and per-head."""
    n_heads, n_features = X_a.shape

    # Per-feature drift: how much does each feature change across all heads?
    per_feature = {}
    for f in range(n_features):
        col_a = X_a[:, f]
        col_b = X_b[:, f]
        diff = col_b - col_a
        per_feature[feature_names[f]] = {
            "mean_abs_change": float(np.mean(np.abs(diff))),
            "max_abs_change": float(np.max(np.abs(diff))),
            "mean_change": float(np.mean(diff)),  # directional
            "correlation": float(np.corrcoef(col_a, col_b)[0, 1]),
        }

    # Per-head drift: Euclidean distance in feature space
    per_head_dist = np.array([euclidean(X_a[i], X_b[i]) for i in range(n_heads)])

    # Reshape to layer x head for spatial analysis
    dist_grid = per_head_dist.reshape(N_LAYERS, N_HEADS)

    # Per-layer mean drift
    per_layer = {f"L{l}": float(dist_grid[l].mean()) for l in range(N_LAYERS)}

    # Top drifting heads
    top_k = 20
    top_indices = np.argsort(per_head_dist)[::-1][:top_k]
    top_drifters = []
    for idx in top_indices:
        layer = idx // N_HEADS
        head = idx % N_HEADS
        top_drifters.append({
            "head": f"L{layer}H{head}",
            "distance": float(per_head_dist[idx]),
            "features_a": X_a[idx].tolist(),
            "features_b": X_b[idx].tolist(),
        })

    return {
        "per_feature": per_feature,
        "per_layer_mean_drift": per_layer,
        "top_drifters": top_drifters,
        "global_mean_drift": float(per_head_dist.mean()),
        "global_max_drift": float(per_head_dist.max()),
        "global_std_drift": float(per_head_dist.std()),
    }


def compare_taxonomies(X_a, X_b, tag_a, tag_b, rho=0.5, k=8, n_runs=100, seed=42):
    """Run consensus ART on both checkpoints and compare category assignments."""
    logger.info(f"Running consensus ART on {tag_a}...")
    result_a = consensus_art(X_a, "fuzzy_art", rho, n_runs=n_runs,
                              linkage_method="average", n_clusters_target=k, seed=seed)

    logger.info(f"Running consensus ART on {tag_b}...")
    result_b = consensus_art(X_b, "fuzzy_art", rho, n_runs=n_runs,
                              linkage_method="average", n_clusters_target=k, seed=seed)

    # Category migration: which heads changed cluster?
    labels_a = result_a.labels
    labels_b = result_b.labels
    ari = adjusted_rand_score(labels_a, labels_b)

    # Migration matrix: count[i,j] = number of heads in cluster i (tag_a) that moved to cluster j (tag_b)
    k_a = len(np.unique(labels_a))
    k_b = len(np.unique(labels_b))
    migration = np.zeros((k_a, k_b), dtype=int)
    for i in range(len(labels_a)):
        migration[labels_a[i], labels_b[i]] += 1

    # Identify migrating heads. Cluster ids from two independent clusterings are
    # arbitrary, so match clusters first (max-overlap assignment), then compare.
    rows, cols = linear_sum_assignment(-migration)
    match = dict(zip(rows, cols))
    changed = np.array([match.get(a, -1) != b for a, b in zip(labels_a, labels_b)])
    n_changed = changed.sum()
    migrating_heads = []
    for idx in np.where(changed)[0]:
        layer = idx // N_HEADS
        head = idx % N_HEADS
        migrating_heads.append({
            "head": f"L{layer}H{head}",
            "from_cluster": int(labels_a[idx]),
            "to_cluster": int(labels_b[idx]),
        })

    # Co-association comparison: how does the block structure change?
    C_a = result_a.coassoc_matrix
    C_b = result_b.coassoc_matrix
    coassoc_corr = float(np.corrcoef(C_a.ravel(), C_b.ravel())[0, 1])

    return {
        "ari": float(ari),
        "n_changed": int(n_changed),
        "n_total": int(len(labels_a)),
        "frac_changed": float(n_changed / len(labels_a)),
        "migration_matrix": migration.tolist(),
        "migrating_heads": migrating_heads,
        "coassoc_correlation": coassoc_corr,
        "sizes_a": {int(l): int((labels_a == l).sum()) for l in np.unique(labels_a)},
        "sizes_b": {int(l): int((labels_b == l).sum()) for l in np.unique(labels_b)},
        "labels_a": labels_a,
        "labels_b": labels_b,
    }


def main():
    parser = argparse.ArgumentParser(description="E4: Cross-checkpoint comparison")
    parser.add_argument("--tags", nargs="+", default=["base", "sft", "instruct"],
                        help="Feature tags to compare")
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = get_results_dir("cross_checkpoint")

    # Load all feature matrices
    all_features = {}
    for tag in args.tags:
        try:
            features = load_features(tag)
            all_features[tag] = features["aggregated"]
            logger.info(f"Loaded {tag}: {all_features[tag].shape}")
        except FileNotFoundError:
            logger.warning(f"Features not found for tag '{tag}' — skipping")

    if len(all_features) < 2:
        logger.error("Need at least 2 checkpoints to compare. Exiting.")
        return

    available_tags = list(all_features.keys())
    all_results = {}

    # Pairwise comparisons
    for i, tag_a in enumerate(available_tags):
        for tag_b in available_tags[i+1:]:
            pair_key = f"{tag_a}_vs_{tag_b}"
            logger.info(f"\n{'='*60}")
            logger.info(f"Comparing {tag_a} vs {tag_b}")
            logger.info(f"{'='*60}")

            X_a = all_features[tag_a]
            X_b = all_features[tag_b]

            # Feature drift analysis
            logger.info("\nFeature drift analysis...")
            drift = compute_feature_drift(X_a, X_b, FEATURE_NAMES)

            logger.info(f"Global drift: mean={drift['global_mean_drift']:.3f}, max={drift['global_max_drift']:.3f}")
            logger.info("\nPer-feature drift (sorted by mean abs change):")
            sorted_features = sorted(drift["per_feature"].items(),
                                      key=lambda x: x[1]["mean_abs_change"], reverse=True)
            for fname, fdata in sorted_features:
                logger.info(f"  {fname:>20}: mean_abs={fdata['mean_abs_change']:.3f}, "
                             f"corr={fdata['correlation']:.3f}, direction={fdata['mean_change']:+.3f}")

            logger.info("\nPer-layer drift:")
            for layer_name, layer_drift in drift["per_layer_mean_drift"].items():
                bar = "█" * int(layer_drift * 20)
                logger.info(f"  {layer_name}: {layer_drift:.3f} {bar}")

            logger.info(f"\nTop 10 drifting heads:")
            for h in drift["top_drifters"][:10]:
                logger.info(f"  {h['head']}: distance={h['distance']:.3f}")

            # Taxonomy comparison
            logger.info(f"\nTaxonomy comparison (rho={args.rho}, k={args.k})...")
            taxonomy = compare_taxonomies(
                X_a, X_b, tag_a, tag_b,
                rho=args.rho, k=args.k, n_runs=args.n_runs, seed=args.seed
            )

            logger.info(f"\nTaxonomy ARI: {taxonomy['ari']:.3f}")
            logger.info(f"Heads that changed category: {taxonomy['n_changed']}/{taxonomy['n_total']} "
                         f"({taxonomy['frac_changed']:.1%})")
            logger.info(f"Co-association correlation: {taxonomy['coassoc_correlation']:.3f}")
            logger.info(f"Cluster sizes ({tag_a}): {taxonomy['sizes_a']}")
            logger.info(f"Cluster sizes ({tag_b}): {taxonomy['sizes_b']}")

            logger.info(f"\nMigration matrix ({tag_a} rows → {tag_b} cols):")
            migration = np.array(taxonomy["migration_matrix"])
            header = "    " + " ".join(f"C{j:>2}" for j in range(migration.shape[1]))
            logger.info(header)
            for row_idx in range(migration.shape[0]):
                row_str = " ".join(f"{migration[row_idx, col_idx]:>3}" for col_idx in range(migration.shape[1]))
                logger.info(f"C{row_idx}: {row_str}")

            if taxonomy["migrating_heads"]:
                logger.info(f"\nMigrating heads (showing first 20):")
                for mh in taxonomy["migrating_heads"][:20]:
                    logger.info(f"  {mh['head']}: C{mh['from_cluster']} → C{mh['to_cluster']}")

            # Save labels
            np.save(out_dir / f"labels_{tag_a}_{pair_key}.npy", taxonomy["labels_a"])
            np.save(out_dir / f"labels_{tag_b}_{pair_key}.npy", taxonomy["labels_b"])

            # Store results (without numpy arrays)
            pair_results = {
                "drift": drift,
                "taxonomy_ari": taxonomy["ari"],
                "n_changed": taxonomy["n_changed"],
                "frac_changed": taxonomy["frac_changed"],
                "coassoc_correlation": taxonomy["coassoc_correlation"],
                "migration_matrix": taxonomy["migration_matrix"],
                "migrating_heads": taxonomy["migrating_heads"],
                "sizes_a": taxonomy["sizes_a"],
                "sizes_b": taxonomy["sizes_b"],
            }
            all_results[pair_key] = pair_results

    # Save all results
    results_path = out_dir / "cross_checkpoint_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nAll results saved to {results_path}")

    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info(f"CROSS-CHECKPOINT SUMMARY")
    logger.info(f"{'='*80}")
    for pair_key, r in all_results.items():
        logger.info(f"\n{pair_key}:")
        logger.info(f"  Global drift: {r['drift']['global_mean_drift']:.3f}")
        logger.info(f"  Taxonomy ARI: {r['taxonomy_ari']:.3f}")
        logger.info(f"  Category migration: {r['frac_changed']:.1%}")
        logger.info(f"  Co-association corr: {r['coassoc_correlation']:.3f}")
        # Top 3 drifting features
        top_feats = sorted(r["drift"]["per_feature"].items(),
                           key=lambda x: x[1]["mean_abs_change"], reverse=True)[:3]
        top_str = ", ".join(f"{n}({d['mean_abs_change']:.3f})" for n, d in top_feats)
        logger.info(f"  Top drifting features: {top_str}")


if __name__ == "__main__":
    main()
