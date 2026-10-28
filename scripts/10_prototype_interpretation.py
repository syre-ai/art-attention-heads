#!/usr/bin/env python3
"""E5-adjacent: Deep prototype interpretation of consensus taxonomy.

Analyzes the 8-cluster consensus taxonomy (rho=0.5 k=8 avg N=100) to:
1. Name each cluster based on dominant features
2. Map clusters to known head types (sink, induction, prev-token, positional)
3. Analyze layer distribution of each cluster
4. Identify signature features per cluster (what makes it unique)
5. Find borderline heads (low co-association with their assigned cluster)

This is a pure analysis script using existing cached results — no new extraction needed.

Usage:
    python scripts/10_prototype_interpretation.py --tag base
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import get_results_dir
from src.utils.reproducibility import set_seed
from src.extraction.cache_manager import load_features
from src.extraction.feature_computer import FEATURE_NAMES
from src.clustering.consensus import consensus_art
from src.validation.known_heads import classify_heads_heuristic, LABEL_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

N_LAYERS = 16
N_HEADS = 16

# Reverse label map for readable names
HEURISTIC_NAMES = {v: k for k, v in LABEL_MAP.items()}


def compute_cluster_profiles(X, labels):
    """Per-cluster mean and std of each feature."""
    profiles = {}
    for label in np.unique(labels):
        mask = labels == label
        cluster_data = X[mask]
        profiles[int(label)] = {
            "mean": cluster_data.mean(axis=0),
            "std": cluster_data.std(axis=0),
            "size": int(mask.sum()),
        }
    return profiles


def identify_signature_features(profiles, feature_names):
    """For each cluster, find features that most distinguish it from the global mean."""
    global_mean = np.mean([p["mean"] for p in profiles.values()], axis=0)
    global_std = np.std([p["mean"] for p in profiles.values()], axis=0)
    global_std = np.maximum(global_std, 0.01)  # avoid /0

    signatures = {}
    for label, profile in profiles.items():
        z_scores = (profile["mean"] - global_mean) / global_std
        # Top 3 most distinctive features (positive or negative)
        top_idx = np.argsort(np.abs(z_scores))[::-1][:5]
        sig_features = []
        for idx in top_idx:
            sig_features.append({
                "feature": feature_names[idx],
                "z_score": float(z_scores[idx]),
                "value": float(profile["mean"][idx]),
                "direction": "high" if z_scores[idx] > 0 else "low",
            })
        signatures[label] = sig_features
    return signatures


def suggest_cluster_name(signature_features):
    """Suggest a name for a cluster based on its signature features."""
    top = signature_features[:3]
    patterns = {feat["feature"]: feat["direction"] for feat in top}

    # Rule-based naming heuristics
    if patterns.get("bos_sink_score") == "high":
        if patterns.get("entropy") == "low":
            return "sink_focused"
        return "sink_distributed"
    if patterns.get("induction_score") == "high":
        if patterns.get("copy_score") == "high":
            return "induction_copy"
        return "induction"
    if patterns.get("previous_token") == "high":
        return "previous_token"
    if patterns.get("positional_consistency") == "high":
        return "positional"
    if patterns.get("local_window") == "high":
        return "local_attention"
    if patterns.get("entropy") == "high" and patterns.get("mean_distance") == "high":
        return "broad_distributed"
    if patterns.get("max_attention") == "high" and patterns.get("entropy") == "low":
        return "sharp_focused"
    if patterns.get("causal_decay_rate") == "high":
        return "recency_biased"
    if patterns.get("self_attention") == "high":
        return "self_focused"
    if patterns.get("kurtosis") == "high":
        return "heavy_tailed"

    # Fallback: use top feature
    return f"{top[0]['feature']}_{top[0]['direction']}"


def analyze_layer_distribution(labels, n_layers=N_LAYERS, n_heads=N_HEADS):
    """How are cluster members distributed across layers?"""
    layer_dist = {}
    for label in np.unique(labels):
        mask = labels == label
        head_indices = np.where(mask)[0]
        layers = head_indices // n_heads
        layer_counts = np.bincount(layers, minlength=n_layers)
        layer_dist[int(label)] = {
            "counts": layer_counts.tolist(),
            "dominant_layers": np.where(layer_counts > 0)[0].tolist(),
            "concentration": float(layer_counts.max() / max(layer_counts.sum(), 1)),
            "spread": int(np.sum(layer_counts > 0)),
        }
    return layer_dist


def map_to_heuristic_types(labels, heuristic_labels):
    """For each ART cluster, what fraction maps to each heuristic type?"""
    mapping = {}
    for label in np.unique(labels):
        mask = labels == label
        h_labels = heuristic_labels[mask]
        type_counts = {}
        for h_label in np.unique(h_labels):
            name = HEURISTIC_NAMES.get(h_label, f"type_{h_label}")
            type_counts[name] = int((h_labels == h_label).sum())
        mapping[int(label)] = {
            "type_counts": type_counts,
            "dominant_type": max(type_counts, key=type_counts.get),
            "purity": float(max(type_counts.values()) / len(h_labels)),
        }
    return mapping


def find_borderline_heads(labels, coassoc_matrix, top_k=20):
    """Find heads with lowest within-cluster co-association (most uncertain assignments)."""
    borderline = []
    for head_idx in range(len(labels)):
        cluster = labels[head_idx]
        cluster_mask = labels == cluster
        # Mean co-association with cluster-mates
        within_coassoc = coassoc_matrix[head_idx, cluster_mask].mean()
        # Mean co-association with nearest other cluster
        other_clusters = np.unique(labels[labels != cluster])
        max_other = 0.0
        best_other = -1
        for other in other_clusters:
            other_mask = labels == other
            other_mean = coassoc_matrix[head_idx, other_mask].mean()
            if other_mean > max_other:
                max_other = other_mean
                best_other = int(other)

        layer = head_idx // N_HEADS
        head = head_idx % N_HEADS
        borderline.append({
            "head": f"L{layer}H{head}",
            "idx": int(head_idx),
            "cluster": int(cluster),
            "within_coassoc": float(within_coassoc),
            "nearest_other_cluster": best_other,
            "nearest_other_coassoc": float(max_other),
            "margin": float(within_coassoc - max_other),
        })

    # Sort by margin (lowest = most borderline)
    borderline.sort(key=lambda x: x["margin"])
    return borderline[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Prototype interpretation")
    parser.add_argument("--tag", default="base")
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    features = load_features(args.tag)
    X = features["aggregated"]
    heuristic_labels, heuristic_groups = classify_heads_heuristic(X)
    out_dir = get_results_dir("interpretation")

    # Run consensus to get labels + co-association matrix
    logger.info(f"Running consensus ART: rho={args.rho}, k={args.k}, N={args.n_runs}")
    result = consensus_art(
        X, "fuzzy_art", args.rho,
        n_runs=args.n_runs,
        linkage_method="average",
        n_clusters_target=args.k,
        seed=args.seed,
    )
    labels = result.labels

    # ==========================================
    # 1. Cluster profiles
    # ==========================================
    logger.info("\n" + "=" * 80)
    logger.info("1. CLUSTER PROFILES")
    logger.info("=" * 80)

    profiles = compute_cluster_profiles(X, labels)
    for label, p in sorted(profiles.items()):
        logger.info(f"\nCluster {label} (n={p['size']}):")
        for i, fname in enumerate(FEATURE_NAMES):
            bar = "█" * int(p["mean"][i] * 30)
            logger.info(f"  {fname:>25}: {p['mean'][i]:.3f} ±{p['std'][i]:.3f} {bar}")

    # ==========================================
    # 2. Signature features + naming
    # ==========================================
    logger.info("\n" + "=" * 80)
    logger.info("2. SIGNATURE FEATURES & CLUSTER NAMES")
    logger.info("=" * 80)

    signatures = identify_signature_features(profiles, FEATURE_NAMES)
    cluster_names = {}
    for label, sigs in sorted(signatures.items()):
        name = suggest_cluster_name(sigs)
        cluster_names[label] = name
        logger.info(f"\nCluster {label} → \"{name}\" (n={profiles[label]['size']})")
        for s in sigs:
            logger.info(f"  {s['feature']:>25}: z={s['z_score']:+.2f} (val={s['value']:.3f}, {s['direction']})")

    # ==========================================
    # 3. Heuristic type mapping
    # ==========================================
    logger.info("\n" + "=" * 80)
    logger.info("3. HEURISTIC TYPE MAPPING")
    logger.info("=" * 80)

    type_mapping = map_to_heuristic_types(labels, heuristic_labels)
    for label, m in sorted(type_mapping.items()):
        logger.info(f"\nCluster {label} \"{cluster_names.get(label, '?')}\":")
        for tname, count in sorted(m["type_counts"].items(), key=lambda x: -x[1]):
            pct = count / profiles[label]["size"] * 100
            logger.info(f"  {tname:>15}: {count:3d} ({pct:5.1f}%)")
        logger.info(f"  Purity: {m['purity']:.3f}, Dominant: {m['dominant_type']}")

    # ==========================================
    # 4. Layer distribution
    # ==========================================
    logger.info("\n" + "=" * 80)
    logger.info("4. LAYER DISTRIBUTION")
    logger.info("=" * 80)

    layer_dist = analyze_layer_distribution(labels)
    for label, ld in sorted(layer_dist.items()):
        counts = ld["counts"]
        logger.info(f"\nCluster {label} \"{cluster_names.get(label, '?')}\" (spread={ld['spread']}/16, conc={ld['concentration']:.2f}):")
        for layer_idx, count in enumerate(counts):
            if count > 0:
                bar = "█" * count
                logger.info(f"  L{layer_idx:2d}: {bar} ({count})")

    # ==========================================
    # 5. Borderline heads
    # ==========================================
    logger.info("\n" + "=" * 80)
    logger.info("5. BORDERLINE HEADS (most uncertain assignments)")
    logger.info("=" * 80)

    borderline = find_borderline_heads(labels, result.coassoc_matrix, top_k=20)
    logger.info(f"\n{'Head':>8} | {'Cluster':>8} | {'Within':>7} | {'Nearest':>8} | {'Other':>7} | {'Margin':>7}")
    logger.info(f"{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}")
    for bh in borderline:
        logger.info(
            f"{bh['head']:>8} | C{bh['cluster']:>7} | {bh['within_coassoc']:>7.3f} | "
            f"C{bh['nearest_other_cluster']:>7} | {bh['nearest_other_coassoc']:>7.3f} | {bh['margin']:>7.3f}"
        )

    # ==========================================
    # Summary table
    # ==========================================
    logger.info("\n" + "=" * 80)
    logger.info("TAXONOMY SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\n{'Cluster':>8} | {'Name':>20} | {'Size':>5} | {'Dominant Type':>15} | {'Purity':>7} | {'Layers'}")
    logger.info(f"{'-'*8}-+-{'-'*20}-+-{'-'*5}-+-{'-'*15}-+-{'-'*7}-+-{'-'*20}")
    for label in sorted(profiles.keys()):
        name = cluster_names.get(label, "?")
        size = profiles[label]["size"]
        dom = type_mapping[label]["dominant_type"]
        purity = type_mapping[label]["purity"]
        layers_active = [i for i, c in enumerate(layer_dist[label]["counts"]) if c > 0]
        layer_str = ",".join(str(l) for l in layers_active[:8])
        if len(layers_active) > 8:
            layer_str += "..."
        logger.info(f"C{label:>7} | {name:>20} | {size:>5} | {dom:>15} | {purity:>7.3f} | {layer_str}")

    # Save results
    results = {
        "cluster_names": cluster_names,
        "profiles": {k: {"mean": v["mean"].tolist(), "std": v["std"].tolist(), "size": v["size"]}
                     for k, v in profiles.items()},
        "signatures": {k: v for k, v in signatures.items()},
        "type_mapping": type_mapping,
        "layer_distribution": layer_dist,
        "borderline_heads": borderline,
        "config": {"rho": args.rho, "k": args.k, "n_runs": args.n_runs, "tag": args.tag},
    }
    results_path = out_dir / "interpretation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Save labels
    np.save(out_dir / f"interpretation_labels_{args.tag}.npy", labels)


if __name__ == "__main__":
    main()
