#!/usr/bin/env python3
"""Step 3: Run all validation checks.

Usage:
    python scripts/03_validate.py [--tag base] [--variant fuzzy_art] [--rho 0.5]
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config, get_results_dir
from src.utils.reproducibility import set_seed
from src.extraction.cache_manager import load_features
from src.clustering.art_runner import run_art
from src.validation.known_heads import classify_heads_heuristic, compute_heuristic_agreement
from src.validation.stability import (
    bootstrap_stability,
    split_half_stability,
    ordering_sensitivity,
    feature_ablation,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run validation")
    parser.add_argument("--tag", default="base", help="Feature cache tag")
    parser.add_argument("--variant", default="fuzzy_art", help="ART variant to validate")
    parser.add_argument("--rho", type=float, default=0.5, help="Vigilance parameter")
    parser.add_argument("--n-bootstrap", type=int, default=100, help="Bootstrap samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = get_results_dir("validation")

    # Load features
    data = load_features(args.tag)
    X = data["aggregated"]
    per_input = data["per_input"]
    logger.info(f"Loaded features: aggregated {X.shape}, per_input {per_input.shape}")

    # Run clustering at specified rho
    result = run_art(X, args.variant, args.rho)
    logger.info(f"Clustering: {result.n_clusters} categories")

    validation_results = {}

    # V1: Known Head Type Agreement
    print("\n=== V1: Known Head Type Agreement ===")
    heuristic_labels, groups = classify_heads_heuristic(X)
    agreement = compute_heuristic_agreement(result.labels, heuristic_labels)
    validation_results["v1_heuristic"] = agreement
    print(f"  ARI: {agreement['ari']:.3f}")
    print(f"  NMI: {agreement['nmi']:.3f}")
    print(f"  Known heads: {agreement['n_known_heads']}")
    for head_type, indices in groups.items():
        if indices:
            print(f"  {head_type}: {len(indices)} heads")

    # V2: Stability
    print("\n=== V2: Bootstrap Stability ===")
    bootstrap = bootstrap_stability(
        X, per_input, args.variant, args.rho,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    validation_results["v2_bootstrap"] = bootstrap
    print(f"  ARI: {bootstrap['mean_ari']:.3f} +/- {bootstrap['std_ari']:.3f}")

    print("\n=== V2: Split-Half Stability ===")
    split_half = split_half_stability(
        per_input, args.variant, args.rho, seed=args.seed,
    )
    validation_results["v2_split_half"] = split_half
    print(f"  ARI: {split_half['mean_ari']:.3f} +/- {split_half['std_ari']:.3f}")

    print("\n=== V2: Ordering Sensitivity ===")
    ordering = ordering_sensitivity(
        X, args.variant, args.rho, seed=args.seed,
    )
    validation_results["v2_ordering"] = ordering
    print(f"  N clusters: {ordering['n_clusters_mean']:.1f} +/- {ordering['n_clusters_std']:.1f}")
    print(f"  CV: {ordering['cv']:.3f}")
    print(f"  Pairwise ARI: {ordering['pairwise_ari_mean']:.3f}")

    print("\n=== V2: Feature Ablation ===")
    ablation = feature_ablation(X, args.variant, args.rho)
    validation_results["v2_feature_ablation"] = ablation
    # Sort by importance (ARI drop)
    sorted_features = sorted(ablation.items(), key=lambda x: x[1]["ari_drop"], reverse=True)
    print("  Top 5 most important features:")
    for name, metrics in sorted_features[:5]:
        print(f"    {name}: ARI drop={metrics['ari_drop']:.3f}, n_clusters diff={metrics['n_clusters_diff']}")

    # Summary
    print("\n=== Validation Summary ===")
    success = True
    checks = [
        ("V1: Heuristic ARI > 0.3", agreement["ari"] > 0.3),
        ("V2: Bootstrap ARI > 0.7", bootstrap["mean_ari"] > 0.7),
        ("V2: Split-half ARI > 0.6", split_half["mean_ari"] > 0.6),
        ("V2: Ordering CV < 0.2", ordering["cv"] < 0.2),
    ]
    for check_name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check_name}")
        if not passed:
            success = False

    # Save results
    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(validation_results, default=_convert))
    with open(out_dir / f"validation_{args.tag}_{args.variant}_rho{args.rho:.2f}.json", "w") as f:
        json.dump(serializable, f, indent=2)

    logger.info(f"Validation {'PASSED' if success else 'FAILED'}")


if __name__ == "__main__":
    main()
