#!/usr/bin/env python3
"""Step 2: Load cached features, run ART clustering, sweep vigilance.

Usage:
    python scripts/02_run_clustering.py [--tag base] [--variants fuzzy_art,gaussian_art,hypersphere_art]
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
from src.clustering.vigilance_sweep import sweep_vigilance, sweep_all_variants
from src.clustering.comparison import compare_all_rho, agreement_summary_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run ART clustering")
    parser.add_argument("--tag", default="base", help="Feature cache tag")
    parser.add_argument("--variants", default=None, help="Comma-separated variant names")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load features
    data = load_features(args.tag)
    X = data["aggregated"]
    logger.info(f"Loaded features: {X.shape}")

    # Run sweeps
    if args.variants:
        variant_names = args.variants.split(",")
        sweep_results = {}
        for variant in variant_names:
            sweep_results[variant] = sweep_vigilance(X, variant, seed=args.seed)
    else:
        sweep_results = sweep_all_variants(X, seed=args.seed)

    # Save sweep results
    out_dir = get_results_dir("clustering")

    # Save as pickle for full object preservation
    with open(out_dir / f"sweep_results_{args.tag}.pkl", "wb") as f:
        pickle.dump(sweep_results, f)

    # Save summary as JSON
    summary = {}
    for variant, sweep in sweep_results.items():
        summary[variant] = {
            "rho_values": sweep.rho_values,
            "n_clusters": sweep.n_clusters,
            "ordering_runs": {str(k): v for k, v in sweep.ordering_runs.items()},
        }
        # Save labels per rho
        for i, rho in enumerate(sweep.rho_values):
            label_path = out_dir / f"labels_{args.tag}_{variant}_rho{rho:.2f}.npy"
            np.save(label_path, sweep.results[i].labels)

    with open(out_dir / f"sweep_summary_{args.tag}.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Cross-variant comparison
    logger.info("=== Cross-Variant Comparison ===")
    all_agreements = compare_all_rho(sweep_results)
    table, rho_values, pair_names = agreement_summary_table(all_agreements)

    # Save comparison
    comparison = {"rho_values": rho_values, "pairs": [str(p) for p in pair_names]}
    for i, rho in enumerate(rho_values):
        for j, pair in enumerate(pair_names):
            key = f"{pair}_rho{rho}"
            comparison[key] = {"ari": float(table[i, j, 0]), "nmi": float(table[i, j, 1])}

    with open(out_dir / f"comparison_{args.tag}.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Print summary
    print("\n=== Vigilance Sweep Summary ===")
    for variant, sweep in sweep_results.items():
        print(f"\n{variant}:")
        for rho, nc in zip(sweep.rho_values, sweep.n_clusters):
            runs = sweep.ordering_runs.get(rho, [])
            run_str = f" (range: [{min(runs)}-{max(runs)}])" if runs else ""
            print(f"  rho={rho:.2f}: {nc} clusters{run_str}")

    print("\n=== Cross-Variant ARI ===")
    for rho, agreements in all_agreements.items():
        for (va, vb), metrics in agreements.items():
            print(f"  rho={rho:.2f}: {va} vs {vb} — ARI={metrics['ari']:.3f}, NMI={metrics['nmi']:.3f}")

    logger.info("Clustering complete!")


if __name__ == "__main__":
    main()
