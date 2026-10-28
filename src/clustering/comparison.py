"""Cross-variant agreement metrics (ARI, NMI)."""

import logging
from itertools import combinations

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .vigilance_sweep import SweepResult

logger = logging.getLogger(__name__)


def compute_agreement(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> dict[str, float]:
    """Compute ARI and NMI between two label vectors.

    Parameters
    ----------
    labels_a, labels_b : np.ndarray
        Cluster label assignments of shape (n_samples,).

    Returns
    -------
    dict with "ari" and "nmi" keys.
    """
    return {
        "ari": adjusted_rand_score(labels_a, labels_b),
        "nmi": normalized_mutual_info_score(labels_a, labels_b),
    }


def compare_variants_at_rho(
    sweep_results: dict[str, SweepResult],
    rho: float,
) -> dict[tuple[str, str], dict[str, float]]:
    """Compute pairwise agreement between all variants at a given rho.

    Parameters
    ----------
    sweep_results : dict[str, SweepResult]
        Mapping variant name to sweep results.
    rho : float
        Vigilance value to compare at.

    Returns
    -------
    dict mapping (variant_a, variant_b) to {"ari": float, "nmi": float}.
    """
    # Find labels at this rho for each variant
    variant_labels = {}
    for variant, sweep in sweep_results.items():
        if rho in sweep.rho_values:
            idx = sweep.rho_values.index(rho)
            variant_labels[variant] = sweep.results[idx].labels

    agreements = {}
    for (va, la), (vb, lb) in combinations(variant_labels.items(), 2):
        agreements[(va, vb)] = compute_agreement(la, lb)

    return agreements


def compare_all_rho(
    sweep_results: dict[str, SweepResult],
) -> dict[float, dict[tuple[str, str], dict[str, float]]]:
    """Compute pairwise agreements at all rho values.

    Returns
    -------
    dict mapping rho to pairwise agreement dicts.
    """
    # Get union of all rho values
    all_rho = set()
    for sweep in sweep_results.values():
        all_rho.update(sweep.rho_values)

    results = {}
    for rho in sorted(all_rho):
        agreements = compare_variants_at_rho(sweep_results, rho)
        if agreements:
            results[rho] = agreements
            for (va, vb), metrics in agreements.items():
                logger.info(
                    f"rho={rho}: {va} vs {vb} — ARI={metrics['ari']:.3f}, NMI={metrics['nmi']:.3f}"
                )

    return results


def agreement_summary_table(
    all_agreements: dict[float, dict[tuple[str, str], dict[str, float]]],
) -> np.ndarray:
    """Build summary array of shape (n_rho, n_pairs, 2) with [ARI, NMI].

    Returns tuple of (table, rho_values, pair_names).
    """
    rho_values = sorted(all_agreements.keys())
    # Get pair names from first rho
    first = all_agreements[rho_values[0]]
    pair_names = sorted(first.keys())

    table = np.zeros((len(rho_values), len(pair_names), 2))
    for i, rho in enumerate(rho_values):
        for j, pair in enumerate(pair_names):
            if pair in all_agreements[rho]:
                table[i, j, 0] = all_agreements[rho][pair]["ari"]
                table[i, j, 1] = all_agreements[rho][pair]["nmi"]

    return table, rho_values, pair_names
