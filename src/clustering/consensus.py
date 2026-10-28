"""Consensus clustering for ART: run N times, build co-association matrix, cluster it.

Addresses ordering sensitivity (Round 1 critical failure: pairwise ARI = 0.07-0.11).
The co-association matrix C[i,j] = fraction of N runs where samples i,j share a cluster.
Agglomerative clustering on distance=1-C yields an order-independent result.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

from ..utils.reproducibility import get_rng
from .art_runner import run_art

logger = logging.getLogger(__name__)


@dataclass
class ConsensusResult:
    """Result of consensus ART clustering."""
    variant: str
    rho: float
    n_runs: int
    linkage_method: str
    n_clusters: int
    labels: np.ndarray
    coassoc_matrix: np.ndarray  # (n_samples, n_samples) co-association matrix
    per_run_n_clusters: list[int]
    per_run_labels: list[np.ndarray]
    linkage_matrix: np.ndarray  # scipy linkage matrix for dendrogram
    metadata: dict = field(default_factory=dict)


def build_coassociation_matrix(
    all_labels: list[np.ndarray],
) -> np.ndarray:
    """Build co-association matrix from multiple clustering runs.

    C[i,j] = fraction of runs where samples i and j were assigned to the same cluster.

    Parameters
    ----------
    all_labels : list[np.ndarray]
        List of label arrays, each of shape (n_samples,).

    Returns
    -------
    np.ndarray
        Co-association matrix of shape (n_samples, n_samples), values in [0, 1].
    """
    n_runs = len(all_labels)
    n_samples = len(all_labels[0])
    C = np.zeros((n_samples, n_samples), dtype=np.float64)

    for labels in all_labels:
        # Vectorized: C[i,j] += 1 if labels[i] == labels[j]
        same_cluster = labels[:, None] == labels[None, :]
        C += same_cluster

    C /= n_runs
    return C


def consensus_art(
    X: np.ndarray,
    variant: str,
    rho: float,
    n_runs: int = 50,
    linkage_method: str = "average",
    n_clusters_target: int | None = None,
    max_iter: int = 1,
    seed: int = 42,
    model_params: dict | None = None,
) -> ConsensusResult:
    """Run consensus ART clustering.

    1. Run ART n_runs times with different random orderings
    2. Build co-association matrix
    3. Apply agglomerative clustering to distance = 1 - C

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features), values in [0, 1].
    variant : str
        ART variant name.
    rho : float
        Vigilance parameter for individual ART runs.
    n_runs : int
        Number of ART runs with different orderings.
    linkage_method : str
        Linkage method for agglomerative clustering ("average", "complete", "ward").
        "average" recommended for co-association distances.
    n_clusters_target : int, optional
        Target number of clusters. If None, uses the median n_clusters from ART runs.
    max_iter : int
        Number of ART epochs per run.
    seed : int
        Random seed.

    Returns
    -------
    ConsensusResult
        Consensus clustering outcome.
    """
    rng = get_rng(seed)
    n_samples = X.shape[0]

    # Step 1: Run ART n_runs times with different orderings
    all_labels = []
    all_n_clusters = []

    for i in range(n_runs):
        shuffle = rng.permutation(n_samples)
        result = run_art(X, variant, rho, shuffle_order=shuffle, max_iter=max_iter, model_params=model_params)
        all_labels.append(result.labels)
        all_n_clusters.append(result.n_clusters)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  Consensus run {i+1}/{n_runs}: "
                f"n_clusters range [{min(all_n_clusters)}-{max(all_n_clusters)}]"
            )

    # Step 2: Build co-association matrix
    C = build_coassociation_matrix(all_labels)
    logger.info(
        f"Co-association matrix built: "
        f"mean={C.mean():.3f}, min_off_diag={C[np.triu_indices_from(C, k=1)].min():.3f}, "
        f"max_off_diag={C[np.triu_indices_from(C, k=1)].max():.3f}"
    )

    # Step 3: Agglomerative clustering on distance = 1 - C
    distance_matrix = 1.0 - C
    np.fill_diagonal(distance_matrix, 0.0)  # ensure zero self-distance
    distance_matrix = np.maximum(distance_matrix, 0.0)  # numerical safety

    # Convert to condensed form for scipy
    condensed_dist = squareform(distance_matrix, checks=False)

    # Linkage
    Z = linkage(condensed_dist, method=linkage_method)

    # Determine n_clusters
    if n_clusters_target is None:
        n_clusters_target = int(np.median(all_n_clusters))
        logger.info(
            f"Auto n_clusters from median of ART runs: {n_clusters_target} "
            f"(range [{min(all_n_clusters)}-{max(all_n_clusters)}])"
        )

    # Cut dendrogram
    labels = fcluster(Z, t=n_clusters_target, criterion="maxclust") - 1  # 0-indexed

    result = ConsensusResult(
        variant=variant,
        rho=rho,
        n_runs=n_runs,
        linkage_method=linkage_method,
        n_clusters=len(np.unique(labels)),
        labels=labels,
        coassoc_matrix=C,
        per_run_n_clusters=all_n_clusters,
        per_run_labels=all_labels,
        linkage_matrix=Z,
        metadata={
            "max_iter": max_iter,
            "seed": seed,
            "n_clusters_target": n_clusters_target,
            "n_clusters_median": int(np.median(all_n_clusters)),
            "n_clusters_range": [int(min(all_n_clusters)), int(max(all_n_clusters))],
        },
    )

    logger.info(
        f"Consensus {variant} rho={rho}: {result.n_clusters} clusters "
        f"(from {n_runs} runs, median ART k={int(np.median(all_n_clusters))})"
    )
    return result


def consensus_stability(
    X: np.ndarray,
    variant: str,
    rho: float,
    n_runs: int = 50,
    n_repeats: int = 5,
    linkage_method: str = "average",
    seed: int = 42,
) -> dict[str, float]:
    """Measure stability of consensus itself by repeating with different seeds.

    This should yield high pairwise ARI since consensus is designed to be
    order-independent. Any remaining instability comes from finite N sampling.

    Parameters
    ----------
    n_repeats : int
        Number of independent consensus runs (each with n_runs ART runs).

    Returns
    -------
    dict with pairwise ARI statistics.
    """
    rng = get_rng(seed)
    all_labels = []

    for rep in range(n_repeats):
        rep_seed = int(rng.integers(0, 2**31))
        result = consensus_art(
            X, variant, rho,
            n_runs=n_runs,
            linkage_method=linkage_method,
            seed=rep_seed,
        )
        all_labels.append(result.labels)

    # Pairwise ARI
    aris = []
    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            aris.append(adjusted_rand_score(all_labels[i], all_labels[j]))

    aris = np.array(aris)
    summary = {
        "pairwise_ari_mean": float(aris.mean()),
        "pairwise_ari_std": float(aris.std()),
        "pairwise_ari_min": float(aris.min()),
        "n_repeats": n_repeats,
        "n_runs_per_repeat": n_runs,
    }
    logger.info(
        f"Consensus stability ({variant}, rho={rho}): "
        f"pairwise ARI={summary['pairwise_ari_mean']:.3f}±{summary['pairwise_ari_std']:.3f}"
    )
    return summary


def compare_consensus_vs_single(
    X: np.ndarray,
    variant: str,
    rho: float,
    consensus_labels: np.ndarray,
    n_single_runs: int = 10,
    seed: int = 42,
) -> dict[str, float]:
    """Compare consensus labels with individual ART runs.

    Shows how much consensus differs from any single run, and how much
    single runs differ from each other (reproducing Round 1 findings).
    """
    rng = get_rng(seed)
    single_vs_consensus = []
    single_vs_single = []
    all_single_labels = []

    for i in range(n_single_runs):
        shuffle = rng.permutation(len(X))
        result = run_art(X, variant, rho, shuffle_order=shuffle)
        all_single_labels.append(result.labels)
        single_vs_consensus.append(
            adjusted_rand_score(consensus_labels, result.labels)
        )

    for i in range(len(all_single_labels)):
        for j in range(i + 1, len(all_single_labels)):
            single_vs_single.append(
                adjusted_rand_score(all_single_labels[i], all_single_labels[j])
            )

    summary = {
        "single_vs_consensus_mean": float(np.mean(single_vs_consensus)),
        "single_vs_consensus_std": float(np.std(single_vs_consensus)),
        "single_vs_single_mean": float(np.mean(single_vs_single)),
        "single_vs_single_std": float(np.std(single_vs_single)),
    }
    logger.info(
        f"Single vs consensus ARI: {summary['single_vs_consensus_mean']:.3f}±{summary['single_vs_consensus_std']:.3f} "
        f"| Single vs single ARI: {summary['single_vs_single_mean']:.3f}±{summary['single_vs_single_std']:.3f}"
    )
    return summary
