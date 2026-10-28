"""Heuristic detectors for known attention head types.

Uses feature thresholds to create "ground truth" labels for validation.
"""

import logging

import numpy as np

from ..extraction.feature_computer import FEATURE_NAMES

logger = logging.getLogger(__name__)

# Feature indices
F_BOS_SINK = FEATURE_NAMES.index("bos_sink_score")
F_INDUCTION = FEATURE_NAMES.index("induction_score")
F_PREV_TOKEN = FEATURE_NAMES.index("previous_token")
F_POS_CONSISTENCY = FEATURE_NAMES.index("positional_consistency")
F_ENTROPY = FEATURE_NAMES.index("entropy")
F_SELF_ATTN = FEATURE_NAMES.index("self_attention")
F_LOCAL_WINDOW = FEATURE_NAMES.index("local_window")

# Thresholds (on normalized [0,1] features)
THRESHOLDS = {
    "sink": (F_BOS_SINK, 0.3),
    "induction": (F_INDUCTION, 0.5),
    "previous_token": (F_PREV_TOKEN, 0.4),
    "positional": (F_POS_CONSISTENCY, 0.7),
}

# Labels: 0 = unknown/other, 1+ = known types
LABEL_MAP = {
    "other": 0,
    "sink": 1,
    "induction": 2,
    "previous_token": 3,
    "positional": 4,
}


def classify_heads_heuristic(features: np.ndarray) -> tuple[np.ndarray, dict[str, list[int]]]:
    """Assign heuristic head-type labels based on feature thresholds.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (n_heads, n_features), values in [0, 1].

    Returns
    -------
    tuple[np.ndarray, dict[str, list[int]]]
        - labels: int array of shape (n_heads,), values from LABEL_MAP
        - groups: dict mapping head type name to list of head indices
    """
    n_heads = features.shape[0]
    labels = np.zeros(n_heads, dtype=int)
    groups: dict[str, list[int]] = {name: [] for name in LABEL_MAP}

    for head_idx in range(n_heads):
        head_features = features[head_idx]
        assigned = False

        # Priority order: induction > sink > previous_token > positional
        for head_type in ["induction", "sink", "previous_token", "positional"]:
            feat_idx, threshold = THRESHOLDS[head_type]
            if head_features[feat_idx] > threshold:
                labels[head_idx] = LABEL_MAP[head_type]
                groups[head_type].append(head_idx)
                assigned = True
                break

        if not assigned:
            groups["other"].append(head_idx)

    for head_type, indices in groups.items():
        if indices:
            logger.info(f"  {head_type}: {len(indices)} heads")

    return labels, groups


def compute_heuristic_agreement(
    art_labels: np.ndarray,
    heuristic_labels: np.ndarray,
) -> dict[str, float]:
    """Compute agreement between ART labels and heuristic labels.

    Only considers heads that have a non-zero heuristic label (known types).

    Parameters
    ----------
    art_labels : np.ndarray
        ART cluster assignments.
    heuristic_labels : np.ndarray
        Heuristic type labels.

    Returns
    -------
    dict with "ari", "nmi", "n_known_heads".
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    # Filter to known heads only
    known_mask = heuristic_labels > 0
    n_known = known_mask.sum()

    if n_known < 2:
        logger.warning("Too few known heads for agreement computation")
        return {"ari": 0.0, "nmi": 0.0, "n_known_heads": int(n_known)}

    art_known = art_labels[known_mask]
    heuristic_known = heuristic_labels[known_mask]

    ari = adjusted_rand_score(heuristic_known, art_known)
    nmi = normalized_mutual_info_score(heuristic_known, art_known)

    logger.info(f"Heuristic agreement (n={n_known}): ARI={ari:.3f}, NMI={nmi:.3f}")
    return {"ari": ari, "nmi": nmi, "n_known_heads": int(n_known)}
