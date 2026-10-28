"""Compute 14 statistical features from attention matrices.

Input: attention matrix A of shape (seq_len, seq_len) per head per input.
  A[i,j] = attention from position i to position j.
  Lower-triangular (causal mask). Rows sum to 1.

Output per head: 14D feature vector (averaged across inputs), all in [0,1].

Vectorized implementation — ~20x faster than loop-based version.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

N_FEATURES = 14
FEATURE_NAMES = [
    "entropy", "max_attention", "top5_concentration", "bos_sink_score",
    "previous_token", "self_attention", "mean_distance", "distance_std",
    "local_window", "induction_score", "copy_score", "kurtosis",
    "positional_consistency", "causal_decay_rate",
]


def compute_features_single(
    attn: np.ndarray,
    input_ids: np.ndarray,
) -> np.ndarray:
    """Compute 14 features for one head on one input (vectorized).

    Parameters
    ----------
    attn : np.ndarray
        Attention matrix of shape (seq_len, seq_len). Causal (lower-triangular).
    input_ids : np.ndarray
        Token IDs of shape (seq_len,). Needed for induction/copy scores.

    Returns
    -------
    np.ndarray
        Feature vector of shape (14,), values in [0, 1].
    """
    seq_len = attn.shape[0]
    features = np.zeros(N_FEATURES, dtype=np.float64)

    if seq_len < 3:
        return features.astype(np.float32)

    eps = 1e-10
    i_idx = np.arange(seq_len)

    # Causal mask: j <= i
    causal_mask = np.tri(seq_len, dtype=bool)

    # --- F1: Entropy (vectorized) ---
    attn_clipped = np.clip(attn, eps, 1.0)
    row_entropy = -np.sum(attn * np.log2(attn_clipped) * causal_mask, axis=1)
    max_entropy = np.log2(i_idx + 1.0)
    max_entropy[0] = 1.0  # avoid /0
    features[0] = np.mean(row_entropy[1:] / max_entropy[1:])

    # --- F2: Max attention (vectorized) ---
    attn_for_max = np.where(causal_mask, attn, -np.inf)
    features[1] = np.mean(np.max(attn_for_max, axis=1))

    # --- F3: Top-5 concentration (vectorized) ---
    attn_sorted = np.sort(np.where(causal_mask, attn, 0.0), axis=1)
    features[2] = np.mean(np.sum(attn_sorted[:, -5:], axis=1))

    # --- F4: BOS/sink score ---
    features[3] = np.mean(attn[1:, 0])

    # --- F5: Previous-token score (subdiagonal) ---
    features[4] = np.mean(np.diag(attn, k=-1))

    # --- F6: Self-attention (diagonal) ---
    features[5] = np.mean(np.diag(attn))

    # --- F7: Mean distance (vectorized) ---
    j_idx = np.arange(seq_len)
    distances = np.abs(i_idx[:, None] - j_idx[None, :]).astype(np.float64)
    weighted_dist = np.sum(attn * distances * causal_mask, axis=1)
    norm = np.maximum(i_idx, 1).astype(np.float64)
    features[6] = np.mean(weighted_dist[1:] / norm[1:])

    # --- F8: Distance std (vectorized) ---
    mean_d = weighted_dist  # attention-weighted mean distance per position
    var_d = np.sum(attn * (distances - mean_d[:, None]) ** 2 * causal_mask, axis=1)
    std_d = np.sqrt(np.maximum(var_d, 0.0))
    features[7] = np.mean(std_d[2:] / norm[2:])

    # --- F9: Local window w=5 (vectorized) ---
    local_mask = (distances <= 4) & causal_mask
    features[8] = np.mean(np.sum(attn * local_mask, axis=1))

    # --- F10: Induction score ---
    features[9] = _compute_induction_score(attn, input_ids)

    # --- F11: Copy score ---
    features[10] = _compute_copy_score(attn, input_ids)

    # --- F12: Kurtosis (vectorized) ---
    features[11] = _compute_kurtosis(attn, causal_mask, seq_len)

    # --- F13: Positional consistency (vectorized) ---
    features[12] = _compute_positional_consistency(attn, seq_len)

    # --- F14: Causal decay rate (vectorized) ---
    features[13] = _compute_causal_decay(attn, seq_len)

    return np.clip(features, 0.0, 1.0).astype(np.float32)


def _compute_induction_score(attn: np.ndarray, input_ids: np.ndarray) -> float:
    """F10: Attention to token after previous occurrence of predecessor."""
    seq_len = len(input_ids)
    scores = []
    for i in range(2, seq_len):
        predecessor = input_ids[i - 1]
        matches = np.where(input_ids[:i - 1] == predecessor)[0]
        if len(matches) == 0:
            continue
        targets = matches + 1
        targets = targets[targets <= i]
        if len(targets) == 0:
            continue
        scores.append(np.sum(attn[i, targets]))
    return float(np.mean(scores)) if scores else 0.0


def _compute_copy_score(attn: np.ndarray, input_ids: np.ndarray) -> float:
    """F11: Attention to previous occurrences of same token."""
    seq_len = len(input_ids)
    scores = []
    for i in range(1, seq_len):
        matches = np.where(input_ids[:i] == input_ids[i])[0]
        if len(matches) == 0:
            continue
        scores.append(np.sum(attn[i, matches]))
    return float(np.mean(scores)) if scores else 0.0


def _compute_kurtosis(attn: np.ndarray, causal_mask: np.ndarray, seq_len: int) -> float:
    """F12: Kurtosis — sigmoid(excess_kurtosis / 10)."""
    if seq_len < 4:
        return 0.5

    counts = np.arange(1, seq_len + 1, dtype=np.float64)
    row_means = 1.0 / counts

    attn_sq = attn ** 2
    e_x2 = np.sum(attn_sq * causal_mask, axis=1)
    row_var = e_x2 - row_means ** 2
    row_std = np.sqrt(np.maximum(row_var, 1e-20))

    valid = (counts >= 4) & (row_std > 1e-10)
    if not np.any(valid):
        return 0.5

    kurt_vals = []
    for i in np.where(valid)[0]:
        row = attn[i, :i + 1]
        m = row_means[i]
        s = row_std[i]
        z4 = np.mean(((row - m) / s) ** 4) - 3.0
        if np.isfinite(z4):
            kurt_vals.append(z4)

    if kurt_vals:
        mean_kurt = np.mean(kurt_vals)
        if np.isfinite(mean_kurt):
            return 1.0 / (1.0 + np.exp(-mean_kurt / 10.0))
    return 0.5


def _compute_positional_consistency(attn: np.ndarray, seq_len: int) -> float:
    """F13: Average pairwise correlation of relative attention profiles.

    Vectorized using diagonal extraction and matrix correlation.
    """
    max_dist = min(32, seq_len)
    n_profiles = seq_len - max_dist
    if n_profiles < 2:
        return 0.0

    profiles = np.zeros((n_profiles, max_dist))
    for d in range(max_dist):
        diag = np.diag(attn, k=-d)
        start = max_dist - d
        profiles[:, d] = diag[start:start + n_profiles]

    if n_profiles > 50:
        indices = np.linspace(0, n_profiles - 1, 50, dtype=int)
        profiles = profiles[indices]

    profiles_centered = profiles - profiles.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(profiles_centered, axis=1, keepdims=True)

    valid = (norms.ravel() > 1e-8)
    if np.sum(valid) < 2:
        return 0.0

    profiles_normed = profiles_centered[valid] / norms[valid]
    corr_matrix = profiles_normed @ profiles_normed.T

    n = corr_matrix.shape[0]
    upper_idx = np.triu_indices(n, k=1)
    corrs = corr_matrix[upper_idx]

    return float(np.mean(corrs)) if len(corrs) > 0 else 0.0


def _compute_causal_decay(attn: np.ndarray, seq_len: int) -> float:
    """F14: 1 - exp(-lambda) from exponential fit to avg attention vs distance.

    Vectorized using diagonal extraction instead of O(n^2) loop.
    """
    max_dist = min(64, seq_len)

    dist_attn = np.zeros(max_dist)
    for d in range(max_dist):
        diag = np.diag(attn, k=-d)
        if len(diag) > 0:
            dist_attn[d] = np.mean(diag)

    valid = dist_attn > 0
    distances = np.arange(max_dist)[valid]
    values = dist_attn[valid]

    if len(distances) < 3:
        return 0.5

    try:
        log_values = np.log(values + 1e-10)
        coeffs = np.polyfit(distances, log_values, 1)
        lam = -coeffs[0]
        lam = np.clip(lam, 0, 10)
        return float(1.0 - np.exp(-lam))
    except (np.linalg.LinAlgError, ValueError):
        return 0.5


def compute_features_for_head(
    attentions: list[np.ndarray],
    input_ids_list: list[np.ndarray],
    layer: int,
    head: int,
) -> np.ndarray:
    """Compute features for one head across all inputs, then average."""
    per_input_features = []
    for attn_tensor, ids in zip(attentions, input_ids_list):
        attn_matrix = attn_tensor[0, head].astype(np.float64)
        features = compute_features_single(attn_matrix, ids)
        per_input_features.append(features)

    return np.mean(per_input_features, axis=0).astype(np.float32)


def compute_all_features(
    all_attentions: list[dict[int, np.ndarray]],
    all_input_ids: list[np.ndarray],
    n_layers: int = 16,
    n_heads: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute features for all heads across all inputs.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        per_input: (n_prompts, n_layers, n_heads, n_features)
        aggregated: (n_layers * n_heads, n_features)
    """
    import torch

    n_prompts = len(all_attentions)
    per_input = np.zeros((n_prompts, n_layers, n_heads, N_FEATURES), dtype=np.float32)

    total_heads = n_layers * n_heads
    done = 0

    for layer in range(n_layers):
        layer_attentions = []
        for sample_attn in all_attentions:
            if layer in sample_attn:
                t = sample_attn[layer]
                if isinstance(t, torch.Tensor):
                    t = t.numpy()
                layer_attentions.append(t)
            else:
                layer_attentions.append(None)

        for head in range(n_heads):
            for prompt_idx in range(n_prompts):
                attn_tensor = layer_attentions[prompt_idx]
                if attn_tensor is None:
                    continue
                attn_matrix = attn_tensor[0, head].astype(np.float64)
                ids = all_input_ids[prompt_idx]
                per_input[prompt_idx, layer, head] = compute_features_single(
                    attn_matrix, ids
                )

            done += 1
            if done % 32 == 0:
                logger.info(f"Feature computation: {done}/{total_heads} heads")

    # Aggregate: mean across prompts
    aggregated = np.mean(per_input, axis=0)
    aggregated_flat = aggregated.reshape(n_layers * n_heads, N_FEATURES)

    # Normalize to [0,1] per feature
    for f in range(N_FEATURES):
        col = aggregated_flat[:, f]
        fmin, fmax = col.min(), col.max()
        if fmax - fmin > 1e-8:
            aggregated_flat[:, f] = (col - fmin) / (fmax - fmin)
        else:
            aggregated_flat[:, f] = 0.5

    logger.info(
        f"Feature computation complete: per_input {per_input.shape}, "
        f"aggregated {aggregated_flat.shape}"
    )
    return per_input, aggregated_flat
