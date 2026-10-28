#!/usr/bin/env python3
"""Step 1: Load model, run prompts, compute and cache features.

Computes features incrementally per-batch to avoid storing all attention
weights in memory (which would be ~630 GB for 600 prompts).

Usage:
    python scripts/01_extract_features.py [--tag base] [--n-prompts 600] [--device cuda]
"""

import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config, get_results_dir
from src.utils.reproducibility import set_seed
from src.extraction.model_loader import load_model
from src.extraction.attention_hooks import AttentionCapture
from src.extraction.feature_computer import compute_features_single, N_FEATURES
from src.extraction.cache_manager import save_features, features_exist

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_dataset(n_prompts: int | None = None) -> tuple[list[str], list[dict]]:
    """Build the prompt dataset from configured sources."""
    dataset_cfg = load_config("dataset")
    max_seq_len = dataset_cfg["max_seq_len"]

    prompts = []
    metadata = []

    for cat_name, cat_cfg in dataset_cfg["categories"].items():
        n = cat_cfg["n"]
        source = cat_cfg["source"]

        if source == "synthetic":
            cat_prompts = _generate_synthetic(cat_cfg["generator"], n)
        else:
            cat_prompts = _load_from_hf(source, cat_cfg, n)

        for p in cat_prompts:
            p_truncated = p[:max_seq_len * 6]
            prompts.append(p_truncated)
            metadata.append({
                "source": source,
                "category": cat_name,
                "hash": hashlib.md5(p_truncated.encode()).hexdigest()[:12],
            })

    if n_prompts and n_prompts < len(prompts):
        rng = random.Random(42)
        indices = rng.sample(range(len(prompts)), n_prompts)
        prompts = [prompts[i] for i in indices]
        metadata = [metadata[i] for i in indices]

    logger.info(f"Built dataset: {len(prompts)} prompts")
    return prompts, metadata


def _load_from_hf(source: str, cfg: dict, n: int) -> list[str]:
    """Load prompts from a HuggingFace dataset using streaming to avoid full downloads."""
    from datasets import load_dataset

    kwargs = {"split": cfg.get("split", "validation")}
    if "subset" in cfg:
        kwargs["name"] = cfg["subset"]

    try:
        ds = load_dataset(source, **kwargs, streaming=True)
    except Exception as e:
        logger.warning(f"Failed to load {source} with streaming: {e}")
        try:
            ds = load_dataset(source, **kwargs)
        except Exception as e2:
            logger.warning(f"Failed to load {source}: {e2}. Using placeholder prompts.")
            return [f"Placeholder prompt {i} for {source}" for i in range(n)]

    text_fields = ["text", "question", "prompt", "ctx", "activity_label", "input"]
    texts = []
    for item in ds:
        for field in text_fields:
            if field in item and isinstance(item[field], str) and len(item[field]) > 20:
                texts.append(item[field])
                break
        if len(texts) >= n:
            break

    if len(texts) < n:
        logger.warning(f"Only got {len(texts)}/{n} prompts from {source}")

    return texts[:n]


def _generate_synthetic(generator: str, n: int) -> list[str]:
    """Generate synthetic prompts."""
    rng = random.Random(42)

    if generator == "repeated_bigrams":
        vocab = ["the", "cat", "sat", "on", "mat", "dog", "ran", "big", "red", "fox",
                 "quick", "brown", "lazy", "jumped", "over", "fence", "and", "then",
                 "went", "home", "after", "long", "day", "of", "play", "in", "park"]
        prompts = []
        for _ in range(n):
            bigrams = [(rng.choice(vocab), rng.choice(vocab)) for _ in range(5)]
            tokens = []
            for _ in range(20):
                for a, b in bigrams:
                    tokens.extend([a, b])
            prompts.append(" ".join(tokens))
        return prompts

    elif generator == "facts_and_questions":
        names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank"]
        cities = ["Paris", "London", "Tokyo", "Berlin", "Rome", "Sydney", "Cairo", "Oslo"]
        jobs = ["doctor", "teacher", "engineer", "artist", "chef", "pilot", "writer", "farmer"]
        prompts = []
        for _ in range(n):
            facts = []
            for _ in range(8):
                name = rng.choice(names)
                city = rng.choice(cities)
                job = rng.choice(jobs)
                facts.append(f"{name} is a {job} who lives in {city}.")
            rng.shuffle(facts)
            padding = " ".join(rng.choices(
                ["Meanwhile", "Furthermore", "In addition", "Also", "Moreover"],
                k=10
            ))
            text = " ".join(facts) + " " + padding + " " + " ".join(facts[:3])
            prompts.append(text)
        return prompts

    else:
        raise ValueError(f"Unknown synthetic generator: {generator}")


def extract_and_compute(model, tokenizer, prompts, n_layers, n_heads, max_seq_len, batch_size):
    """Run extraction and compute features incrementally per-batch.

    Instead of storing all attention weights (~1 GB per prompt), compute
    per-input features immediately and discard the raw attention.

    Returns per_input features of shape (n_prompts, n_layers, n_heads, n_features).
    """
    device = next(model.parameters()).device
    n_prompts = len(prompts)
    per_input = np.zeros((n_prompts, n_layers, n_heads, N_FEATURES), dtype=np.float32)

    capture = AttentionCapture(model, n_layers=n_layers)
    capture.register()

    try:
        for batch_start in range(0, n_prompts, batch_size):
            batch_prompts = prompts[batch_start:batch_start + batch_size]
            actual_batch_size = len(batch_prompts)

            # Tokenize
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            ).to(device)

            # Forward pass
            capture.clear()
            with torch.no_grad():
                model(**inputs)

            attn_dict = capture.get()
            attention_mask = inputs.get("attention_mask")

            # Get input IDs for this batch
            batch_input_ids = []
            for prompt in batch_prompts:
                ids = tokenizer.encode(prompt, truncation=True, max_length=max_seq_len)
                batch_input_ids.append(np.array(ids))

            # Compute features for each sample in batch immediately
            for sample_idx in range(actual_batch_size):
                prompt_idx = batch_start + sample_idx

                if attention_mask is not None:
                    seq_len = attention_mask[sample_idx].sum().item()
                else:
                    seq_len = inputs["input_ids"].shape[1]

                input_ids = batch_input_ids[sample_idx]

                for layer_idx in range(n_layers):
                    attn_tensor = attn_dict.get(layer_idx)
                    if attn_tensor is None:
                        continue

                    # Slice to this sample's actual (non-padded) sequence length
                    sample_attn = attn_tensor[sample_idx, :, :seq_len, :seq_len]
                    if isinstance(sample_attn, torch.Tensor):
                        sample_attn = sample_attn.numpy()

                    for head_idx in range(n_heads):
                        attn_matrix = sample_attn[head_idx].astype(np.float64)
                        per_input[prompt_idx, layer_idx, head_idx] = compute_features_single(
                            attn_matrix, input_ids[:seq_len]
                        )

            # Free memory
            del inputs, attn_dict
            if device.type == "cuda":
                torch.cuda.empty_cache()

            n_done = min(batch_start + batch_size, n_prompts)
            logger.info(f"Processed {n_done}/{n_prompts} prompts (features computed inline)")

    finally:
        capture.remove()

    return per_input


def main():
    parser = argparse.ArgumentParser(description="Extract attention features")
    parser.add_argument("--tag", default="base", help="Tag for this extraction run")
    parser.add_argument("--n-prompts", type=int, default=None, help="Limit number of prompts")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument("--model-id", default=None, help="Override model ID")
    parser.add_argument("--force", action="store_true", help="Overwrite cached features")
    args = parser.parse_args()

    set_seed(42)
    model_cfg = load_config("model")
    dataset_cfg = load_config("dataset")

    if features_exist(args.tag) and not args.force:
        logger.info(f"Features already cached for tag '{args.tag}'. Use --force to recompute.")
        return

    # Build dataset
    prompts, meta = build_dataset(args.n_prompts)

    # Save dataset metadata
    meta_path = get_results_dir("features") / f"dataset_{args.tag}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved dataset metadata to {meta_path}")

    # Load model
    model, tokenizer = load_model(model_id=args.model_id, device=args.device)

    # Extract and compute features incrementally
    batch_size = dataset_cfg.get("batch_size", 4)
    per_input = extract_and_compute(
        model, tokenizer, prompts,
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        max_seq_len=model_cfg["max_seq_len"],
        batch_size=batch_size,
    )

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Aggregate: mean across prompts, then min-max normalize per feature
    aggregated = np.mean(per_input, axis=0)  # (n_layers, n_heads, n_features)
    aggregated_flat = aggregated.reshape(model_cfg["n_layers"] * model_cfg["n_heads"], N_FEATURES)

    for f in range(N_FEATURES):
        col = aggregated_flat[:, f]
        fmin, fmax = col.min(), col.max()
        if fmax - fmin > 1e-8:
            aggregated_flat[:, f] = (col - fmin) / (fmax - fmin)
        else:
            aggregated_flat[:, f] = 0.5

    # Cache
    save_features(
        per_input, aggregated_flat, tag=args.tag,
        metadata={"n_prompts": len(prompts), "model_id": model_cfg["model_id"]},
    )

    logger.info("Feature extraction complete!")
    logger.info(f"  Per-input shape: {per_input.shape}")
    logger.info(f"  Aggregated shape: {aggregated_flat.shape}")
    logger.info(f"  Feature ranges: min={aggregated_flat.min():.3f}, max={aggregated_flat.max():.3f}")


if __name__ == "__main__":
    main()
