"""Batched forward passes with GPU memory management."""

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .attention_hooks import AttentionCapture
from ..utils.config_loader import load_config

logger = logging.getLogger(__name__)


def run_batched_extraction(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    batch_size: Optional[int] = None,
    max_seq_len: Optional[int] = None,
) -> list[dict[int, torch.Tensor]]:
    """Run batched forward passes and collect attention weights.

    Parameters
    ----------
    model : AutoModelForCausalLM
        Model with eager attention.
    tokenizer : AutoTokenizer
        Tokenizer for the model.
    prompts : list[str]
        Input prompts.
    batch_size : int, optional
        Batch size. Defaults to config value.
    max_seq_len : int, optional
        Maximum sequence length. Defaults to config value.

    Returns
    -------
    list[dict[int, torch.Tensor]]
        Per-input attention weights. Each dict maps layer_idx to
        tensor of shape (1, n_heads, seq_len, seq_len).
    """
    model_cfg = load_config("model")
    dataset_cfg = load_config("dataset")
    batch_size = batch_size or dataset_cfg.get("batch_size", 4)
    max_seq_len = max_seq_len or model_cfg.get("max_seq_len", 512)
    n_layers = model_cfg.get("n_layers", 16)
    device = next(model.parameters()).device

    capture = AttentionCapture(model, n_layers=n_layers)
    capture.register()

    all_attentions: list[dict[int, torch.Tensor]] = []

    try:
        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start:batch_start + batch_size]
            actual_batch_size = len(batch_prompts)

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            ).to(device)

            capture.clear()
            with torch.no_grad():
                model(**inputs)

            attn_dict = capture.get()

            # Split batch dimension into individual samples
            attention_mask = inputs.get("attention_mask")
            for sample_idx in range(actual_batch_size):
                sample_attn = {}
                if attention_mask is not None:
                    seq_len = attention_mask[sample_idx].sum().item()
                else:
                    seq_len = inputs["input_ids"].shape[1]

                for layer_idx, attn_tensor in attn_dict.items():
                    # Slice to actual (non-padded) sequence length
                    sample_attn[layer_idx] = attn_tensor[
                        sample_idx:sample_idx + 1, :, :seq_len, :seq_len
                    ]
                all_attentions.append(sample_attn)

            if (batch_start // batch_size) % 10 == 0:
                n_done = min(batch_start + batch_size, len(prompts))
                logger.info(f"Processed {n_done}/{len(prompts)} prompts")

            # Free GPU memory
            del inputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        capture.remove()

    logger.info(f"Extraction complete: {len(all_attentions)} samples")
    return all_attentions
