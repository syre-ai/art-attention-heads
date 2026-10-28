"""Load OLMo-2 with eager attention, handle checkpoints."""

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.config_loader import load_config

logger = logging.getLogger(__name__)


def load_model(
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    revision: Optional[str] = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer with eager attention (required for attention weights).

    Parameters
    ----------
    model_id : str, optional
        HuggingFace model ID. Defaults to config value.
    device : str, optional
        Device to load model on. Defaults to "cuda" if available.

    Returns
    -------
    tuple[AutoModelForCausalLM, AutoTokenizer]
        Model and tokenizer.
    """
    cfg = load_config("model")
    model_id = model_id or cfg["model_id"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(cfg.get("dtype", "float16"), torch.float16)

    # Some OLMo-2 checkpoints only have safetensors in PR branches
    # Auto-detect: if model_id is a known checkpoint without safetensors on main, use PR revision
    safetensors_pr_revisions = {
        "allenai/OLMo-2-0425-1B-SFT": "refs/pr/1",
        "allenai/OLMo-2-0425-1B-DPO": "refs/pr/1",
        "allenai/OLMo-2-0425-1B-Instruct": "refs/pr/1",
    }
    if revision is None and model_id in safetensors_pr_revisions:
        revision = safetensors_pr_revisions[model_id]
        logger.info(f"Using safetensors PR revision: {revision}")

    logger.info(f"Loading {model_id} with eager attention on {device} ({dtype})")

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation=cfg.get("attn_implementation", "eager"),
        output_attentions=True,
        revision=revision,
    )
    model = model.to(device)
    model.eval()

    logger.info(
        f"Loaded: {cfg['n_layers']} layers, {cfg['n_heads']} heads, "
        f"{cfg['hidden_dim']} hidden dim"
    )
    return model, tokenizer
