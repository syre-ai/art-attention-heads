"""Forward hooks on self_attn to capture attention weights."""

from typing import Optional

import torch
import torch.nn as nn


class AttentionCapture:
    """Captures attention weights via forward hooks on model attention layers.

    Usage
    -----
    >>> capture = AttentionCapture(model, n_layers=16)
    >>> capture.register()
    >>> with torch.no_grad():
    ...     model(**inputs)
    >>> attn = capture.get()  # dict[int, Tensor(batch, heads, seq, seq)]
    >>> capture.clear()
    >>> # When done:
    >>> capture.remove()
    """

    def __init__(self, model: nn.Module, n_layers: int = 16):
        self.model = model
        self.n_layers = n_layers
        self._storage: dict[int, torch.Tensor] = {}
        self._hooks: list = []

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # OLMo-2 Attention forward returns (attn_output, attn_weights, past_kv)
            # attn_weights shape: (batch, n_heads, seq_len, seq_len)
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    self._storage[layer_idx] = attn_weights.detach().cpu()
        return hook_fn

    def register(self) -> None:
        """Register forward hooks on all attention layers."""
        self.remove()  # Clear any existing hooks
        for layer_idx in range(self.n_layers):
            attn_module = self.model.model.layers[layer_idx].self_attn
            hook = attn_module.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)

    def remove(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def clear(self) -> None:
        """Clear stored attention weights (free memory)."""
        self._storage.clear()

    def get(self) -> dict[int, torch.Tensor]:
        """Return captured attention weights.

        Returns
        -------
        dict[int, torch.Tensor]
            Mapping from layer index to attention weights tensor
            of shape (batch, n_heads, seq_len, seq_len).
        """
        return dict(self._storage)

    def get_stacked(self) -> Optional[torch.Tensor]:
        """Return attention weights stacked as (n_layers, batch, n_heads, seq, seq).

        Returns None if no weights were captured.
        """
        if not self._storage:
            return None
        layers = sorted(self._storage.keys())
        return torch.stack([self._storage[l] for l in layers], dim=0)
