# kv_cache.py
"""
KV Cache management for the progressive Transformer course.

Usage pattern inside attention:

    present = cache.update(layer_idx, new_k, new_v)   # returns the full K/V so far
    # or
    k, v = cache.get(layer_idx)
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any
import torch
from torch import Tensor, nn


class KVCache:
    """
    Manages key/value tensors for every layer.

    Internally stores a list of (K, V) pairs, one per layer.
    Shape convention (always):
        K, V : (batch_size, seq_len, n_heads, head_dim)
        (single-head is just n_heads=1)
    """

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        preallocate: bool = True,
    ):
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.preallocate = preallocate

        # Current length of the cache (number of tokens already stored)
        self.seq_len = 0

        if preallocate:
            # Pre-allocate the full buffer → much faster decode
            self.k_cache = [
                torch.zeros(
                    batch_size, max_seq_len, n_heads, head_dim,
                    device=device, dtype=dtype
                )
                for _ in range(n_layers)
            ]
            self.v_cache = [
                torch.zeros(
                    batch_size, max_seq_len, n_heads, head_dim,
                    device=device, dtype=dtype
                )
                for _ in range(n_layers)
            ]
        else:
            # Dynamic growth (simpler, slightly slower)
            self.k_cache: List[Optional[Tensor]] = [None] * n_layers
            self.v_cache: List[Optional[Tensor]] = [None] * n_layers

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update(
        self,
        layer_idx: int,
        new_k: Tensor,
        new_v: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Append new_k / new_v (the tokens just computed) to the cache
        and return the *full* key/value tensors that attention should use.

        new_k, new_v shape: (batch, new_len, n_heads, head_dim)
        """
        assert 0 <= layer_idx < self.n_layers
        new_len = new_k.shape[1]
        assert new_k.shape == new_v.shape
        assert new_k.shape[0] == self.batch_size
        assert new_k.shape[2] == self.n_heads
        assert new_k.shape[3] == self.head_dim

        if self.preallocate:
            start = self.seq_len
            end = start + new_len
            if end > self.max_seq_len:
                raise RuntimeError(
                    f"KV cache overflow: tried to write up to {end}, "
                    f"max_seq_len={self.max_seq_len}"
                )
            self.k_cache[layer_idx][:, start:end] = new_k
            self.v_cache[layer_idx][:, start:end] = new_v

            # Return a view of everything written so far
            full_k = self.k_cache[layer_idx][:, :end]
            full_v = self.v_cache[layer_idx][:, :end]
        else:
            # Dynamic path
            if self.k_cache[layer_idx] is None:
                self.k_cache[layer_idx] = new_k
                self.v_cache[layer_idx] = new_v
            else:
                self.k_cache[layer_idx] = torch.cat(
                    [self.k_cache[layer_idx], new_k], dim=1
                )
                self.v_cache[layer_idx] = torch.cat(
                    [self.v_cache[layer_idx], new_v], dim=1
                )
            full_k = self.k_cache[layer_idx]
            full_v = self.v_cache[layer_idx]

        # Only the first layer that is updated advances the global seq_len.
        # (All layers receive the same number of new tokens.)
        if layer_idx == 0:
            self.seq_len += new_len

        return full_k, full_v

    def get(self, layer_idx: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Return the current full K/V for a layer (or None if empty)."""
        if self.preallocate:
            if self.seq_len == 0:
                return None, None
            return (
                self.k_cache[layer_idx][:, : self.seq_len],
                self.v_cache[layer_idx][:, : self.seq_len],
            )
        else:
            return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def reset(self):
        """Clear the cache (keeps pre-allocated buffers)."""
        self.seq_len = 0
        if not self.preallocate:
            self.k_cache = [None] * self.n_layers
            self.v_cache = [None] * self.n_layers
        # For the pre-allocated path we simply set seq_len=0;
        # the buffers stay allocated.

    def __len__(self) -> int:
        return self.seq_len

    def __repr__(self) -> str:
        return (
            f"KVCache(n_layers={self.n_layers}, seq_len={self.seq_len}, "
            f"max_seq_len={self.max_seq_len}, preallocate={self.preallocate})"
        )


# ----------------------------------------------------------------------
# Convenience factory used by the model
# ----------------------------------------------------------------------

def build_kv_cache(
    config: Dict[str, Any],
    batch_size: int,
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    preallocate: bool = True,
) -> KVCache:
    """
    Create a KVCache from a MODEL_CONFIG dictionary.

    Expected keys (with sensible defaults for early course levels):
        n_layers, n_heads, head_dim  (or d_model / n_heads)
    """
    n_layers = config.get("n_layers", 1)
    n_heads  = config.get("n_heads", 1)
    if "head_dim" in config:
        head_dim = config["head_dim"]
    else:
        d_model = config.get("d_model", config.get("embed_dim", 64))
        head_dim = d_model // n_heads

    return KVCache(
        n_layers=n_layers,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        n_heads=n_heads,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
        preallocate=preallocate,
    )