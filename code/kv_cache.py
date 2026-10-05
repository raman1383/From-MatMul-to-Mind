import torch
from typing import Tuple
from model_config import ModelConfig




# TODO: Ring Buffer: turn each lane into a ring buffer, so the oldest token's cache entry is overwritten by the newest.


# @dataclass
class KV_cache:
    """
    Per-lane ring buffer.
      keys/values: [num_layers, num_lanes, num_kv_heads, cache_window, head_dim]
      lengths:     [num_lanes] -> the number of sequences in flight, and finished ones hand their lane to waiting ones

    Token with absolute position a lives at physical index absolute_pos % cache_window.
    """

    def __init__(
        self,
        config: ModelConfig,
        num_lanes: int,
        device: torch.device = 'cuda',
        dtype: torch.dtype = torch.float32,

    ):
        self.cache_window   = config.max_context_window

        shape = (config.num_layers, 
                 num_lanes, 
                 config.num_kv_heads,
                 self.cache_window, 
                 config.head_dim
                )

        self.keys    = torch.zeros(shape, device=device, dtype=dtype)
        self.values  = torch.zeros(shape, device=device, dtype=dtype)

        self.lengths = torch.zeros(num_lanes, dtype=torch.long, device=device) # ABSOLUTE count(no wrap-around)
       
        
        self._n_old  = 0 # how much of the cache this forward pass reads



    def free(self, lane: int):
        # no need to zero memory; the mask hides stale data
        self.lengths[lane] = 0    


    # ONCE per forward, after ALL layers
    def commit(self, lane_ids, q_len):
        # lengths[lane] is the absolute token count and never wraps. 
        # RoPE and the mask use absolute positions (lengths + arange(T))
        self.lengths[lane_ids] += q_len


    # runs once before the layers and returns position_ids and a bool mask
    def prepare(self, lane_ids, q_len):
        cache_window = self.cache_window
        start = self.lengths[lane_ids]   # [B] ABSOLUTE tokens seen so far
        dev = start.device

        # absolute position of every token in this chunk. Drives RoPE. Never wraps.
        pos = start[:, None] + torch.arange(q_len, device=dev)[None, :]      # [B, T]

        # what the ring currently holds (state BEFORE this chunk is written)
        n_old = min(int(start.max()), cache_window)  
        j     = torch.arange(n_old, device=dev)
        last  = (start - 1)[:, None]                                 # newest old position, -1 if empty
        
        a_old = last - (last - j[None, :]) % cache_window            # abs position at each physical idx
                                                                    # (negative => never written)
        # keys seen by attention = [old ring | new chunk]
        a_all = torch.cat([a_old, pos], dim=1)                       # [B, n_old + T]

        a, p = a_all[:, None, :], pos[:, :, None]
        mask = (a >= 0) & (a <= p) & (a > p - cache_window)                     # exists, causal, in window

        self._n_old = n_old
       
        return pos, mask[:, None]                                    # [B, 1, T, n_old + T]



    # reads first, then writes. Attention runs over [old ring | new chunk], and only 
    # the last min(T, W) chunk tokens are written into the ring. This makes chunked 
    # prefill safe on a full cache for any chunk size, and the cache stays exactly W long.
    def update(self, layer_idx, lane_ids, k, v):
        B, H, T, D = k.shape
        K, V = self.keys[layer_idx], self.values[layer_idx]
        k, v = k.to(K.dtype), v.to(K.dtype)

        # 1) READ first. Advanced indexing returns a copy, so the later write can't corrupt it.
        k_all = torch.cat([K[lane_ids, :, :self._n_old], k], dim=2)
        v_all = torch.cat([V[lane_ids, :, :self._n_old], v], dim=2)

        # 2) WRITE only the last min(T, W) tokens (indices are then unique, so no write races)
        m    = min(T, self.cache_window)
        pos  = self.lengths[lane_ids][:, None] + torch.arange(T - m, T, device=k.device)[None, :]
        phys = pos % self.cache_window
        K[lane_ids[:, None], :, phys] = k[:, :, T - m:].transpose(1, 2)
        V[lane_ids[:, None], :, phys] = v[:, :, T - m:].transpose(1, 2)

        return k_all, v_all
