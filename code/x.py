# ================================================================
# config (no head_dim)
# ================================================================
config = {
    "vocab_size": 32000,
    "dim": 4096,
    "num_layers": 32,
    "num_q_heads": 32,
    "num_kv_heads": 8,          # 32=MHA, 8=GQA, 1=MQA
    "max_seq_len": 4096,
    "context_window": 4096,
    "rope_theta": 10000.0,
    "dropout": 0.0,
}

# ================================================================
# Rotary Embedding (pre-computed)
# ================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len, device)

    def _build_cache(self, seq_len: int, device: Optional[torch.device]):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len = seq_len

    def forward(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        max_pos = int(position_ids.max()) + 1
        if max_pos > self.max_seq_len:
            self._build_cache(max_pos, position_ids.device)

        cos = self.cos_cached[position_ids].to(dtype=dtype)
        sin = self.sin_cached[position_ids].to(dtype=dtype)
        return cos.unsqueeze(1), sin.unsqueeze(1)   # [B, 1, seq, head_dim]

def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

# ================================================================
# Static KV Cache
# ================================================================
class StaticKVCache:
    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        context_window: Optional[int] = None,
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.context_window = context_window or max_seq_len
        self.device = device
        self.dtype = dtype

        shape = (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim)
        self.keys   = torch.zeros(shape, device=device, dtype=dtype)
        self.values = torch.zeros(shape, device=device, dtype=dtype)
        self.seq_len  = 0
        self.position = 0          # absolute position counter

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,         # [B, num_kv_heads, q_len, head_dim] (RoPE already applied)
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_len = key.shape[2]

        if self.seq_len + q_len > self.max_seq_len:
            self._shift(q_len)

        start, end = self.seq_len, self.seq_len + q_len
        self.keys  [layer_idx, :, :, start:end, :] = key
        self.values[layer_idx, :, :, start:end, :] = value

        if layer_idx == 0:
            self.seq_len  = end
            self.position += q_len

        return self.get(layer_idx)

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        usable = min(self.seq_len, self.context_window)
        start  = self.seq_len - usable
        return (
            self.keys  [layer_idx, :, :, start:self.seq_len, :],
            self.values[layer_idx, :, :, start:self.seq_len, :],
        )

    def _shift(self, incoming: int):
        keep = max(0, self.context_window - incoming)
        if keep > 0:
            self.keys  [..., :keep, :] = self.keys  [..., self.seq_len-keep:self.seq_len, :].clone()
            self.values[..., :keep, :] = self.values[..., self.seq_len-keep:self.seq_len, :].clone()
        self.seq_len = keep


    def reset(self):
        self.seq_len  = 0
        self.position = 0

    @property
    def current_abs_pos(self) -> int:
        return self.position

# ================================================================
# SubBlock (head_dim calculated here)
# ================================================================
class SubBlock(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.dim          = config["dim"]
        self.num_q_heads  = config["num_q_heads"]
        self.num_kv_heads = config["num_kv_heads"]
        self.dropout      = config.get("dropout", 0.0)

        # head_dim is derived, never taken from config
        assert self.dim % self.num_q_heads == 0, "dim must be divisible by num_q_heads"
        self.head_dim = self.dim // self.num_q_heads

        self.q_proj = nn.Linear(self.dim, self.num_q_heads  * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_q_heads * self.head_dim, self.dim, bias=False)

        self.rotary = RotaryEmbedding(
            head_dim    = self.head_dim,
            max_seq_len = config["max_seq_len"],
            base        = config.get("rope_theta", 10000.0),
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[StaticKVCache] = None,
        layer_idx: int = 0,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, q_len, _ = x.shape

        q = self.q_proj(x).view(B, q_len, self.num_q_heads,  self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE
        if position_ids is None:
            position_ids = torch.arange(q_len, device=x.device)
        cos, sin = self.rotary(position_ids, dtype=q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV cache (k already has RoPE)
        if cache is not None:
            k, v = cache.update(layer_idx, k, v)

        # GQA / MQA
        if self.num_q_heads != self.num_kv_heads:
            n_rep = self.num_q_heads // self.num_kv_heads
            k = torch.repeat_interleave(k, n_rep, dim=1)
            v = torch.repeat_interleave(v, n_rep, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().view(B, q_len, -1)
        return self.o_proj(out)

# ================================================================
# Block
# ================================================================
class Block(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config["dim"])
        self.attn      = SubBlock(config)
        self.ffn_norm  = nn.RMSNorm(config["dim"])

        hidden = 4 * config["dim"]
        self.ffn = nn.Sequential(
            nn.Linear(config["dim"], hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, config["dim"], bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[StaticKVCache] = None,
        layer_idx: int = 0,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cache=cache, layer_idx=layer_idx, position_ids=position_ids)
        x = x + self.ffn(self.ffn_norm(x))
        return x

# ================================================================
# Model
# ================================================================
class Model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.embed  = nn.Embedding(config["vocab_size"], config["dim"])
        self.layers = nn.ModuleList([Block(config) for _ in range(config["num_layers"])])
        self.norm   = nn.RMSNorm(config["dim"])
        self.lm_head = nn.Linear(config["dim"], config["vocab_size"], bias=False)

        # expose for the inference engine
        self.num_layers   = config["num_layers"]
        self.num_kv_heads = config["num_kv_heads"]
        # head_dim is taken from the first sub-block (all layers share the same value)
        self.head_dim     = self.layers[0].attn.head_dim

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Optional[StaticKVCache] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, seq_len = input_ids.shape
        x = self.embed(input_ids)

        if position_ids is None:
            if cache is not None and cache.current_abs_pos > 0:
                start = cache.current_abs_pos
                position_ids = torch.arange(start, start + seq_len, device=input_ids.device)
            else:
                position_ids = torch.arange(seq_len, device=input_ids.device)

        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0).expand(B, -1)

        for i, layer in enumerate(self.layers):
            x = layer(x, cache=cache, layer_idx=i, position_ids=position_ids)

        x = self.norm(x)
        return self.lm_head(x)

# ================================================================
# Inference Engine
# ================================================================
class InferenceEngine:
    def __init__(
        self,
        model: Model,
        max_seq_len: Optional[int] = None,
        context_window: Optional[int] = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.model = model.eval()
        self.config = model.config
        self.max_seq_len = max_seq_len or self.config["max_seq_len"]
        self.context_window = context_window or self.config.get("context_window", self.max_seq_len)
        self.dtype = dtype
        self.cache: Optional[StaticKVCache] = None

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        device = input_ids.device
        B = input_ids.shape[0]

        self.cache = StaticKVCache(
            num_layers    = self.model.num_layers,
            batch_size    = B,
            num_kv_heads  = self.model.num_kv_heads,
            head_dim      = self.model.head_dim,          # taken from model
            max_seq_len   = self.max_seq_len,
            device        = device,
            dtype         = self.dtype,
            context_window= self.context_window,
        )

        # Prefill
        logits = self.model(input_ids, cache=self.cache)
        next_token = self._sample(logits[:, -1:], temperature, top_k)
        generated = [next_token]

        # Decode
        for _ in range(max_new_tokens - 1):
            logits = self.model(next_token, cache=self.cache)
            next_token = self._sample(logits[:, -1:], temperature, top_k)
            generated.append(next_token)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        self.cache.reset()
        return torch.cat(generated, dim=1)

    def _sample(self, logits: torch.Tensor, temperature: float, top_k: Optional[int]):
        if temperature == 0.0:
            return logits.argmax(dim=-1)
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)