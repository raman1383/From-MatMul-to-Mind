import torch
from typing import Union
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ModelConfig:

    model_name: str 
    tied_embeddings: bool
    embed_dim: int
    vocab_size: int
    num_layers: int
    max_context_window: int

    num_q_heads: int
    num_kv_heads: int

    max_training_batch_size: int
    learning_rate: float

    rope_theta: float

    mlp_to_embed_expand_factor: float | None

    num_experts: int | None
    num_experts_per_tkn: int | None

    save_path: str
    val_loss_history: str


    @property
    def head_dim(self):
        return self.embed_dim // self.num_q_heads

    @property
    def num_groups(self):
        return self.num_q_heads // self.num_kv_heads


    def __post_init__(self):
        
        if self.embed_dim % self.num_q_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_q_heads"
            )

        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_q_heads must be divisible by num_kv_heads"
            )

        if self.head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be even for rotary embeddings"
            )
