from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionConfig:
    num_q_heads: int
    num_kv_heads: int
    rope_theta: float


@dataclass(frozen=True)
class FFNConfig:
    use_gate: bool
    mlp_to_embed_expand_factor: float
    num_experts: int
    num_experts_per_tkn: int



@dataclass(frozen=True)
class ModelConfig:

    model_name: str 
    tied_embeddings: bool
    max_training_batch_size: int
    embed_dim: int
    vocab_size: int
    
    max_context_window: int | None 
    num_layers: int | None
    
    learning_rate: float


    attention: AttentionConfig | None
    ffn: FFNConfig | None

    save_path: str
    val_loss_history: str

    checkpoint_dir: str = "../checkpoints"
    checkpoint_every_steps: int = 1000
    keep_last_n_checkpoints: int = 3



    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.attention.num_q_heads

    @property
    def num_groups(self) -> int:
        return self.attention.num_q_heads // self.attention.num_kv_heads


    def __post_init__(self):

        if self.attention is not None:

            if self.embed_dim % self.attention.num_q_heads != 0:
                raise ValueError(
                    "embed_dim must be divisible by num_q_heads"
                )

            if self.attention.num_q_heads % self.attention.num_kv_heads != 0:
                raise ValueError(
                    "num_q_heads must be divisible by num_kv_heads"
                )

            if self.head_dim % 2 != 0:
                raise ValueError(
                    "head_dim must be even for rotary embeddings"
                )

