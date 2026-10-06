
import torch

from kv_cache import KV_cache
from model_config import ModelConfig



# cache equivalence

# A causality test: perturbing a future token must not change earlier logits

# GQA test: output with num_kv_heads < num_q_heads should match manually repeat_interleaved K/V.

# overfit-one-batch test: the loss should reach near zero.

# RoPE test: scores should depend only on relative offset.

# multiple lanes at different lengths
# lane reuse after free()
# wraparound, with chunk sizes both smaller and larger than W
# advanced_inference at temperature 0 vs. a naive loop
# GQA vs. manually expanded MHA
# the RoPE relative-position property
# that a single batch can be overfit to near-zero loss