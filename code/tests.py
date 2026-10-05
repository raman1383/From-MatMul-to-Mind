
# cache equivalence
import torch

from kv_cache import KV_cache
from model_config import ModelConfig


def test_cache_matches_full_forward():
    torch.manual_seed(0)
    cfg = ModelConfig(..., num_q_heads=4, num_kv_heads=2)   # exercise GQA
    model = Model(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 20))
    full = model(ids)

    cache = KV_cache(cfg, inference_batch_size=2, device="cpu", dtype=torch.float32)
    outs = [model(ids[:, :12], cache=cache)]
    for t in range(12, 20):
        outs.append(model(ids[:, t:t+1], cache=cache))

    torch.testing.assert_close(torch.cat(outs, 1), full, atol=1e-5, rtol=1e-4)




# A causality test: perturbing a future token must not change earlier logits

# A GQA test: output with num_kv_heads < num_q_heads should match manually repeat_interleaved K/V.

# An initial-loss test: it should be ≈ ln(vocab).

# An overfit-one-batch test: the loss should reach near zero.

# A RoPE test: scores should depend only on relative offset.