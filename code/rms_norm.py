import torch
import torch.nn as nn
from einops import reduce



class RMSNorm(nn.Module):
    """
    Root Mean Square Norm.

    Input:
        x: Float tensor with shape (batch, seq_len, embed_dim)

    Output:
        normalized_x: Float tensor with shape (batch, seq_len, embed_dim)
    """

    def __init__(self, dim: int, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        
        # x: (batch, seq_len, embed_dim)
        x_fp32 = x.float()

        # variance: (batch, seq_len, 1)
        variance = reduce(
            x_fp32.pow(2),
            "batch seq_len embed_dim -> batch seq_len 1",
            "mean",
        )

        # rms_scale: (batch, seq_len, 1)
        rms_scale = torch.rsqrt(variance + self.eps)

        # normalized_x: (batch, seq_len, embed_dim)
        normalized_x = x * rms_scale.to(dtype=x.dtype)

        # self.weight broadcasts over batch and seq_len.
        return normalized_x * self.weight

