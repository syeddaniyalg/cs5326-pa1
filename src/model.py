import torch
import torch.nn as nn

from src.attention import CausalGroupedQuerySelfAttention
from src.layers import Embedding, Linear, RMSNorm, SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int, d_ff: int, context_length: int, rope_theta: float, norm_eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.attention_norm = RMSNorm(d_model, norm_eps, device=device, dtype=dtype)
        self.attention = CausalGroupedQuerySelfAttention(d_model, n_q_heads, n_kv_heads, context_length, rope_theta, device=device, dtype=dtype)
        self.ffn_norm = RMSNorm(d_model, norm_eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions=None):
        x = x + self.attention(self.attention_norm(x), token_positions=token_positions)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, n_q_heads: int, n_kv_heads: int, d_ff: int, rope_theta: float, norm_eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_q_heads, n_kv_heads, d_ff, context_length, rope_theta, norm_eps, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model, norm_eps, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor, token_positions=None):
        S = token_ids.shape[-1]
        if S < 1 or S > self.context_length:
            raise ValueError(f"sequence_length must be between 1 and context_length ({self.context_length})")
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x, token_positions=token_positions)
        x = self.final_norm(x)
        return self.lm_head(x)
