import math

import torch
import torch.nn as nn

from src.layers import Linear


def softmax(x: torch.Tensor, dim: int):
    x_max = x.amax(dim=dim, keepdim=True)
    exp = (x - x_max).exp()
    return exp / exp.sum(dim=dim, keepdim=True)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, rope_theta: float, head_dim: int, context_length: int, device=None):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
        self.head_dim = head_dim
        self.context_length = context_length
        exponents = torch.arange(0, head_dim, 2, device=device).float() / head_dim
        inv_freq = rope_theta ** (-exponents)
        positions = torch.arange(context_length, device=device).float()
        angles = positions[:, None] * inv_freq[None, :]
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        if x.shape[-1] != self.head_dim:
            raise ValueError(f"input last dimension must equal head_dim ({self.head_dim})")
        if token_positions.dtype == torch.bool or token_positions.is_floating_point() or token_positions.is_complex():
            raise TypeError("token_positions must contain integer positions")
        if token_positions.ndim == 0 or token_positions.shape[-1] != x.shape[-2]:
            raise ValueError("token_positions final dimension must equal sequence length")
        token_positions = token_positions.long()
        if token_positions.numel() > 0 and (token_positions.min() < 0 or token_positions.max() >= self.context_length):
            raise ValueError("token_positions outside supported context_length")

        while token_positions.ndim < x.ndim - 1:
            token_positions = token_positions.unsqueeze(-2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2).to(x.dtype)


def scaled_dot_product_attention(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask=None):
    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError("queries and keys must share the same feature width")
    if keys.shape[-2] != values.shape[-2]:
        raise ValueError("keys and values must share the same sequence length")
    if mask is not None and mask.dtype != torch.bool:
        raise TypeError("mask must be a boolean tensor")

    d_k = queries.shape[-1]
    scores = queries @ keys.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        if not mask.any(dim=-1).all():
            raise ValueError("every query must have at least one unmasked key")
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = softmax(scores, dim=-1)
    return weights @ values


class CausalGroupedQuerySelfAttention(nn.Module):
    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int, context_length: int, rope_theta: float, device=None, dtype=None):
        super().__init__()
        if d_model % n_q_heads != 0:
            raise ValueError("d_model must be divisible by n_q_heads")
        if n_q_heads % n_kv_heads != 0:
            raise ValueError("n_q_heads must be divisible by n_kv_heads")
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_q_heads
        self.q_proj = Linear(d_model, n_q_heads * self.head_dim, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, n_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, n_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.out_proj = Linear(n_q_heads * self.head_dim, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(rope_theta, self.head_dim, context_length, device=device)

    def forward(self, x: torch.Tensor, token_positions=None):
        B, S, _ = x.shape
        if token_positions is None:
            token_positions = torch.arange(S, device=x.device)

        q = self.q_proj(x).view(B, S, self.n_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        group_size = self.n_q_heads // self.n_kv_heads
        q = q.reshape(B, self.n_kv_heads, group_size, S, self.head_dim)

        causal_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=x.device))
        scores = torch.einsum("bngid,bnjd->bngij", q, k) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        weights = softmax(scores, dim=-1)
        out = torch.einsum("bngij,bnjd->bngid", weights, v)

        out = out.permute(0, 3, 1, 2, 4).reshape(B, S, self.n_q_heads * self.head_dim)
        return self.out_proj(out)
