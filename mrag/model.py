"""A small causal transformer used as an embedding model.

Following the paper's setup, the embedding of a text is read at its last token. The
standard embedding is the final hidden state there. The multi-aspect embedding is the
set of per-head attention outputs of the last attention layer at the same token,
taken before the output projection merges the heads, so each head contributes one
vector of size dim / heads. Both come from a single forward pass.
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class EncoderConfig:
    vocab: int
    dim: int = 128
    heads: int = 8
    layers: int = 3
    max_len: int = 64


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def __call__(self, x):
        B, T, D = x.shape
        q, k, v = (t.reshape(B, T, self.heads, -1).transpose(0, 2, 1, 3) for t in mx.split(self.qkv(x), 3, axis=-1))
        per_head = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask="causal")
        merged = per_head.transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.out(merged), per_head  # per_head: (B, heads, T, dim // heads)


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def __call__(self, x):
        a, per_head = self.attn(self.norm1(x))
        x = x + a
        return x + self.mlp(self.norm2(x)), per_head


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.dim)
        self.pos = nn.Embedding(cfg.max_len, cfg.dim)
        self.blocks = [Block(cfg.dim, cfg.heads) for _ in range(cfg.layers)]
        self.norm = nn.RMSNorm(cfg.dim)

    def forward(self, tokens):
        """Return final hidden states (B, T, D) and the last block's per-head outputs."""
        x = self.tok(tokens) + self.pos(mx.arange(tokens.shape[1]))
        per_head = None
        for block in self.blocks:
            x, per_head = block(x)
        return self.norm(x), per_head

    def encode(self, tokens, lengths):
        """Embeddings at each row's last real token. Rows are right-padded; the causal
        mask means padding after the last token cannot influence it.

        Returns (final, heads): (B, D) and (B, heads, D // heads).
        """
        hidden, per_head = self.forward(tokens)
        last = lengths - 1
        final = mx.take_along_axis(hidden, last[:, None, None], axis=1)[:, 0]
        heads = mx.take_along_axis(per_head, last[:, None, None, None], axis=2)[:, :, 0]
        return final, heads
