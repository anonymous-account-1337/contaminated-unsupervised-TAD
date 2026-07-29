import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.wq = nn.Linear(d_model, d_k)
        self.wk = nn.Linear(d_model, d_k)
        self.wv = nn.Linear(d_model, d_v)
        self.scaling = math.sqrt(d_k)

    def forward(self, q, k, v):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        k = k.permute(0, 2, 1)

        return F.softmax((q @ k) / self.scaling, dim=2) @ v


class MultiheadAttention(nn.Module):

    def __init__(self, num_heads, d_model, d_k=None, d_v=None):
        super().__init__()

        if d_k is None:
            if d_model % num_heads != 0:
                raise ValueError('invalid combination of d_model and num_heads')

            d_k = d_model // num_heads

        if d_v is None:
            if d_model % num_heads != 0:
                raise ValueError('invalid combination of d_model and num_heads')

            d_v = d_model // num_heads

        self.heads = nn.ModuleList([DotProductAttention(d_model, d_k, d_v) for _ in range(num_heads)])
        self.wo = nn.Linear(d_v * num_heads, d_model)

    def forward(self, q, k, v):
        h = torch.cat([head(q, k, v) for head in self.heads], dim=2)
        h = self.wo(h)
        return h


class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiheadAttention(num_heads=num_heads, d_model=d_model)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout = nn.Dropout(p=dropout, inplace=False)

    def forward(self, x, **kwargs):
        h = x
        h = self.norm1(h + self.dropout(self.attn(h, h, h)))
        h = self.norm2(h + self.dropout(self.ff2(F.relu(self.ff1(h)))))
        return h
