import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class Attention(nn.Module):

    def __init__(self, embed_dim, d_k, d_v, bias=True):
        super().__init__()
        self.q = nn.Linear(embed_dim, d_k, bias=bias)
        self.k = nn.Linear(embed_dim, d_k, bias=bias)
        self.v = nn.Linear(embed_dim, d_v, bias=bias)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        return F.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]), dim=1) @ v


class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, embed_dim, d_model, bias=True):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')

        d_v = d_model // n_heads
        self.heads = nn.ModuleList()
        for _ in range(n_heads):
            self.heads.append(Attention(embed_dim=embed_dim, d_k=d_model, d_v=d_v, bias=bias))

        self.proj = nn.Linear(n_heads * d_v, d_model, bias=bias)

    def forward(self, x):
        return self.proj(torch.concat([head(x) for head in self.heads], dim=-1))


class TransformerEncoderLayer(nn.Module):

    def __init__(self, n_heads, d_model, d_feed_forward = 2048, bias = True, dropout = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(n_heads, d_model, d_model, bias=bias)
        self.ln1 = nn.LayerNorm(d_model, bias=bias)
        self.feed_forward1 = nn.Linear(d_model, d_feed_forward, bias=bias)
        self.feed_forward2 = nn.Linear(d_feed_forward, d_model, bias=bias)
        self.ln2 = nn.LayerNorm(d_model, bias=bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        h = x

        attention = self.dropout(self.attention(h))
        h = self.ln1(attention + h)

        feed_forward = self.dropout(self.feed_forward2(self.feed_forward1(h)))
        h = self.ln2(feed_forward + h)

        return h


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.encoder_layers = nn.Sequential()
        for _ in range(num_layers):
            self.encoder_layers.append(deepcopy(encoder_layer))

    def forward(self, *args, **kwargs):
        x = args[0]
        return self.encoder_layers(x)
