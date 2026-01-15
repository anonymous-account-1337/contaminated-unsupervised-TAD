import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.Embed import EmbeddingWrapper, PositionalEncoding


def l2_norm(a, b, reduce=True):
    if reduce:
        return torch.linalg.norm((a - b).reshape(-1), dim=0, ord=2)
    else:
        return torch.linalg.norm(a - b, dim=-1, ord=2)


class Encoder(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, dropout, layer_norm):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=d_model) if layer_norm else nn.Identity()
        self.feed_forward = nn.Sequential(nn.Linear(in_features=d_model, out_features=d_ff, bias=True),
                                          nn.ReLU(),
                                          nn.Linear(in_features=d_ff, out_features=d_model, bias=True))
        self.layer_norm2 = nn.LayerNorm(normalized_shape=d_model) if layer_norm else nn.Identity()

    def forward(self, x):
        # x = [B, L, C] or [L, C]
        h = self.layer_norm1(x + self.attn(x, x, x)[0])
        h = self.layer_norm2(h + self.feed_forward(h))
        return h


class WindowEncoder(nn.Module):

    def __init__(self, d_model, n_heads, dropout, layer_norm):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=d_model) if layer_norm else nn.Identity()
        self.attn2 = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=d_model) if layer_norm else nn.Identity()

    def forward(self, x, memory):
        # x = [B, L, C] or [L, C]
        attn_mask = nn.Transformer.generate_square_subsequent_mask(sz=x.shape[-2], device=x.device)
        h = self.layer_norm1(x + self.attn1(x, x, x, attn_mask=attn_mask)[0])
        h = self.layer_norm2(h + self.attn2(h, memory, memory)[0])
        return h


class Decoder(nn.Module):

    def __init__(self, d_model, d_ff, d_out, activation=None):
        super().__init__()
        self.feed_forward = nn.Sequential(nn.Linear(in_features=d_model, out_features=d_ff, bias=True),
                                          nn.ReLU(),
                                          nn.Linear(in_features=d_ff, out_features=d_out, bias=True))
        self.activation = nn.Identity() if activation is None else nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.feed_forward(x))


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.embed_encoder = EmbeddingWrapper(embed_type=cfg.embed_type, in_features=cfg.enc_in * 2, out_features=cfg.d_model, positional_encoding=False, bias=False, batch_first=True, dropout=cfg.dropout, kernel_size=cfg.embed_kernel_size)
        self.embed_w_encoder = EmbeddingWrapper(embed_type=cfg.embed_type, in_features=cfg.enc_in, out_features=cfg.d_model, positional_encoding=False, bias=False, batch_first=True, dropout=cfg.dropout, kernel_size=cfg.embed_kernel_size)
        self.pe = PositionalEncoding(d_model=cfg.d_model, dropout=cfg.dropout, batch_first=True)

        self.encoder = Encoder(d_model=cfg.d_model, n_heads=cfg.n_heads, d_ff=cfg.d_ff, dropout=cfg.dropout, layer_norm=cfg.layer_norm)
        self.w_encoder = WindowEncoder(d_model=cfg.d_model, n_heads=cfg.n_heads, dropout=cfg.dropout, layer_norm=cfg.layer_norm)

        self.dec1 = Decoder(d_model=cfg.d_model, d_ff=cfg.d_ff, d_out=cfg.enc_in)
        self.dec2 = Decoder(d_model=cfg.d_model, d_ff=cfg.d_ff, d_out=cfg.enc_in)

        self.epsilon = cfg.epsilon

    def embed(self, x, focus_score=None):
        if focus_score is None:
            x = self.embed_w_encoder(x)
            x = self.pe(x)
        else:
            x = torch.cat((x, focus_score), dim=2)
            x = self.embed_encoder(x)
            x = self.pe(x)
        return x

    def forward(self, x):
        z = self.w_encoder(self.embed(x), self.encoder(self.embed(x, torch.zeros_like(x))))
        o1 = self.dec1(z)
        o2 = self.dec2(z)
        o2_hat = self.dec2(self.w_encoder(self.embed(x), self.encoder(self.embed(x, (x - o1) ** 2))))
        return o1, o2, o2_hat

    def train_step(self, x, epoch):
        criterion = l2_norm
        o1, o2, o2_hat = self(x)

        l1 = self.epsilon ** (-epoch) * criterion(o1, x) + (1 - self.epsilon ** (-epoch)) * criterion(o2_hat, x)
        l2 = self.epsilon ** (-epoch) * criterion(o2, x) - (1 - self.epsilon ** (-epoch)) * criterion(o2_hat, x)

        loss = l1 + l2
        return loss

    def anomaly_score(self, x, alpha=0.5, beta=0.5):
        criterion = l2_norm
        o1, o2, o2_hat = self(x)
        return alpha * criterion(o1, x, reduce=False) + beta * criterion(o2_hat, x, reduce=False)
