import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import EmbeddingWrapper, PositionalEncoding
from models.custom_transformer import TransformerBlock
# from models.vanilla_transformer import TransformerEncoder, TransformerEncoderLayer


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mask = self.cfg.mask if hasattr(self.cfg, 'mask') else None
        self.embedding = EmbeddingWrapper(embed_type=self.cfg.embed_type, in_features=self.cfg.enc_in, out_features=self.cfg.d_model, batch_first=True, dropout=self.cfg.dropout, kernel_size=self.cfg.embed_kernel_size)
        self.pe = PositionalEncoding(d_model=self.cfg.d_model, dropout=self.cfg.dropout, batch_first=True)

        custom_implementation = self.cfg.custom_implementation if hasattr(self.cfg, 'custom_implementation') else False
        if custom_implementation:
            self.transformer = TransformerBlock(d_model=self.cfg.d_model, num_heads=self.cfg.n_heads, d_ff=self.cfg.d_ff, dropout=self.cfg.dropout)
        else:
            self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.cfg.d_model, nhead=self.cfg.n_heads, batch_first=True, dim_feedforward=self.cfg.d_ff, dropout=self.cfg.dropout, activation=self.cfg.activation), num_layers=self.cfg.e_layers)
        self.projection = nn.Linear(self.cfg.d_model, self.cfg.c_out)

    @staticmethod
    def window_attention(sz, win_size, device=None, dtype=torch.float32):
        mask = torch.zeros(size=(sz, sz), device=device, dtype=dtype)

        for i in range(sz):
            i_start = max(0, i - win_size)
            i_end = min(sz, i + 1)

            mask[i][:i_start] = float('-inf')
            mask[i][i_end:] = float('-inf')

        return mask

    def forward(self, x, *args, **kwargs):
        if self.mask is None:
            mask = None
        elif self.mask == 'square':
            mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device, dtype=torch.float32)
        else:
            mask = self.window_attention(x.shape[1], self.mask, device=x.device)

        h = self.embedding(x)
        h = self.pe(h)
        h = self.transformer(h, mask=mask)
        h = self.projection(h)
        return h

    def train_step(self, x, epoch):
        x_reconstructed = self(x)
        return F.mse_loss(x_reconstructed, x)

    def anomaly_score(self, x):
        x_reconstructed = self(x)
        return F.mse_loss(x_reconstructed, x, reduction='none')
