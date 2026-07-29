import torch
import torch.nn as nn

from utils.dim import ensure_batch_first, restore_shape
from layers.Embed import PositionalEncoding, EmbeddingWrapper
from models.training import TranADTrainer


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.epsilon = 1.05
        self.d_model = cfg.d_model

        self.src_embed = EmbeddingWrapper(embed_type=cfg.embed_type, kernel_size=cfg.embed_kernel_size, in_features=cfg.enc_in * 2, out_features=self.d_model, positional_encoding=False, batch_first=True, dropout=0.0)
        self.tgt_embed = EmbeddingWrapper(embed_type=cfg.embed_type, kernel_size=cfg.embed_kernel_size, in_features=cfg.enc_in * 1, out_features=self.d_model, positional_encoding=False, batch_first=True, dropout=0.0)
        self.pos_encoder = PositionalEncoding(d_model=self.d_model, dropout=cfg.dropout, batch_first=True)

        # shared encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)

        dec_layer1 = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True
        )
        self.decoder1 = nn.TransformerDecoder(dec_layer1, num_layers=1)

        dec_layer2 = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True
        )
        self.decoder2 = nn.TransformerDecoder(dec_layer2, num_layers=1)
        self.fcn = nn.Linear(self.d_model, cfg.enc_in)

    def encode(self, src, focus_score, tgt):
        h = torch.cat([src, focus_score], dim=-1)
        h = self.pos_encoder(self.src_embed(h))
        memory = self.encoder(h)

        tgt = self.pos_encoder(self.tgt_embed(tgt))
        return tgt, memory

    def forward(self, x):
        dim_flags, x = ensure_batch_first(x, True)

        # phase 1 (reconstruction)
        focus_score = torch.zeros_like(x)
        o1 = self.fcn(self.decoder1(*self.encode(x, focus_score, x)))
        o2 = self.fcn(self.decoder2(*self.encode(x, focus_score, x)))

        # phase 2 (self-conditioning)
        focus_score = (o1 - x) ** 2
        o2_hat = self.fcn(self.decoder2(*self.encode(x, focus_score, x)))

        o1 = restore_shape(o1, *dim_flags)
        o2 = restore_shape(o2, *dim_flags)
        o2_hat = restore_shape(o2_hat, *dim_flags)
        return o1, o2, o2_hat

    def get_trainer(self):
        return TranADTrainer(self)
