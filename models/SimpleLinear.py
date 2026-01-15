import torch
import torch.nn as nn
from layers.Embed import DataEmbedding


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.embedding = DataEmbedding(self.cfg.enc_in, self.cfg.d_model, self.cfg.embed, self.cfg.freq, self.cfg.dropout)
        self.stack = nn.Sequential(
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
            nn.ReLU(),
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
            nn.ReLU(),
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
            nn.ReLU(),
        )
        self.projection = nn.Linear(self.cfg.d_model, self.cfg.c_out)

    def forward(self, x, *args):
        h = self.embedding(x, None)
        h = self.stack(h)
        h = self.projection(h)
        return h
