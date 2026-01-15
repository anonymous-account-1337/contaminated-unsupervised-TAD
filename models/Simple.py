import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.rnn = nn.RNN(self.cfg.enc_in, self.cfg.d_model, batch_first=True)
        self.proj = nn.Linear(self.cfg.d_model, self.cfg.c_out)

    def forward(self, x, *args):
        h, _ = self.rnn(x)
        h = self.proj(h)

        return h
