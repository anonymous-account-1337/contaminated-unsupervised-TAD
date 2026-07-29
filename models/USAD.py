import torch
import torch.nn as nn

from utils.dim import ensure_batch_first, restore_shape
from models.training import ReconstructionTrainer, USADTrainer


class Model(nn.Module):

    def __init__(self, cfg):
        super(Model, self).__init__()
        self.enc_in = cfg.enc_in
        self.seq_len = cfg.seq_len
        self.n = self.enc_in * self.seq_len
        self.latent_size = cfg.latent_size

        self.encoder = self.create_encoder()
        self.decoder1 = self.create_decoder()
        self.decoder2 = self.create_decoder()

    def create_encoder(self):
        return nn.Sequential(
            nn.Linear(self.n, self.n // 2), nn.ReLU(),
            nn.Linear(self.n // 2, self.n // 4), nn.ReLU(),
            nn.Linear(self.n // 4, self.latent_size), nn.ReLU(),
        )

    def create_decoder(self):
        return nn.Sequential(
            nn.Linear(self.latent_size, self.n // 4), nn.ReLU(),
            nn.Linear(self.n // 4, self.n // 2), nn.ReLU(),
            nn.Linear(self.n // 2, self.n),
        )

    def forward(self, x, return_latent_space=False):
        dim_flags, x = ensure_batch_first(x, True)
        _, seq_len, n_feats = x.shape
        x = torch.flatten(x, start_dim=1, end_dim=-1)

        z = self.encoder(x)

        if return_latent_space:
            return z

        ae1 = self.decoder1(z)
        ae2 = self.decoder2(z)

        ae1 = torch.unflatten(ae1, dim=1, sizes=(seq_len, n_feats))
        ae2 = torch.unflatten(ae2, dim=1, sizes=(seq_len, n_feats))

        ae1 = restore_shape(ae1, *dim_flags)
        ae2 = restore_shape(ae2, *dim_flags)
        return ae1, ae2

    def get_trainer(self):
        return USADTrainer(self)
