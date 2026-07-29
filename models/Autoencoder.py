import torch
import torch.nn as nn

from utils.dim import ensure_batch_first, restore_shape
from models.training import ReconstructionTrainer


class Model(nn.Module):

    """Just like USAD, but without adversarial training scheme"""

    def __init__(self, cfg):
        super(Model, self).__init__()
        self.enc_in = cfg.enc_in
        self.seq_len = cfg.seq_len
        self.n = self.enc_in * self.seq_len
        self.latent_size = cfg.latent_size

        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()

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

        x_hat = self.decoder(z)

        x_hat = torch.unflatten(x_hat, dim=1, sizes=(seq_len, n_feats))
        x_hat = restore_shape(x_hat, *dim_flags)
        return x_hat

    def get_trainer(self):
        return ReconstructionTrainer(self)
