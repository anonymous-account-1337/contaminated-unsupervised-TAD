import torch
import torch.nn as nn


def l2_norm(a, b, reduce=True):
    if reduce:
        return torch.linalg.norm((a - b).reshape(-1), dim=0, ord=2)
    else:
        return torch.linalg.norm(a - b, dim=-1, ord=2)


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
        # x = [B, L, C] or [L, C]
        squeeze = False
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=0)
            squeeze = True
        elif len(x.shape) != 3:
            raise ValueError('invalid shape')

        _, seq_len, n_feats = x.shape

        x = torch.flatten(x, start_dim=1, end_dim=-1)

        z = self.encoder(x)

        if return_latent_space:
            return z

        ae1 = self.decoder1(z)
        ae2 = self.decoder2(z)
        ae2ae1 = self.decoder2(self.encoder(ae1))

        ae1 = torch.unflatten(ae1, dim=1, sizes=(seq_len, n_feats))
        ae2 = torch.unflatten(ae2, dim=1, sizes=(seq_len, n_feats))
        ae2ae1 = torch.unflatten(ae2ae1, dim=1, sizes=(seq_len, n_feats))

        if squeeze:
            ae1 = ae1.squeeze(dim=0)
            ae2 = ae2.squeeze(dim=0)
            ae2ae1 = ae2ae1.squeeze(dim=0)

        return ae1, ae2, ae2ae1  # [B, L, C] or [L, C]

    def train_step(self, x, epoch):
        criterion = l2_norm
        w1, w2, w21 = self(x)
        l1 = (1 / epoch) * criterion(x, w1) + (1 - 1 / epoch) * criterion(x, w21)
        l2 = (1 / epoch) * criterion(x, w2) - (1 - 1 / epoch) * criterion(x, w21)
        loss = l1 + l2
        return loss

    def anomaly_score(self, x, alpha=0.5, beta=0.5):
        criterion = l2_norm
        w1, _, w21 = self(x)
        score = alpha * criterion(x, w1, reduce=False) + beta * criterion(x, w21, reduce=False)
        return score
