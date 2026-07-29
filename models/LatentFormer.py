import logging
import torch.nn as nn

from layers.Embed import EmbeddingWrapper
from layers.mha import MultiheadAttentionMemory, MultiheadAttentionPooling
from models.training import ReconstructionTrainer
from utils.dim import ensure_batch_first, restore_shape

logger = logging.getLogger(__name__)


class Encoder(nn.Module):

    def __init__(self, in_features, latent_size, d_model, num_heads, d_ff, num_layers=1, batch_first=False, embed_type='linear'):
        super().__init__()
        self.num_layers = num_layers
        self.embed = EmbeddingWrapper(embed_type=embed_type, in_features=in_features, out_features=d_model, positional_encoding=True, dropout=0.0, batch_first=batch_first)

        self.enc = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff, dropout=0, batch_first=batch_first) for _ in range(self.num_layers)]
        )

        self.bottleneck = nn.Sequential(
            MultiheadAttentionPooling(num_seeds=latent_size,
                                      embed_dim=d_model,
                                      batch_first=batch_first,
                                      num_heads=num_heads),
        )

    def forward(self, x):
        h = self.embed(x)
        for layer in range(self.num_layers):
            h = self.enc[layer](h)
        h = self.bottleneck(h)
        return h


class Decoder(nn.Module):

    def __init__(self, latent_size, d_model, num_heads, seq_len, out_features):
        super().__init__()
        self.memory_expansion = nn.Linear(in_features=1, out_features=d_model)
        self.dec = nn.Sequential(
            MultiheadAttentionMemory(num_seeds=latent_size,
                                     embed_dim=d_model,
                                     num_heads=num_heads,
                                     batch_first=True),
            MultiheadAttentionPooling(num_seeds=seq_len,
                                      embed_dim=d_model,
                                      num_heads=num_heads,
                                      batch_first=True),
            nn.Linear(in_features=d_model, out_features=out_features),
        )

    def forward(self, x):
        h = x

        if h.shape[-1] == 1:
            h = self.memory_expansion(h)

        h = self.dec(h)
        return h


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(in_features=cfg.enc_in,
                               latent_size=cfg.latent_size,
                               d_model=cfg.d_model,
                               num_heads=cfg.n_heads,
                               d_ff=cfg.d_ff,
                               num_layers=cfg.num_layers,
                               batch_first=True,
                               embed_type=cfg.embed_type)
        self.decoder = Decoder(latent_size=cfg.latent_size,
                               d_model=cfg.d_model,
                               num_heads=cfg.n_heads,
                               seq_len=cfg.seq_len,
                               out_features=cfg.enc_in)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        dim_flags, x = ensure_batch_first(x, batch_first=True)

        z = self.encode(x)
        x_hat = self.decode(z)

        x_hat = restore_shape(x_hat, *dim_flags)
        return x_hat

    def get_trainer(self):
        return ReconstructionTrainer(self)
