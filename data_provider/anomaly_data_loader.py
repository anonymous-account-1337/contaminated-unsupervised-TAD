import numpy as np
import torch

from torch.utils.data import Dataset
from pathlib import Path


class AnomalyDataset(Dataset):

    def __init__(self, root_path, flag, win_size=None, device=None, **kwargs):
        if flag in ['train', 'val', 'test']:
            self.x = np.load(Path(root_path).joinpath(f'{flag}_x.npy'))
            self.y = np.load(Path(root_path).joinpath(f'{flag}_y.npy'))

            if device:
                self.x = torch.tensor(self.x, device=device, dtype=torch.float32)
                self.y = torch.tensor(self.y, device=device, dtype=torch.float32)
        else:
            raise ValueError(f'invalid split {flag}')

        if self.x.shape != self.y.shape:
            raise ValueError('shape mismatch')

        if win_size is not None and win_size != self.x.shape[0]:
            raise ValueError('win size should match sequence length')

    def get_dummy_sample(self, val=1, batch_size=1):
        seq_len, _, features = self.x.shape
        if val == 'rand':
            return torch.rand(size=(batch_size, seq_len, features), dtype=torch.float32)
        else:
            return torch.full(size=(batch_size, seq_len, features), fill_value=val, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[1]

    def __getitem__(self, item):
        return self.x[:, item, :], self.y[:, item, :]
