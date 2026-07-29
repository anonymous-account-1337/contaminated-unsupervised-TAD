import numpy as np
import torch

from torch.utils.data import Dataset
from pathlib import Path


class AnomalyDataset(Dataset):

    def __init__(self, root_path, flag, win_size=None, device=None, **kwargs):
        if flag not in ['train', 'val', 'test']:
            raise ValueError(f'invalid split {flag}')

        self.x = np.load(Path(root_path).joinpath(f'{flag}_x.npy'))
        try:
            self.y = np.load(Path(root_path).joinpath(f'{flag}_y.npy'))
        except FileNotFoundError as e:
            if flag in ['train', 'val']:
                # missing y values are ok in unsupervised TAD scenarios for train and val as these are assumed to be free of anomalies
                self.y = np.zeros(shape=(self.x.shape[0], self.x.shape[1], 1), dtype=np.float32)
            else:
                raise e

        if device:
            self.x = torch.tensor(self.x, device=device, dtype=torch.float32)
            self.y = torch.tensor(self.y, device=device, dtype=torch.float32)

        if self.x.shape[:-1] != self.y.shape[:-1]:
            raise ValueError('shape mismatch')

        if self.y.shape[-1] != 1:
            raise ValueError('y must have only one channel')

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


def add_noise(x, rng, k=0.1):
    return x + rng.normal(loc=0, scale=k, size=x.shape)


def add_spike(x, rng, k=2):
    x = x.copy()

    t = rng.integers(0, x.shape[0])
    scale = np.std(x) + 1e-8
    spike = k * scale

    if np.mean(x[t]) < 0:
        spike = -spike

    c = rng.integers(0, x.shape[1])
    x[t, c] = spike

    return x


def add_missing_values(x, rng, missing_value=0.0, p=0.1):
    x = x.copy()
    mask = rng.random(x.shape) < p
    x[mask] = missing_value
    return x


def add_drift(x, rng, k=0.5):
    drift = k * np.linspace(0, 1, num=x.shape[0])

    if x.ndim == 1:
        return x + drift

    return x + drift.reshape(-1, 1)


class CorruptedAnomalyDataset(AnomalyDataset):

    CORRUPTION_FUNCTIONS = {
        'noise': add_noise,
        'spike': add_spike,
        'missing_values': add_missing_values,
        'drift': add_drift,
    }

    def __init__(self, *args, corruption_ratio=1.0, corruptions=None, seed=42, **kwargs):
        super().__init__(*args, **kwargs)

        if not 0 <= corruption_ratio <= 1:
            raise ValueError('corruption_ratio must be between 0 and 1')

        self.corruption_ratio = corruption_ratio
        self.rng = np.random.default_rng(seed)

        corruptions = list(self.CORRUPTION_FUNCTIONS.keys() if corruptions is None else corruptions)

        unknown = set(corruptions) - set(self.CORRUPTION_FUNCTIONS.keys())
        if unknown:
            raise ValueError(f'invalid corruptions: {unknown}')

        if self.corruption_ratio > 0 and len(corruptions) == 0:
            raise ValueError('at least one corruption must be provided when corruption_ratio gt 0')

        self.corruption_names = corruptions
        self.corruption_functions = [self.CORRUPTION_FUNCTIONS[name] for name in corruptions]
        n_corrupt = int(len(self) * self.corruption_ratio)
        self.corrupt_indices = set(self.rng.choice(len(self), size=n_corrupt, replace=False).tolist())

        self.index_to_corruption = {}
        for idx in self.corrupt_indices:
            fn_idx = self.rng.integers(0, len(self.corruption_functions))
            self.index_to_corruption[idx] = self.corruption_functions[fn_idx]

    def __getitem__(self, item):
        x, y = super().__getitem__(item)

        if item in self.corrupt_indices:
            corruption_function = self.index_to_corruption[item]
            x = corruption_function(x, self.rng)

        return x, y
