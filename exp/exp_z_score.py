import json
import os
import numpy as np

from pathlib import Path
from exp.exp_ad import get_max_f1_score_threshold, get_metrics


def prepare_data(x):
    x = x.transpose(1, 0, 2)
    x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
    return x


class ZScoreExperiment:

    def __init__(self, exp):
        self.exp = exp
        self.results = {}

        self.mean = None
        self.std = None

    def load_data(self, flag):
        ds_path = os.getenv('DATASET_PATH')
        if ds_path is None:
            path = Path(self.exp.root_path)
        else:
            path = Path(ds_path).joinpath(self.exp.root_path)

        x = np.load(path.joinpath(f'{flag}_x.npy'))
        y = np.load(path.joinpath(f'{flag}_y.npy'))

        y[y > 0] = 1

        x = prepare_data(x)
        y = prepare_data(y)
        y = y.reshape(-1)

        if x.shape[0] != y.shape[0]:
            raise ValueError('shape mismatch')

        return x, y

    def train(self):
        x, y = self.load_data('train')

        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)

    def score(self, x):
        z = np.abs((x - self.mean) / self.std)

        if z.shape[1] == 1:
            return z[:, 0]

        if not hasattr(self.exp, 'z_score_agg') or self.exp.z_score_agg is None or self.exp.z_score_agg == 'l2':
            return np.linalg.norm(z, axis=1)
        elif self.exp.z_score_agg == 'max':
            return np.max(z, axis=1)
        else:
            raise ValueError(f'unknown z-score agg {self.exp.z_score_agg}')

    def find_threshold(self):
        val_x, val_y = self.load_data('val')
        val_score = self.score(val_x)

        threshold_max_f1_score = get_max_f1_score_threshold(y_true=val_y, y_score=val_score)
        return threshold_max_f1_score

    def test(self):
        test_x, test_y = self.load_data('test')
        test_score = self.score(test_x)

        thresh = self.find_threshold()
        test_pred = (test_score > thresh).astype(int)

        metrics = {
            'results': self.results,
            'eval': get_metrics(test_y, test_pred, point_adjust=False, threshold=thresh)
        }

        metrics_json = json.dumps(metrics, indent=4)
        print(metrics_json)

        with open(self.exp.output_folder.joinpath('metrics.json'), 'w') as f:
            f.write(metrics_json)
