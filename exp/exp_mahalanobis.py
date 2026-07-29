import json
import numpy as np

from pathlib import Path
from scipy.spatial.distance import mahalanobis
from utils.plot import avg_curve
from exp.exp_ad import get_max_f1_score_threshold, get_metrics


def prepare_data(x):
    x = x.transpose(1, 0, 2)
    return x.reshape(x.shape[0], x.shape[1] * x.shape[2])


def pw_to_sw_label(y):
    """transform point-wise labels to sample-wise labels. If one timestep of a sample is anomalous, the whole sample will be considered to be anomalous"""
    return np.any(y, axis=1).astype(np.int32)


class MahalanobisExperiment:

    def __init__(self, exp):
        self.exp = exp
        self.results = {}

        self.mean = None
        self.sigma = None
        self.sigma_inv = None

    def load_data(self, flag):
        x = np.load(Path(self.exp.root_path).joinpath(f'{flag}_x.npy'))
        y = np.load(Path(self.exp.root_path).joinpath(f'{flag}_y.npy'))

        y[y > 0] = 1

        x = prepare_data(x)
        y = prepare_data(y)
        y = pw_to_sw_label(y)

        return x, y

    def train(self):
        x, y = self.load_data('train')

        self.mean = np.mean(x, axis=0)
        self.sigma = np.cov(x, rowvar=False)
        self.sigma_inv = np.linalg.pinv(self.sigma)

        avg_curve(self.mean, np.diagonal(self.sigma), out_file=self.exp.output_folder.joinpath('avg.png'))

    def score(self, x):
        if len(x.shape) == 1:
            return mahalanobis(x, self.mean, self.sigma_inv)
        elif len(x.shape) == 2:
            return np.array([self.score(x[i]) for i in range(len(x))])
        else:
            raise ValueError('invalid shape')

    def find_threshold(self):
        val_x, val_y = self.load_data('val')
        val_score = self.score(val_x)

        threshold_max_f1_score = get_max_f1_score_threshold(y_true=val_y, y_score=val_score)
        return threshold_max_f1_score

    def test(self):
        thresh = self.find_threshold()
        test_x, test_y = self.load_data('test')

        test_score = self.score(test_x)
        test_pred = (test_score > thresh).astype(int)

        metrics = {
            'results': self.results,
            'max-score': get_metrics(test_y, test_pred, point_adjust=False, threshold=thresh),
        }

        metrics_json = json.dumps(metrics, indent=4)
        print(metrics_json)

        with open(self.exp.output_folder.joinpath('metrics.json'), 'w') as f:
            f.write(metrics_json)
