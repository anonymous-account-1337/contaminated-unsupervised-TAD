import json
import numpy as np

from pathlib import Path
from utils.plot import avg_curve
from exp.exp_ad import get_max_f1_score_threshold, get_metrics


def prepare_data(x):
    x = x.transpose(1, 0, 2)
    return x.reshape(x.shape[0], x.shape[1] * x.shape[2])


class ZScoreExperiment:

    def __init__(self, exp):
        self.exp = exp
        self.results = {}

        self.mean = None
        self.std = None

    def load_data(self, flag):
        x = np.load(Path(self.exp.root_path).joinpath(f'{flag}_x.npy'))
        y = np.load(Path(self.exp.root_path).joinpath(f'{flag}_y.npy'))

        y[y > 0] = 1

        x = prepare_data(x)
        y = prepare_data(y)

        return x, y

    def train(self):
        x, y = self.load_data('train')
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

        avg_curve(self.mean, self.std, out_file=self.exp.output_folder.joinpath('avg.png'))

    def score(self, x):
        return np.abs((x - self.mean) / self.std)

    def find_threshold(self):
        val_x, val_y = self.load_data('val')

        val_score = self.score(val_x).reshape(-1)
        val_y = val_y.reshape(-1)

        threshold_max_f1_score = get_max_f1_score_threshold(y_true=val_y, y_score=val_score)
        return threshold_max_f1_score

    def test(self):
        test_x, test_y = self.load_data('test')

        test_score = self.score(test_x).reshape(-1)
        test_y = test_y.reshape(-1)

        thresh = self.find_threshold()
        test_pred = (test_score > thresh).astype(int)

        metrics = {
            'results': self.results,
            'max-score': get_metrics(test_y, test_pred, point_adjust=False, threshold=thresh),
            'pa-max-score': get_metrics(test_y, test_pred, point_adjust=True, threshold=thresh),
        }

        metrics_json = json.dumps(metrics, indent=4)
        print(metrics_json)

        with open(self.exp.output_folder.joinpath('metrics.json'), 'w') as f:
            f.write(metrics_json)
