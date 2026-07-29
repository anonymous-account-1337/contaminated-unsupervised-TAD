import json
import os
import numpy as np

from sklearn.ensemble import IsolationForest
from pathlib import Path

from exp.exp_ad import get_max_f1_score_threshold, get_metrics


def prepare_data(x):
    x = x.transpose(1, 0, 2)
    x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
    return x


# def pw_to_sw_label(y):
#     """transform point-wise labels to sample-wise labels. If one timestep of a sample is anomalous, the whole sample will be considered to be anomalous"""
#     return np.any(y, axis=1).astype(np.int32)


def map_label(label):
    if label == 1:
        return 0
    elif label == -1:
        return 1
    else:
        raise ValueError(f'invalid label {label}')


class IsolationForestExperiment:

    def __init__(self, exp):
        self.exp = exp
        self.results = {}
        self.model = self.create_model()

    def create_model(self):
        return IsolationForest(n_estimators=self.exp.n_estimators,
                               max_samples=self.exp.max_samples,
                               contamination=self.exp.contamination,
                               max_features=self.exp.max_features,
                               random_state=self.exp.seed,
                               bootstrap=False,
                               warm_start=False,
                               verbose=0,
                               n_jobs=1)

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
        self.model.fit(X=x)

    def score(self, x):
        """sklearn implementation uses negated score of the original paper"""
        return -self.model.score_samples(x)  # 0 < s <= 1; higher scores indicate higher probabilities of anomalies

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
            'eval': get_metrics(test_y, test_pred, point_adjust=False, threshold=thresh),
        }

        metrics_json = json.dumps(metrics, indent=4)
        print(metrics_json)

        with open(self.exp.output_folder.joinpath('metrics.json'), 'w') as f:
            f.write(metrics_json)
