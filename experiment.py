import itertools
import json
import logging
import shutil
import numpy as np
import pandas as pd
from hashlib import sha256
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass
from multiprocessing import Pool

from exp.exp_ad import AnomalyDetectionExperiment
from exp.exp_isolation_forest import IsolationForestExperiment
from exp.exp_z_score import ZScoreExperiment
from exp.exp_mahalanobis import MahalanobisExperiment
from utils.definition import expand


logger = logging.getLogger(__name__)


def _hash_dict(d, excluded_keys=None):
    """calculates a hash based on all current attributes"""

    if excluded_keys is None:
        excluded_keys = []

    h = sha256()
    for k, v in d.items():
        if k in excluded_keys or k.startswith('_'):
            continue

        h.update(k.encode('utf-8'))
        h.update(str(v).encode('utf-8'))
    return h.hexdigest()


class Experiment:

    EXP_TYPE = {
        'deep-learning': AnomalyDetectionExperiment,
        'isolation-forest': IsolationForestExperiment,
        'z-score': ZScoreExperiment,
        'mahalanobis': MahalanobisExperiment,
    }

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)

        self.id = _hash_dict(self.__dict__, excluded_keys=['id'])

        if hasattr(self, 'output_folder'):
            self.output_folder = Path(self.output_folder).joinpath(self.id)
        else:
            raise ValueError('experiment requires an output folder')

    def __getattr__(self, item):
        if item.startswith('_'):
            raise AttributeError
        else:
            """try to access private attributes by prepending an underscore"""
            return self.__getattribute__('_' + item)

    def save(self):
        _d = deepcopy(self.__dict__)
        for key in list(_d.keys()):
            if _d[key] is None:
                continue

            if not isinstance(_d[key], (int, bool, float, str)):
                del _d[key]

        with open(self.output_folder.joinpath('experiment.json'), mode='w', encoding='utf-8') as f:
            json.dump(_d, f, indent=4)

    def print(self, row_length=120):
        print('=' * row_length)
        for k, v in self.__dict__.items():
            key_string = f'{k}: '
            value_string = str(v)
            print(f'{key_string}{" " * (row_length - len(key_string) - len(value_string))}{value_string}')
        print('=' * row_length)

    def get_experiment_cls(self):
        if hasattr(self, 'exp_type'):
            return self.EXP_TYPE[self.exp_type]

        return AnomalyDetectionExperiment

    def run(self, force=False):
        if self.output_folder.exists():
            if force:
                print(f'overwriting completed experiment {self.id}')
                shutil.rmtree(self.output_folder)
            else:
                print(f'skipping already completed experiment {self.id}')
                return

        self.output_folder.mkdir(exist_ok=True, parents=True)
        self.save()
        self.print()

        exp = self.get_experiment_cls()(self)
        exp.train()
        exp.test()

    @classmethod
    def parse(cls, **kwargs):
        d = kwargs['default']
        e = kwargs['experiments']
        e = expand(e, 'quick_def', 'grid_search')
        e = [{**d, **i} for i in e]
        return [cls(**i) for i in e]

    @classmethod
    def of(cls, path, exp=None, recursive=True, recursion_depth=0, verbose=False):
        if exp is None:
            exp = []

        if isinstance(path, list):
            path_list = list(map(Path, path))
        else:
            path_list = [Path(path)]

        for path in path_list:
            if path.is_file():
                exp.extend(cls.from_file(path, verbose=verbose))
            elif path.is_dir():
                if recursive or recursion_depth == 0:
                    for sub_path in path.iterdir():
                        cls.of(sub_path, exp=exp, recursive=recursive, recursion_depth=recursion_depth + 1, verbose=verbose)
            else:
                raise ValueError(f'invalid path type of {path}')

        return exp

    @classmethod
    def from_file(cls, path, verbose=False):
        with open(path, mode='r', encoding='utf-8') as f:
            experiments = cls.parse(**json.load(f))
            if verbose:
                print(f'successfully loaded {len(experiments)} experiment(s) of file {path}.')
            return experiments

    def __str__(self):
        content = ", ".join([f'{k}={v}' for k, v in self.__dict__.items()])
        return f'Experiment({content})'


def _flatten(v, k=None, l=None):
    if l is None:
        l = []

    if isinstance(v, dict):
        for key, val in v.items():
            new_key = key if k is None else f'{k}_{key}'
            _flatten(val, new_key, l)
    else:
        l.append((k, v))

    return dict(l)


@dataclass
class FinishedExperiment:

    experiment: dict
    metrics: dict
    exception: str
    ds_def: dict
    path: Path

    EQUIVALENT_ID_EXCLUDED_KEYS = ['output_folder', 'id', 'iteration', 'gpu', 'gpu_type', 'use_gpu', 'use_multi_gpu', 'checkpoints', 'num_workers', 'plot', 'seed']


    def __getitem__(self, item):
        return self.__getattribute__(item)

    @property
    def equivalent_id(self):
        """finished experiments with the same equivalent id had the same experiment parameters / experiment setup"""
        return _hash_dict(self.experiment, excluded_keys=FinishedExperiment.EQUIVALENT_ID_EXCLUDED_KEYS)

    @staticmethod
    def create_summary(finished_experiments, merge_iterations=False):
        df = pd.DataFrame(map(lambda e: {**_flatten(e.__dict__), **{'equivalent_id': e.equivalent_id}}, finished_experiments))

        if merge_iterations:
            df.drop(['experiment_id', 'experiment_iteration'], axis=1, errors='ignore', inplace=True)
            group = df.groupby(by='equivalent_id')

            numeric_metric_columns = list(filter(lambda col: col.startswith('metrics'), df.select_dtypes(include='number').columns))
            metrics = group[numeric_metric_columns].agg(['mean', 'std'])
            metrics.columns = ['_'.join(col).strip() for col in metrics.columns.values]

            experiment_columns = list(filter(lambda col: col.startswith('experiment') or col.startswith('ds_def'), df.columns))
            def _only_equal(s):
                if any(map(lambda e: isinstance(e[1], list), s.items())):
                    return None

                unique = s.unique()
                if len(unique) == 1:
                    return unique[0]
                else:
                    return 'NOT-EQUAL'

            experiments = group[experiment_columns].aggregate(_only_equal)
            df = experiments.join(metrics, how='inner')
            df['iterations'] = group.size()
        else:
            df.set_index(keys=['experiment_id'], drop=True, inplace=True)

        return df

    @staticmethod
    def is_valid_folder(p):
        if not p.is_dir():
            return False

        if len(p.name) != 64:
            return False

        for c in p.name:
            if not ('0' <= c <= '9' or 'a' <= c <= 'z'):
                return False

        return True

    @classmethod
    def from_folder(cls, p, only_successful=True, ds_root=None):
        p = Path(p)

        if not cls.is_valid_folder(p):
            raise ValueError(f'{p} is no finished experiment folder')

        with open(p.joinpath('experiment.json'), 'r') as f:
            experiment_setup = json.load(f)

        try:
            with open(p.joinpath('metrics.json'), 'r') as f:
                metrics = json.load(f)
        except FileNotFoundError as e:
            metrics = None

        try:
            with open(p.joinpath('exception.txt'), 'r') as f:
                exception = f.read()
        except FileNotFoundError as e:
            exception = None

        ds_def = None
        if ds_root is not None:
            ds_root = Path(ds_root)

            try:
                with open(ds_root.joinpath(experiment_setup['root_path']).joinpath('def.json'), 'r') as f:
                    ds_def = json.load(f)
            except FileNotFoundError as e:
                pass

        e = FinishedExperiment(experiment_setup, metrics, exception, ds_def, p)

        if only_successful and not e.successful:
            raise ValueError(f'experiment {p} not successfully finished.')

        return e

    @classmethod
    def from_folders(cls, p, skip_invalid=True, **kwargs):
        if isinstance(p, list):
            return list(itertools.chain(*[cls.from_folders(e, skip_invalid=skip_invalid, **kwargs) for e in p]))
        else:
            p = Path(p)

        if not p.is_dir():
            raise ValueError('p must be a directory')

        folders = []
        for folder in p.iterdir():
            if not FinishedExperiment.is_valid_folder(folder):
                if skip_invalid:
                    continue
                else:
                    raise ValueError(f'found invalid experiment {folder}')

            folders.append(folder)

        experiments = []
        with Pool() as p:
            futures = [p.apply_async(FinishedExperiment.from_folder, args=(folder, ), kwds=kwargs) for folder in folders]

            for future in futures:
                try:
                    experiments.append(future.get())
                except ValueError as e:
                    logger.warning(e)

        return experiments

    @property
    def test_energy(self):
        return np.load(self.path.joinpath('test_energy.npy'))

    @property
    def test_labels(self):
        return np.load(self.path.joinpath('test_labels.npy'))

    @property
    def val_energy(self):
        return np.load(self.path.joinpath('val_energy.npy'))

    @property
    def val_labels(self):
        return np.load(self.path.joinpath('val_labels.npy'))

    @property
    def successful(self):
        return self.metrics is not None

    @property
    def failed(self):
        return self.exception is not None
