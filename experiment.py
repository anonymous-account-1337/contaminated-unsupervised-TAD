import itertools
import json
import logging
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from exp.exp_ad import AnomalyDetectionExperiment
from exp.exp_isolation_forest import IsolationForestExperiment
from exp.exp_z_score import ZScoreExperiment
from exp.exp_mahalanobis import MahalanobisExperiment
from utils.definition import expand
from utils.dict_util import hash_dict, flatten_dict, merge_dicts


logger = logging.getLogger(__name__)


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

        self.id = hash_dict(self.__dict__, excluded_keys=['id'])

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
                fe = FinishedExperiment.from_folder(self.output_folder, only_successful=False)
                if fe.successful:
                    print(f'skipping already completed experiment {self.id}')
                    return
                else:
                    print(f'overwriting failed experiment {self.id}')
                    shutil.rmtree(self.output_folder)

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
        e = [{**d, **i} for i in e]
        e = expand(e, 'quick_def', 'grid_search')
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


@dataclass
class MergedFinishedExperiment:

    experiment: dict
    metrics: dict
    ds_def: dict
    hashes: dict = field(default_factory=dict)

    @classmethod
    def from_folders(cls, p, skip_invalid=True, seeds=None, **kwargs):
        """load finished experiments and merge them. Optionally, include only experiments with specific seeds"""
        fe_list = FinishedExperiment.from_folders(p, skip_invalid, **kwargs)

        if seeds is not None:
            fe_list = list(filter(lambda e: e.experiment['seed'] in seeds, fe_list))

        len_fe_list = len(fe_list)
        merged_list = cls.from_finished_experiments(fe_list)
        len_merged_list = len(merged_list)

        logger.info(f'Loaded {len_fe_list} experiment(s) and merged them into {len_merged_list} experiment(s).')

        return merged_list

    @classmethod
    def from_finished_experiments(cls, fe_list):
        d = {}

        for fe in fe_list:
            if fe.setup_id in d:
                d[fe.setup_id].append(fe)
            else:
                d[fe.setup_id] = [fe]

        l = []
        for setup_id in d:
            fe0 = d[setup_id][0]
            exp = fe0.experiment
            for excluded_key in FinishedExperiment.SETUP_ID_EXCLUDED_KEYS:
                exp.pop(excluded_key, None)

            metrics = [fe.metrics for fe in d[setup_id]]
            metrics = merge_dicts(*metrics)
            l.append(cls(experiment=exp, metrics=metrics, ds_def=fe0.ds_def))
        return l

    @property
    def setup_id(self):
        return hash_dict(self.experiment)

    def to_dict(self):
        d = flatten_dict(self.experiment, k='exp') | flatten_dict(self.ds_def, k='ds') | flatten_dict(self.metrics, k='m') | flatten_dict(self.hashes, k='h')
        d['setup_id'] = self.setup_id
        return d

    def add_hash(self, name, included_keys=None, excluded_keys=None):
        self.hashes[name] = hash_dict(self.experiment, included_keys=included_keys, excluded_keys=excluded_keys)

    @staticmethod
    def to_data_frame(fe_list):
        df = pd.DataFrame(map(lambda e: e.to_dict(), fe_list))
        df.set_index(keys=['setup_id'], drop=True, inplace=True)
        return df


@dataclass
class FinishedExperiment:

    experiment: dict
    metrics: dict
    exception: str
    ds_def: dict
    path: Path
    hashes: dict = field(default_factory=dict)

    SETUP_ID_EXCLUDED_KEYS = ['id', 'seed', 'output_folder']

    @property
    def setup_id(self):
        """finished experiments with the same setup id had the same experiment setup"""
        return hash_dict(self.experiment, excluded_keys=self.SETUP_ID_EXCLUDED_KEYS)

    def add_hash(self, name, included_keys=None, excluded_keys=None):
        self.hashes[name] = hash_dict(self.experiment, included_keys=included_keys, excluded_keys=excluded_keys)

    def to_dict(self):
        d = flatten_dict(self.experiment, k='exp') | flatten_dict(self.ds_def, k='ds') | flatten_dict(self.metrics, k='m') | flatten_dict(self.hashes, k='h')
        d['setup_id'] = self.setup_id
        d['exception'] = self.exception
        d['path'] = str(self.path)
        return d

    @staticmethod
    def to_data_frame(finished_experiments):
        df = pd.DataFrame(map(lambda e: e.to_dict(), finished_experiments))
        df.set_index(keys=['exp_id'], drop=True, inplace=True)
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
        with ThreadPoolExecutor() as p:
            futures = [p.submit(FinishedExperiment.from_folder, p=folder, **kwargs) for folder in folders]

            for future in as_completed(futures):
                try:
                    experiments.append(future.result())
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
