import json
import numpy as np

from pathlib import Path
from dataclasses import dataclass

from utils.definition import expand_quick_def
from synthetic_data.voltage_plf import VoltageFunction
from synthetic_data.voltage_real import VoltageRealFunction
from synthetic_data.base import SineFunction, CosineFunction
from synthetic_data.anomaly import AnomalySet, Anomaly
from synthetic_data.util import shift, pad_or_cut, point_wise_anomaly_ratio, euclidean_distance


class CustomJSONEncoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, Path):
            return str(o)
        elif isinstance(o, AnomalySet):
            return o.anomalies
        elif isinstance(o, (DatasetDefinition, Anomaly)):
            return o.__dict__

        return super().default(o)


@dataclass
class DatasetDefinition:

    BASIS_FUNCTION_MAP = {
        'voltage': VoltageFunction,
        'voltage-real': VoltageRealFunction,
        'sine': SineFunction,
        'cosine': CosineFunction,
    }

    name: str
    basis_function: str
    basis_function_args: dict
    dp_shift_mean: float
    dp_shift_std: float
    num_dp: int
    sample_interval: int
    num_samples: int
    test_ratio: float
    val_ratio: float
    anomaly_ratio_train: float
    anomaly_ratio_val: float
    anomaly_ratio_test: float
    anomalies: AnomalySet
    scaling: str = None

    @classmethod
    def of(cls, path, l=None, recursive=True, recursion_depth=0):
        path = Path(path)

        if l is None:
            l = []

        if path.is_file():
            l.extend(cls.from_file(path))
        elif path.is_dir():
            if recursive or recursion_depth == 0:
                for sub_path in path.iterdir():
                    cls.of(sub_path, l=l, recursive=recursive, recursion_depth=recursion_depth + 1)
        else:
            raise ValueError(f'invalid path type of {path}')

        return l

    @classmethod
    def from_file(cls, path):
        with open(path, 'r') as f:
            return cls.parse(**json.load(f))

    @classmethod
    def parse(cls, **kwargs):
        d = kwargs['default']
        e = kwargs['datasets']
        e = expand_quick_def(e, 'quick')
        e = [{**d, **i} for i in e]

        for i in e:
            i['anomalies'] = AnomalySet.parse(i['anomalies'])

        return [cls(**i) for i in e]

    @property
    def num_test_samples(self):
        return int(self.num_samples * self.test_ratio)

    @property
    def num_val_samples(self):
        return int(self.num_samples * self.val_ratio)

    @property
    def num_train_samples(self):
        return self.num_samples - self.num_val_samples - self.num_test_samples

    @property
    def num_test_anomaly_samples(self):
        return int(self.num_test_samples * self.anomaly_ratio_test)

    @property
    def num_val_anomaly_samples(self):
        return int(self.num_val_samples * self.anomaly_ratio_val)

    @property
    def num_train_anomaly_samples(self):
        return int(self.num_train_samples * self.anomaly_ratio_train)

    def get_basis_function(self):
        return self.BASIS_FUNCTION_MAP[self.basis_function]

    def generate_sample(self, sample_idx: int, split: str):
        if split == 'train':
            num_anomaly_samples = self.num_train_anomaly_samples
        elif split == 'val':
            num_anomaly_samples = self.num_val_anomaly_samples
        elif split == 'test':
            num_anomaly_samples = self.num_test_anomaly_samples
        else:
            raise ValueError(f'invalid split {split}')

        base_function_args = {**self.basis_function_args, 'sample_idx': sample_idx, 'split': split}
        f = self.get_basis_function()(**base_function_args)
        x = np.arange(f.domain.start, f.domain.stop, self.sample_interval)
        y = f(x)
        y_clean = np.copy(y)  # sample without any anomalies added
        gt = np.zeros(shape=y.shape)

        if sample_idx < num_anomaly_samples:
            anomaly_applied = False
            while not anomaly_applied:
                try:
                    anomaly: Anomaly = self.anomalies.get_random_anomaly()
                    anomaly.apply(f, x, y, gt)
                    anomaly_applied = True
                except ValueError as e:
                    print(e)
                    anomaly_applied = False

        dp_shift = int(np.random.normal(loc=self.dp_shift_mean, scale=self.dp_shift_std))
        y = shift(y, dp_shift)
        y_clean = shift(y_clean, dp_shift)
        gt = shift(gt, dp_shift)

        y = pad_or_cut(y, self.num_dp)
        y_clean = pad_or_cut(y_clean, self.num_dp)
        gt = pad_or_cut(gt, self.num_dp)

        return y_clean, y, gt

    def generate_split(self, split: str):
        if split == 'train':
            num_samples = self.num_train_samples
        elif split == 'val':
            num_samples = self.num_val_samples
        elif split == 'test':
            num_samples = self.num_test_samples
        else:
            raise ValueError(f'invalid split {split}')

        tasks = [(sample_idx, split) for sample_idx in range(num_samples)]
        res = np.array([self.generate_sample(*task) for task in tasks])

        ds_x_clean = res[:, 0, :]
        ds_x = res[:, 1, :]
        ds_y = res[:, 2, :]
        return ds_x_clean, ds_x, ds_y

    def generate_ds(self):
        train_x_clean, train_x, train_y = self.generate_split('train')
        val_x_clean, val_x, val_y = self.generate_split('val')
        test_x_clean, test_x, test_y = self.generate_split('test')

        if self.scaling is not None:
            if self.scaling == 'minmax':
                _min, _max = np.min(train_x), np.max(train_x)
                train_x = (train_x - _min) / (_max - _min)
                val_x = (val_x - _min) / (_max - _min)
                test_x = (test_x - _min) / (_max - _min)

                train_x_clean = (train_x_clean - _min) / (_max - _min)
                val_x_clean = (val_x_clean - _min) / (_max - _min)
                test_x_clean = (test_x_clean - _min) / (_max - _min)
            elif self.scaling == 'std':
                _mean, _std = np.mean(train_x), np.std(train_x)
                train_x = (train_x - _mean) / _std
                val_x = (val_x - _mean) / _std
                test_x = (test_x - _mean) / _std

                train_x_clean = (train_x_clean - _mean) / _std
                val_x_clean = (val_x_clean - _mean) / _std
                test_x_clean = (test_x_clean - _mean) / _std
            elif isinstance(self.scaling, dict):
                _subtrahend, _divisor = self.scaling['subtrahend'], self.scaling['divisor']
                train_x = (train_x - _subtrahend) / _divisor
                val_x = (val_x - _subtrahend) / _divisor
                test_x = (test_x - _subtrahend) / _divisor

                train_x_clean = (train_x_clean - _subtrahend) / _divisor
                val_x_clean = (val_x_clean - _subtrahend) / _divisor
                test_x_clean = (test_x_clean - _subtrahend) / _divisor
            else:
                raise ValueError(f'invalid scaling "{self.scaling}"')

        transpose_order = (1, 0, 2)  # sequence_length, batch_size, features
        train_x_clean = np.reshape(train_x_clean, (*train_x_clean.shape, 1)).transpose(transpose_order)
        train_x = np.reshape(train_x, (*train_x.shape, 1)).transpose(transpose_order)
        train_y = np.reshape(train_y, (*train_y.shape, 1)).transpose(transpose_order)

        val_x_clean = np.reshape(val_x_clean, (*val_x_clean.shape, 1)).transpose(transpose_order)
        val_x = np.reshape(val_x, (*val_x.shape, 1)).transpose(transpose_order)
        val_y = np.reshape(val_y, (*val_y.shape, 1)).transpose(transpose_order)

        test_x_clean = np.reshape(test_x_clean, (*test_x_clean.shape, 1)).transpose(transpose_order)
        test_x = np.reshape(test_x, (*test_x.shape, 1)).transpose(transpose_order)
        test_y = np.reshape(test_y, (*test_y.shape, 1)).transpose(transpose_order)

        if train_x.shape != train_y.shape:
            raise ValueError('shape mismatch')

        if val_x.shape != val_y.shape:
            raise ValueError('shape mismatch')

        if test_x.shape != test_y.shape:
            raise ValueError('shape mismatch')

        return train_x, train_x_clean, val_x, val_x_clean, test_x, test_x_clean, train_y, val_y, test_y

    def save_to(self, output_folder: Path, force=False, seed=None):
        if seed is not None:
            np.random.seed(seed)

        output_folder = Path(output_folder).joinpath(self.name)
        if output_folder.exists():
            if force:
                print(f'Forcibly overwriting folder {output_folder}')
            else:
                print(f'Ignoring folder {output_folder}')
                return

        output_folder.mkdir(exist_ok=True, parents=True)
        train_x, train_x_clean, val_x, val_x_clean, test_x, test_x_clean, train_y, val_y, test_y = self.generate_ds()

        for flag, x, x_clean, y in [('train', train_x, train_x_clean, train_y), ('val', val_x, val_x_clean, val_y), ('test', test_x, test_x_clean, test_y)]:
            dist = euclidean_distance(x, x_clean)
            self.__setattr__(f'euclidean_distance_{flag}', dist)
            self.__setattr__(f'pointwise_anomaly_ratio_{flag}', point_wise_anomaly_ratio(y))

        with open(output_folder.joinpath('def.json'), mode='w', encoding='utf-8') as f:
            json.dump(self, fp=f, cls=CustomJSONEncoder, indent=4)

        save(output_folder.joinpath('train_x_clean.npy'), train_x_clean)
        save(output_folder.joinpath('train_x.npy'), train_x)
        save(output_folder.joinpath('train_y.npy'), train_y)

        save(output_folder.joinpath('val_x_clean.npy'), val_x_clean)
        save(output_folder.joinpath('val_x.npy'), val_x)
        save(output_folder.joinpath('val_y.npy'), val_y)

        save(output_folder.joinpath('test_x_clean.npy'), test_x_clean)
        save(output_folder.joinpath('test_x.npy'), test_x)
        save(output_folder.joinpath('test_y.npy'), test_y)


def save(output_file: Path, data: np.ndarray):
    print(f'shape of {output_file} = {data.shape}')
    np.save(output_file, data)
