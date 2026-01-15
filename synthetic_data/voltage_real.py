import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.interpolate import make_smoothing_spline
from synthetic_data.base import BaseFunction, Interval, IntervalType
from synthetic_data.util import array_wrapper_method


def sample_gen(df, num_dp):
    groups = df.groupby(by='start_time')

    for start_time, s in groups:
        ts = s['m1_ankerspannung'].to_numpy()
        if num_dp is None:
            yield ts
        else:
            # interpolate and return num_dp data points
            x = np.linspace(0, len(ts), num_dp)
            y = np.interp(x, np.arange(0, len(ts)), ts)
            yield y.reshape(-1, 1)


def load_voltage_data(path, num_dp=None):
    df = pd.read_feather(path)
    return list(sample_gen(df, num_dp))


def plot(x, y, intervals=None, out_file=None):
    if intervals is None:
        intervals = []

    fig, ax = plt.subplots()
    ax.plot(x, y)

    for i in intervals:
        ax.vlines(x[min(i[0], len(x) - 1)], np.min(y), np.max(y))
        ax.vlines(x[min(i[1], len(x) - 1)], np.min(y), np.max(y))

    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)
    plt.close(fig)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def smooth(x, y, lam=10000, window_length=4):
    spline = make_smoothing_spline(x, y, lam=lam)
    return moving_average(spline(x), w=window_length)


def get_section_labels(s_grad, min_dp_count=2):
    sections = s_grad > 0.2
    section_start = 0
    section_labels = np.zeros(shape=sections.shape)
    section_label = 1

    intervals = []

    for i in range(len(sections) - 1):
        if sections[i] != sections[i + 1]:
            # transitions indicate sections
            if i - section_start < min_dp_count:
                # section is obviously too short
                continue

            section_labels[section_start:i] = section_label
            intervals.append((section_start, i))
            section_label += 1
            section_start = i

    section_labels[section_start:] = section_label
    intervals.append((section_start, len(section_labels)))
    return intervals, section_labels


def preprocess(prep_folder, input_path, max_x=None, num_dp=100, smoothing=True):
    prep_folder = Path(prep_folder)
    prep_folder.mkdir(exist_ok=True)

    if input_path.endswith('.feather'):
        # drain0 file
        df = load_voltage_data(input_path)
    elif input_path.endswith('.npy'):
        # numpy file
        df = np.load(input_path)
    else:
        raise ValueError(f'invalid input file {input_path}')

    ds_list = []
    segment_list = []

    for i, y_orig in enumerate(df):
        if max_x is None:
            x_orig = np.arange(0, len(y_orig))
            x = np.linspace(0, len(x_orig), num_dp)
            y = np.interp(x, x_orig, y_orig)
        else:
            y = y_orig.reshape(-1)
            x = np.linspace(0, max_x, num_dp)

        if smoothing:
            y = smooth(x, y)

        y_grad = np.gradient(y)
        y_grad = np.abs(moving_average(y_grad, 4))
        y_grad /= y_grad.max()

        intervals, section_labels = get_section_labels(y_grad)

        if len(intervals) != 7:
            continue

        ds_list.append([x, y])
        segment_list.append(intervals)

    ds_list = np.array(ds_list)
    segment_list = np.array(segment_list)

    train_slice = slice(0, int(len(ds_list) * 0.7))
    test_slice = slice(int(len(ds_list) - len(ds_list) * 0.15), len(ds_list))
    val_slice = slice(train_slice.stop, test_slice.start)

    np.save(prep_folder.joinpath('train.npy'), ds_list[train_slice])
    np.save(prep_folder.joinpath('train_segments.npy'), segment_list[train_slice])

    np.save(prep_folder.joinpath('val.npy'), ds_list[val_slice])
    np.save(prep_folder.joinpath('val_segments.npy'), segment_list[val_slice])

    np.save(prep_folder.joinpath('test.npy'), ds_list[test_slice])
    np.save(prep_folder.joinpath('test_segments.npy'), segment_list[test_slice])


def load_preprocessed(prep_folder):
    prep_folder = Path(prep_folder)

    train = np.load(prep_folder.joinpath('train.npy'))
    train_segments = np.load(prep_folder.joinpath('train_segments.npy'))

    val = np.load(prep_folder.joinpath('val.npy'))
    val_segments = np.load(prep_folder.joinpath('val_segments.npy'))

    test = np.load(prep_folder.joinpath('test.npy'))
    test_segments = np.load(prep_folder.joinpath('test_segments.npy'))

    return {
        'train': (train, train_segments),
        'val': (val, val_segments),
        'test': (test, test_segments),
    }


class VoltageRealFunction(BaseFunction):

    CACHE = {}
    SEGMENTS = ('start', 'acceleration', 'plateau', 'deceleration', 'move_back', 'move_back_deceleration', 'stop')

    def __init__(self, ds_path, sample_idx, split, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if ds_path in self.CACHE:
            self.ds = self.CACHE[ds_path]
        else:
            self.ds = load_preprocessed(ds_path)
            self.CACHE[ds_path] = self.ds

        self.idx = sample_idx
        samples, self.segments = self.ds[split]
        self.x = samples[:, 0]
        self.y = samples[:, 1]

    def get_effective_index(self):
        return self.idx % len(self.y)

    def current_x(self):
        return self.x[self.get_effective_index()]

    def current_y(self):
        return self.y[self.get_effective_index()]

    def current_intervals(self):
        return self.segments[self.get_effective_index()]

    def get_segment_interval(self, name: str):
        for segment_name, interval in zip(self.SEGMENTS, self.current_intervals()):
            if segment_name == name:
                return Interval(float(self.current_x()[interval[0]]), float(self.current_x()[min(interval[1], len(self.current_x()) - 1)]), IntervalType.RIGHT_OPEN)

        return super().get_segment_interval(name)

    @property
    def domain(self):
        return Interval(float(self.current_x()[0]), float(self.current_x()[-1]), IntervalType.CLOSED)

    @array_wrapper_method
    def __call__(self, x):
        return np.interp(x, self.current_x(), self.current_y())
