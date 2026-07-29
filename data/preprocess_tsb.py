import json
import logging
import argparse
import math
import traceback
import numpy as np
import pandas as pd

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from contamination_split import create_split, contamination_train_val_test_split
from utils.metrics import sample_wise_ar, point_wise_ar

logger = logging.getLogger(__name__)


def get_options():
    # https://github.com/thedatumorg/TSB-AD
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-w', '--win-size', type=int, default=100)
    parser.add_argument('--min-samples', type=int, default=-1)
    parser.add_argument('-n', '--num-processes', type=int, default=None)

    parser.add_argument('--split', type=float, nargs=3, default=[0.7, 0.15, 0.15])
    parser.add_argument('--anomaly-ratio-train', type=float, default=None)

    args = parser.parse_args()
    args.input = Path(args.input)
    args.output = Path(args.output)

    for s in args.split:
        if s <= 0:
            raise ValueError('each split must be gt 0')

    if math.isclose(sum(args.split), 1):
        args.split = {'train_ratio': args.split[0], 'val_ratio': args.split[1], 'test_ratio': args.split[2]}
    else:
        raise ValueError('splits do not add up to 1')

    return args


def preprocess_df(df):
    cols = df.columns.tolist()
    cols.remove('Label')
    mapper = dict([(col, f'feat_{idx}') for idx, col in enumerate(cols)])
    df = df.rename(columns=mapper)

    labels = tuple(sorted(df['Label'].unique()))
    if labels != (0, 1):
        raise ValueError('data set contains invalid labels')

    return df


def create_windows(df, win_size):
    regular = []
    faulty = []

    for i in range(0, len(df), win_size):
        window = df.iloc[i:i + win_size]
        if len(window) != win_size:
            continue

        if 1 in window['Label'].unique():
            faulty.append(window)
        else:
            regular.append(window)

    return regular, faulty


def create_scaler(windows):
    df = pd.concat(windows, axis=0).drop(columns=['Label'])
    scaler = StandardScaler()
    scaler.fit(df)
    return scaler


def scale(windows, scaler):
    scaled_windows = []
    for window in windows:
        feats = window.columns.drop('Label')
        new = pd.DataFrame(index=window.index, data=scaler.transform(window[feats]), columns=feats)
        new['Label'] = window['Label']
        scaled_windows.append(new)
    return scaled_windows


def to_numpy(windows):
    x_list = []
    y_list = []

    for window in windows:
        feats = window.columns.drop('Label')

        x = window[feats].to_numpy()
        y = window['Label'].to_numpy().reshape(-1, 1)

        x_list.append(x)
        y_list.append(y)

    x, y = np.array(x_list), np.array(y_list)
    return x, y


def save(x, y, flag, output_folder, transpose=True):
    if len(x) == 0 and len(y) == 0:
        return

    if transpose:
        x = x.transpose(1, 0, 2)
        y = y.transpose(1, 0, 2)

    if x.shape[:-1] != y.shape[:-1]:
        raise ValueError('shape mismatch')

    if y.shape[-1] != 1:
        raise ValueError('invalid label shape')

    # print(f'{output_folder}/{flag:8s} = {x.shape} {y.shape}')
    np.save(output_folder.joinpath(f'{flag}_x'), x)
    np.save(output_folder.joinpath(f'{flag}_y'), y)


def default_train_val_test_split(regular_windows, faulty_windows, val_ratio, test_ratio):
    faulty_half_size = len(faulty_windows) // 2
    n_total = len(regular_windows) + len(faulty_windows)

    test = create_split(regular_windows, faulty_windows, num_reg=int(n_total * test_ratio - faulty_half_size), num_faulty=faulty_half_size)
    val = create_split(regular_windows, faulty_windows, num_reg=int(n_total * val_ratio - len(faulty_windows)), num_faulty=len(faulty_windows))
    train = regular_windows

    return train, val, test


def preprocess_file(input_file, win_size, min_samples, split, anomaly_ratio_train):
    df = preprocess_df(pd.read_csv(input_file))
    regular_windows, faulty_windows = create_windows(df, win_size)

    if anomaly_ratio_train is None:
        train, val, test = default_train_val_test_split(regular_windows, faulty_windows, split['val_ratio'], split['test_ratio'])
    else:
        train, val, test = contamination_train_val_test_split(regular_windows, faulty_windows, anomaly_ratio_train, split['val_ratio'], split['test_ratio'])

    num_samples = len(train) + len(val) + len(test)
    if min_samples > 0 and num_samples < min_samples:
        raise ValueError('too few samples')

    scaler = create_scaler(train)
    train = scale(train, scaler)
    val = scale(val, scaler)
    test = scale(test, scaler)

    train_x, train_y = to_numpy(train)
    val_x, val_y = to_numpy(val)
    test_x, test_y = to_numpy(test)

    return {
        'train': (train_x, train_y),
        'val': (val_x, val_y),
        'test': (test_x, test_y),
    }


def handle_file(file, opt):
    ds = preprocess_file(file, opt.win_size, opt.min_samples, opt.split, opt.anomaly_ratio_train)
    def_d = {
        'file_name': file.name,
        'index': file.name.split('_')[0],
        'name': file.name.split('_')[1],
        'domain': file.name.split('_')[4],
        'anomaly_ratio_train': opt.anomaly_ratio_train,
    }
    def_d |= opt.split

    output_folder = opt.output.joinpath(file.name.split('.')[0])
    output_folder.mkdir(exist_ok=True)

    for split, (x, y) in ds.items():
        save(x, y, split, output_folder)
        def_d[f'{split}_samples'] = x.shape[0]
        def_d[f'{split}_seq_len'] = x.shape[1]
        def_d[f'{split}_features'] = x.shape[2]
        def_d[f'{split}_sw_anomaly_ratio'] = sample_wise_ar(y, batch_first=True)
        def_d[f'{split}_pw_anomaly_ratio'] = point_wise_ar(y)

    def_d['num_samples'] = def_d['train_samples'] + def_d['val_samples'] + def_d['test_samples']
    def_d['actual_train_ratio'] = def_d['train_samples'] / def_d['num_samples']
    def_d['actual_val_ratio'] = def_d['val_samples'] / def_d['num_samples']
    def_d['actual_test_ratio'] = def_d['test_samples'] / def_d['num_samples']

    with open(output_folder.joinpath('def.json'), 'w') as f:
        json.dump(def_d, f, indent=4)


def main():
    opt = get_options()
    opt.output.mkdir(exist_ok=True, parents=True)
    errors = []

    files = []
    for file in opt.input.iterdir():
        if not file.name.endswith('.csv'):
            logger.warning(f'skipping non-csv file {file}')
            continue

        files.append(file)

    if opt.num_processes == 1:
        for file in files:
            try:
                handle_file(file, opt)
            except Exception as ex:
                errors.append({
                    'file': str(file),
                    'type': type(ex).__name__,
                    'message': str(ex),
                })
                traceback.print_exc()
    else:
        with ProcessPoolExecutor(max_workers=opt.num_processes) as pool:
            future_to_file = {}
            futures = []

            for file in files:
                future = pool.submit(handle_file, file, opt)
                future_to_file[future] = file
                futures.append(future)

            for future in tqdm(as_completed(futures), unit=' file(s)', total=len(futures)):
                file = future_to_file[future]
                try:
                    future.result()
                except Exception as ex:
                    errors.append({
                        'file': str(file),
                        'type': type(ex).__name__,
                        'message': str(ex),
                    })

    error_df = pd.DataFrame(errors, columns=['file', 'type', 'message'])
    error_df.to_csv(opt.output.joinpath('errors.csv'), index=False)
    error_df.to_excel(opt.output.joinpath('errors.xlsx'), index=False)


if __name__ == '__main__':
    main()
