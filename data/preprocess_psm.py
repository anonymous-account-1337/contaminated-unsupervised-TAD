import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler


def get_options():
    # https://github.com/eBay/RANSynCoders/tree/main
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-w', '--win-size', type=int, default=100)
    args = parser.parse_args()
    args.input = Path(args.input)
    args.output = Path(args.output)
    return args


def load_labels(p):
    return pd.read_csv(p).to_numpy()[:, 1:]


def add_windows(x, win_size):
    num_windows = x.shape[0] // win_size
    new_length = num_windows * win_size
    x = np.reshape(x[:new_length, :], (num_windows, win_size, x.shape[1]))
    return x


def load_raw_data(p, scaler=None, scaling=True):
    x = pd.read_csv(p).to_numpy()[:, 1:]
    x = np.nan_to_num(x)

    if scaling:
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(x)

        x = scaler.transform(x)
    return x, scaler


def load_data(x, win_size, y=None):
    if y is not None:
        if x.shape[0] != y.shape[0]:
            raise ValueError('shape mismatch')
        y = add_windows(y, win_size)

    x = add_windows(x, win_size)
    return x, y


def save_data(output_folder, flag, x, y):
    output_folder.mkdir(exist_ok=True)

    x = x.transpose(1, 0, 2)
    np.save(output_folder.joinpath(f'{flag}_x.npy'), x)
    print(f'x-{flag:<8s} = {x.shape}')

    if y is not None:
        y = y.transpose(1, 0, 2)
        np.save(output_folder.joinpath(f'{flag}_y.npy'), y)
        print(f'y-{flag:<8s} = {y.shape}')


def main():
    opt = get_options()
    raw_train_x, scaler = load_raw_data(opt.input.joinpath('train.csv'))
    raw_test_x, _ = load_raw_data(opt.input.joinpath('test.csv'), scaler=scaler)

    test_labels = load_labels(opt.input.joinpath('test_label.csv'))
    test_x, test_y = load_data(raw_test_x, win_size=opt.win_size, y=test_labels)
    train_x, train_y = load_data(raw_train_x, win_size=opt.win_size)

    save_data(opt.output, 'test', test_x, test_y)
    save_data(opt.output, 'train', train_x, train_y)


if __name__ == '__main__':
    main()
