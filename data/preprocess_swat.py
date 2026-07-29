import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler


SCALE_COLS = ['FIT101', 'LIT101', 'MV101', 'P101', 'P102', 'AIT201', 'AIT202', 'AIT203', 'FIT201', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206', 'DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302', 'AIT401', 'AIT402', 'FIT401', 'LIT401', 'P401', 'P402', 'P403', 'P404', 'UV401', 'AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504', 'P501', 'P502', 'PIT501', 'PIT502', 'PIT503', 'FIT601', 'P601', 'P602', 'P603']


def get_options():
    # https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-w', '--win-size', type=int, default=100)
    parser.add_argument('--train-split', type=float, default=0.7)
    parser.add_argument('--val-split', type=float, default=0.15)
    args = parser.parse_args()
    args.input = Path(args.input)
    args.output = Path(args.output)
    return args


def preprocess_df(df):
    df = df.rename(lambda c: c.strip(), axis='columns')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by='Timestamp', ascending=True)
    df['Normal/Attack'] = df['Normal/Attack'].map(lambda e: 0 if e == 'Normal' else 1)
    return df


def generate_continuous_segments(df):
    gap = (df['Timestamp'] - df['Timestamp'].shift(1, fill_value=df['Timestamp'].iloc[-1])).map(lambda e: e.total_seconds())
    gap = gap > 1
    cut_indices = [0] + gap.loc[gap].index.tolist() + [df.index[-1] + 1]

    segments = []
    for i in range(len(cut_indices) - 1):
        start_idx = cut_indices[i]
        end_idx = cut_indices[i + 1] - 1
        segments.append(df.loc[start_idx:end_idx])
    return segments


def split_segments(segments, win_size):
    regular = []
    faulty = []

    for segment in segments:
        for i in range(0, len(segment), win_size):
            window = segment.iloc[i:i + win_size]
            if len(window) != win_size:
                continue

            if 1 in window['Normal/Attack'].unique():
                faulty.append(window)
            else:
                regular.append(window)

    return regular, faulty


def stack_windows(windows, transpose=True):
    x = []
    y = []

    for win in windows:
        win = win.drop(columns=['Timestamp']).to_numpy()
        x.append(win[:, :-1])
        y.append(win[:, -1:])

    x, y = np.stack(x, axis=0), np.stack(y, axis=0)
    if transpose:
        x, y = x.transpose(1, 0, 2), y.transpose(1, 0, 2)
    return x, y


def split_windows(windows, train_split, val_split):
    train_val_split_idx = int(len(windows) * train_split)
    val_test_split_idx = int(len(windows) * (train_split + val_split))

    train_windows = windows[:train_val_split_idx]
    val_windows = windows[train_val_split_idx:val_test_split_idx]
    test_windows = windows[val_test_split_idx:]

    return train_windows, val_windows, test_windows


def save(windows, p, flag):
    if len(windows) == 0:
        print(f'{flag:<8s} = None')
        return

    x, y = stack_windows(windows)
    print(f'{flag:<8s} = {x.shape} {y.shape}')
    np.save(p.joinpath(f'{flag}_x.npy'), x)
    np.save(p.joinpath(f'{flag}_y.npy'), y)


def get_scaler(windows):
    scaler = StandardScaler()
    scaler.fit(pd.concat(windows, axis=0)[SCALE_COLS])

    return scaler


def scale_windows(windows, scaler):
    for win in windows:
        win[SCALE_COLS] = scaler.transform(win[SCALE_COLS])


def main():
    opt = get_options()
    opt.output.mkdir(exist_ok=True)

    df = pd.read_csv(opt.input.joinpath('merged.csv'))
    df = preprocess_df(df)

    # df = df.drop(index=df.index[350:355].tolist())  # simulate gap
    # df = df.drop(index=df.index[500:510].tolist())  # simulate gap

    regular_windows, faulty_windows = split_segments(generate_continuous_segments(df), opt.win_size)
    train_windows, val_windows, test_windows = split_windows(regular_windows, opt.train_split, opt.val_split)
    test_windows.extend(faulty_windows)  # add faulty windows to test data set

    scaler = get_scaler(train_windows)
    scale_windows(train_windows, scaler)
    scale_windows(val_windows, scaler)
    scale_windows(test_windows, scaler)

    save(train_windows, opt.output, 'train')
    save(val_windows, opt.output, 'val')
    save(test_windows, opt.output, 'test')


if __name__ == '__main__':
    main()
