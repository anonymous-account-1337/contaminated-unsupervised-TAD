import argparse
import ast
import numpy as np
import pandas as pd

from pathlib import Path


def get_options():
    # https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-w', '--win-size', type=int, default=100)
    args = parser.parse_args()
    args.input = Path(args.input)
    args.output = Path(args.output)
    return args


def parse_classes(s):
    mapping = {'point': '0', 'contextual': '1'}
    rev_mapping = dict([(int(v), k) for k, v in mapping.items()])

    for k, v in mapping.items():
        s = s.replace(k, v)

    return list(map(lambda e: rev_mapping[e], ast.literal_eval(s)))


def load_test_labels(p, deduplicate=True):
    test_label_df = pd.read_csv(p).set_index(keys=['chan_id'], drop=True)
    test_label_df['anomaly_sequences'] = test_label_df['anomaly_sequences'].map(ast.literal_eval)
    test_label_df['class'] = test_label_df['class'].map(parse_classes)

    if deduplicate:
        dup_channels = test_label_df.index.value_counts()[test_label_df.index.value_counts() > 1].index
        for chan_id in dup_channels:
            dup = test_label_df.loc[chan_id]
            spacecraft = dup['spacecraft'].unique()
            num_values = dup['num_values'].unique()

            if len(spacecraft) != 1 or len(num_values) != 1:
                raise ValueError

            spacecraft = spacecraft[0]
            num_values = num_values[0]
            anomaly_sequences = []
            classes = []

            for _, d in dup.iterrows():
                anomaly_sequences.extend(d['anomaly_sequences'])
                classes.extend(d['class'])

            merged = pd.DataFrame(index=[chan_id], data={
                'spacecraft': [spacecraft],
                'anomaly_sequences': [anomaly_sequences],
                'class': [classes],
                'num_values': [num_values],
            })
            test_label_df = test_label_df.drop(index=chan_id)
            test_label_df = pd.concat([test_label_df, merged], axis=0)

    return test_label_df


def load_data(p, data, test_labels=None):
    for f in p.iterdir():
        channel_id = f.name.split('.')[0]
        x = np.load(f)
        if x.shape[-1] == 55:
            ds = 'MSL'
        elif x.shape[-1] == 25:
            ds = 'SMAP'
        else:
            raise ValueError('invalid shape')

        key = ds + '-' + channel_id
        if key not in data:
            data[key] = {}

        if test_labels is None:
            data[key]['train_x'] = x
        else:
            y = np.zeros(shape=x.shape[0], dtype=np.int32)
            if channel_id in test_labels.index:
                test_label = test_labels.loc[channel_id]

                if test_label['spacecraft'] != ds:
                    raise ValueError('ds mismatch')

                if test_label['num_values'] != x.shape[0]:
                    raise ValueError('shape mismatch')

                for anomaly_seq in test_label['anomaly_sequences']:
                    if len(anomaly_seq) != 2:
                        raise ValueError('invalid anomaly seq')

                    y[anomaly_seq[0]:anomaly_seq[1]] = 1
            y = y.reshape(-1, 1)

            data[key]['test_x'] = x
            data[key]['test_y'] = y


def add_windows(x, win_size):
    num_windows = x.shape[0] // win_size
    new_length = num_windows * win_size
    x = np.reshape(x[:new_length, :], (num_windows, win_size, x.shape[1]))
    return x


def save_data(output_folder, data, win_size):
    for key in data.keys():
        o = output_folder.joinpath(key)
        o.mkdir(exist_ok=True, parents=True)

        print(key)
        for ds in data[key]:
            ds_npy = add_windows(data[key][ds], win_size).transpose(1, 0, 2)
            print(f'{ds:<8s} = {ds_npy.shape}')
            np.save(o.joinpath(f'{ds}.npy'), ds_npy)


def main():
    opt = get_options()

    test_labels = load_test_labels(opt.input.joinpath('labeled_anomalies.csv'))
    data = {}
    load_data(opt.input.joinpath('data/data/test'), data, test_labels)
    load_data(opt.input.joinpath('data/data/train'), data)
    save_data(opt.output, data, win_size=opt.win_size)


if __name__ == '__main__':
    main()
