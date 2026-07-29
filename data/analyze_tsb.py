import argparse
import json
import pandas as pd

from pathlib import Path
from utils.xlsx import adjust_worksheet_column_width


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('--data-sets', type=str, nargs='+', required=False, default=[])
    parser.add_argument('--min-samples', type=int, required=False, default=100)
    parser.add_argument('--max-features', type=int, required=False, default=70)
    parser.add_argument('--train-ratio-range', type=float, required=False, nargs=2, default=[0.65, 0.75])
    parser.add_argument('--skip-filters', action='store_true', default=False)
    args = parser.parse_args()
    args.input = [Path(i) for i in args.input]
    args.output = Path(args.output)
    args.output.mkdir(exist_ok=True)
    return args


def is_data_set_folder(p: Path):
    if not p.is_dir():
        return False

    actual = [e.name for e in p.iterdir()]
    for expected in ['def.json', 'test_x.npy', 'test_y.npy', 'train_x.npy', 'train_y.npy', 'val_x.npy', 'val_y.npy']:
        if expected not in actual:
            return False

    return True


def resolve_data_set_folders(p: Path, l=None):
    if l is None:
        l = []

    if p.is_dir():
        if is_data_set_folder(p):
            l.append(p)
        else:
            for e in p.iterdir():
                resolve_data_set_folders(e, l)

    return l


def main():
    opt = get_options()

    data_set_folders = []
    print(f'resolving data sets from {opt.input}.')
    for i in opt.input:
        resolve_data_set_folders(i, data_set_folders)
    print(f'resolved {len(data_set_folders)} data set(s).')

    df = []
    for ds_folder in data_set_folders:
        with open(ds_folder.joinpath('def.json'), 'r') as f:
            def_d = json.load(f)
        def_d['folder'] = str(ds_folder).replace('\\', '/')

        if not opt.skip_filters:
            if opt.max_features is not None and def_d['test_features'] > opt.max_features:
                continue

            if opt.train_ratio_range is not None and not (opt.train_ratio_range[0] <= def_d['actual_train_ratio'] <= opt.train_ratio_range[1]):
                continue

            if def_d['test_sw_anomaly_ratio'] == 0:
                continue  # how to evaluate TAD performance without anomalies in test data set?

            if opt.min_samples > 0 and def_d['num_samples'] < opt.min_samples:
                continue

            if len(opt.data_sets) > 0 and def_d['name'] not in opt.data_sets:
                continue

        df.append(def_d)

    df = pd.DataFrame(data=df).sort_values(by='num_samples', ascending=False)
    with pd.ExcelWriter(opt.output.joinpath('summary.xlsx')) as writer:
        df.to_excel(writer, sheet_name='summary', index=False)
        adjust_worksheet_column_width(writer.sheets['summary'], df, index=False)

    folder_list = df['folder'].tolist()
    with open(opt.output.joinpath('folders.json'), 'w') as f:
        json.dump(folder_list, f)


if __name__ == '__main__':
    main()
