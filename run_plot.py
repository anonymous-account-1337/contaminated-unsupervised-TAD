import argparse

from pathlib import Path
from data_provider.anomaly_data_loader import AnomalyDataset
from utils.plot import ts_plot


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-path', type=str, required=True)
    parser.add_argument('--flag', type=str, default='train')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('-o', '--output-folder', type=str, default='results/plots')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=-1)
    args = parser.parse_args()
    args.output_folder = Path(args.output_folder)
    return args


def main():
    """plots the features the way a model will see those features"""
    opt = get_options()
    opt.output_folder.mkdir(exist_ok=True, parents=True)

    ds = AnomalyDataset(opt.root_path, opt.flag)
    _, _, num_features = ds.get_dummy_sample().numpy().shape

    for idx in range(opt.start, len(ds) if opt.stop == -1 else opt.stop, 1):
        x, y = ds[idx]

        for feature in range(num_features):
            f_x = x[:, feature]
            f_y = y[:, feature]
            ts_plot(f_x, y_true=f_y, out_file=opt.output_folder.joinpath(f'{idx:04d}_f{feature}.png'))


if __name__ == '__main__':
    main()
