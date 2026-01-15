import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from types import SimpleNamespace
from pathlib import Path
from data_provider.anomaly_data_loader import AnomalyDataset
from torch.utils.data import DataLoader
from models.USAD import Model as USAD
from experiment import FinishedExperiment
from utils.divergence import kl_symmetric_gaussian
from utils.latex import to_latex_table, save_tex


def estimate_multivariate_gaussian(x):
    mean = np.mean(x, axis=0)
    cov = np.cov(x, rowvar=False)
    return mean, cov


def create_model(experiment, device):
    cfg = SimpleNamespace(**{
        'enc_in': experiment.experiment['enc_in'],
        'seq_len': experiment.experiment['seq_len'],
        'latent_size': experiment.experiment['latent_size'],
    })
    model = USAD(cfg).to(device=device)
    model.load_state_dict(torch.load(experiment.path.joinpath('checkpoints/best.pt'), weights_only=True, map_location=device))
    model.eval()
    return model


def inference(model, ds):
    x_list, y_list, z_list = [], [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=32, shuffle=False):
            z = model(x, return_latent_space=True)

            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
            y = y.reshape(y.shape[0], y.shape[1] * y.shape[2])

            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            z = z.detach().cpu().numpy()

            y = np.array([np.unique(y[i])[-1] for i in range(y.shape[0])], dtype=np.int32)

            x_list.extend(x)
            y_list.extend(y)
            z_list.extend(z)

    x_list = np.array(x_list)
    y_list = np.array(y_list)
    z_list = np.array(z_list)

    return x_list, y_list, z_list


def plot(l, out_file=None):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    for e in l:
        ax.plot(e['x'], e['y'], marker='o', color=e['color'], label=e['label'] if 'label' in e else None)
        if 'std' in e:
            ax.fill_between(e['x'], e['y'] - e['std'], e['y'] + e['std'], color=e['color'], alpha=.1)

    ax.set_xlabel('contamination ratio')
    ax.set_ylabel('KL divergence')

    ax.legend()
    plt.tight_layout()
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)
    plt.close(fig)


def load_experiments(experiments_path, ds_root_path, seed, latent_size):
    experiments = FinishedExperiment.from_folders(experiments_path, ds_root=ds_root_path)
    experiments = list(filter(lambda e: e.experiment['seed'] == seed and e.experiment['latent_size'] == latent_size, experiments))
    experiments.sort(key=lambda e: e.ds_def['anomaly_ratio_train'])

    return experiments


class LatentSpaceExperiment:

    @classmethod
    def create(cls, exp, ds, device):
        model = create_model(exp, device)
        x, y, z = inference(model, ds)
        return cls(exp, x, y, z)

    def __init__(self, exp, x, y, z):
        self.exp = exp
        self.x = x
        self.y = y
        self.z = z

        self.z_gaussian = {}
        for i in self.labels():
            mean, cov = estimate_multivariate_gaussian(self.z[self.y == i])
            self.z_gaussian[i] = (mean, cov)

    def z_distribution_params(self, label):
        return self.z_gaussian[label]

    def labels(self, with_zero=True):
        labels = np.unique(self.y)
        if not with_zero:
            labels = labels[labels != 0]
        return labels

    def train_anomaly_ratio(self):
        return self.exp.ds_def['anomaly_ratio_train'] * 100


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--ds-root', required=False, help='the root folder of the data sets', default='.')
    parser.add_argument('-i', '--input-folder', required=False, default='results/ablation_latent_size')
    parser.add_argument('-o', '--output-folder', required=False, default='results/plots/latent_size_analysis_usad')
    opt = parser.parse_args()

    opt.ds_root = Path(opt.ds_root)
    opt.input_folder = Path(opt.input_folder)
    opt.output_folder = Path(opt.output_folder)

    return opt


def format_float(s):
    return f'{s:.2f}'


def to_tex(plot_list, out_file):
    dfs = []
    for e in plot_list.copy():
        label = e['label']
        del e['label']
        del e['color']

        df = pd.DataFrame(e)
        df[label] = df['y'].map(format_float) + '±' + df['std'].map(format_float)
        df = df.drop(columns=['std', 'y'])
        df = df.rename(columns={'x': 'ar'})
        df = df.set_index(keys='ar', drop=True)
        dfs.append(df)

    dfs = pd.concat(dfs, axis=1)
    save_tex(dfs, out_file)


def main():
    opt = get_options()
    ds = AnomalyDataset(root_path=opt.ds_root.joinpath(r'dataset/voltage-main/voltage-l-ar0.0'), flag='test', device=opt.device)
    opt.output_folder.mkdir(exist_ok=True, parents=True)

    color_map = {
        1: 'blue',
        2: 'red',
        3: 'green',
    }

    label_map = {
        1: 'point',
        2: 'drop',
        3: 'noise',
    }

    for latent_size in [8, 16, 32, 64]:
        plot_list = []
        for label in range(1, 4):
            plot_dict = {}

            for seed in [47, 48, 49]:
                exp_list = [LatentSpaceExperiment.create(e, ds, opt.device) for e in load_experiments(opt.input_folder, opt.ds_root, seed=seed, latent_size=latent_size)]
                for e in exp_list:
                    mean, cov = e.z_distribution_params(label=0)
                    label_mean, label_cov = e.z_distribution_params(label=label)
                    # dist = np.linalg.norm(mean - label_mean, ord=2)
                    # dist = kl_gaussian(mean, cov, label_mean, label_cov)
                    dist = kl_symmetric_gaussian(mean, cov, label_mean, label_cov)

                    ar = e.train_anomaly_ratio()
                    if ar in plot_dict:
                        plot_dict[ar].append(dist)
                    else:
                        plot_dict[ar] = [dist]

            x = []
            y = []
            std = []

            for ar in sorted(plot_dict.keys()):
                x.append(ar)
                y.append(np.mean(plot_dict[ar]))
                std.append(np.std(plot_dict[ar]))

            x = np.array(x)
            y = np.array(y)
            std = np.array(std)

            plot_list.append({
                'x': x,
                'y': y,
                'std': std,
                'color': color_map[label],
                'label': f'{label} - {label_map[label]}',
            })

        plot(plot_list, out_file=opt.output_folder.joinpath(f'kl_divergence_{latent_size}.png'))
        to_tex(plot_list, out_file=opt.output_folder.joinpath(f'kl_divergence_{latent_size}.tex'))


if __name__ == '__main__':
    main()
