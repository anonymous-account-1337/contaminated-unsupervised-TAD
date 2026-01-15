import argparse
import pandas as pd
import matplotlib as mpl

from pathlib import Path
from experiment import FinishedExperiment
from utils.plot import contamination_plot
from utils.latex import save_tex


MODEL_COLOR = {
    'z-score': mpl.colormaps['Greys'],
    'isolation-forest': mpl.colormaps['Oranges'],
    'TranAD': mpl.colormaps['Blues'],
    'Transformer': mpl.colormaps['Reds'],
    'TimesNet': mpl.colormaps['Greens'],
    'USAD': mpl.colormaps['Purples'],
}

MODEL_ORDER = {
    'z-score': 6,
    'isolation-forest': 5,
    'TranAD': 4,
    'Transformer': 3,
    'TimesNet': 2,
    'USAD': 1,
}


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=False, default=Path('results'))
    parser.add_argument('-o', '--output', required=False, default=Path('results/plots/ablation_analysis'))
    parser.add_argument('--ds-root', required=False, help='the root folder of the data sets', default=Path('.'))
    parser.add_argument('-v', '--verbose', required=False, action='store_true', default=False)
    opt = parser.parse_args()

    opt.input = Path(opt.input)
    opt.output = Path(opt.output)

    return opt


def format_float(s):
    return f'{s:.2f}'


def result_table(plot_list, group_by=None, mark=True, drop_percentage=0.5):
    ts_dict = {}

    for e in plot_list:
        ts = e['ts'].copy()
        ts.loc[ts['f1'].sort_values(ascending=False).index, 'rank'] = range(1, len(ts) + 1)
        ts['rank'] = ts['rank'].astype(int)

        best_f1 = ts.loc[ts['rank'] == 1, 'f1'][0]
        ts['drop'] = False
        ts.loc[ts['f1'] <= best_f1 * drop_percentage, 'drop'] = True

        if ts['drop'].any():
            first_drop_idx = ts['drop'].idxmax()
            ts['drop'] = False
            ts.loc[first_drop_idx, 'drop'] = True

        ts['f1'] = ts['f1'].map(format_float)

        if 'f1std' in ts.columns:
            ts['f1'] = ts['f1'] + '±' + ts['f1std'].map(format_float)

        if mark:
            if len(ts) >= 1:
                idx = ts.loc[ts['rank'] == 1].index
                ts.loc[idx, 'f1'] = r'\textbf{' + ts.loc[idx, 'f1'] + '}'

            if len(ts) >= 2:
                idx = ts.loc[ts['rank'] == 2].index
                ts.loc[idx, 'f1'] = r'\underline{' + ts.loc[idx, 'f1'] + '}'

            if ts['drop'].any():
                ts.loc[ts['drop'], 'f1'] = r'\textcolor{red}{' + ts.loc[ts['drop'], 'f1'] + '}'

        ts = ts.set_index('ar')['f1'].rename(e['label'])

        group_key = None if group_by is None else e[group_by]
        if group_key not in ts_dict:
            ts_dict[group_key] = []

        ts_dict[group_key].append(ts)

    tables = [pd.concat(ts_dict[group_key], axis=1) for group_key in sorted(ts_dict.keys())]
    return tables


def keys_to_str(keys):
    if isinstance(keys, tuple):
        key_str = '-'.join(map(str, keys))
    else:
        key_str = keys

    return key_str


def prepare_contamination_data_frame(e, point_adjust=False):
    if point_adjust:
        if 'metrics_pa-max-score_f1_score_mean' in e.columns:
            ts = pd.DataFrame(data=e[['metrics_pa-max-score_f1_score_mean', 'metrics_pa-max-score_f1_score_std', 'ds_def_anomaly_ratio_train']])
            ts = ts.rename(columns={
                'metrics_pa-max-score_f1_score_mean': 'f1',
                'metrics_pa-max-score_f1_score_std': 'f1std',
                'ds_def_anomaly_ratio_train': 'ar',
            })
        else:
            ts = pd.DataFrame(data=e[['metrics_pa-max-score_f1_score', 'ds_def_anomaly_ratio_train']])
            ts = ts.rename(columns={
                'metrics_pa-max-score_f1_score': 'f1',
                'ds_def_anomaly_ratio_train': 'ar',
            })
    else:
        if 'metrics_max-score_f1_score_mean' in e.columns:
            ts = pd.DataFrame(data=e[['metrics_max-score_f1_score_mean', 'metrics_max-score_f1_score_std', 'ds_def_anomaly_ratio_train']])
            ts = ts.rename(columns={
                'metrics_max-score_f1_score_mean': 'f1',
                'metrics_max-score_f1_score_std': 'f1std',
                'ds_def_anomaly_ratio_train': 'ar',
            })
        else:
            ts = pd.DataFrame(data=e[['metrics_max-score_f1_score', 'ds_def_anomaly_ratio_train']])
            ts = ts.rename(columns={
                'metrics_max-score_f1_score': 'f1',
                'ds_def_anomaly_ratio_train': 'ar',
            })

    ts = ts.sort_values(by='ar')
    ts['ar'] = ts['ar'] * 100
    return ts


def ablation_latent_size(opt):
    experiments = FinishedExperiment.from_folders(opt.input.joinpath('ablation_latent_size'), ds_root=opt.ds_root)
    df = FinishedExperiment.create_summary(experiments, merge_iterations=True)

    line_styles = {
        8: ':',
        16: '-.',
        32: '--',
        64: '-',
    }

    plot_list = []
    for (model, latent_size), e in df.groupby(by=['experiment_model', 'experiment_latent_size']):
        plot_list.append({
            'label': f'{model}-{latent_size}',
            'ts': prepare_contamination_data_frame(e),
            'color': MODEL_COLOR[model](0.8),
            'line_style': line_styles[latent_size],
            'order': latent_size,
        })

    plot_list.sort(key=lambda e: e['order'])
    save_tex(result_table(plot_list), out_file=opt.output.joinpath('ablation_latent_size.tex'))
    contamination_plot(plot_list, out_file=opt.output.joinpath('ablation_latent_size.png'))


def ablation_d_model(opt):
    experiments = FinishedExperiment.from_folders(opt.input.joinpath('ablation_d_model'), ds_root=opt.ds_root)
    df = FinishedExperiment.create_summary(experiments, merge_iterations=True)

    line_styles = {
        16: ':',
        32: '-.',
        64: '--',
        128: '-',
    }

    plot_list = []
    for (model, d_model), e in df.groupby(by=['experiment_model', 'experiment_d_model']):
        plot_list.append({
            'label': f'{model}-{d_model}',
            'ts': prepare_contamination_data_frame(e),
            'color': MODEL_COLOR[model](0.8),
            'line_style': line_styles[d_model],
            'model': model,
            'order': d_model,
        })

    plot_list.sort(key=lambda e: (e['model'], e['order']))
    save_tex(result_table(plot_list, group_by='model'), out_file=opt.output.joinpath('ablation_d_model.tex'))
    contamination_plot(plot_list, out_file=opt.output.joinpath('ablation_d_model.png'))


def ablation_embedding(opt):
    experiments = FinishedExperiment.from_folders(opt.input.joinpath('ablation_embedding'), ds_root=opt.ds_root)
    df = FinishedExperiment.create_summary(experiments, merge_iterations=True)
    df.loc[df['experiment_embed_type'] == 'linear', 'experiment_embed_kernel_size'] = 1
    df['experiment_embed_kernel_size'] = df['experiment_embed_kernel_size'].astype(int)

    line_styles = {
        3: ':',
        5: '-.',
        7: '--',
        1: '-'
    }

    plot_list = []
    for (model, embed_type, kernel_size), e in df.groupby(by=['experiment_model', 'experiment_embed_type', 'experiment_embed_kernel_size']):
        plot_list.append({
            'label': f'{model}-{embed_type}' if embed_type == 'linear' else f'{model}-{embed_type}-{kernel_size}',
            'ts': prepare_contamination_data_frame(e),
            'color': MODEL_COLOR[model](0.8),
            'line_style': line_styles[kernel_size],
            'model': model,
            'order': kernel_size,
        })

    plot_list.sort(key=lambda e: (e['model'], e['order']))
    save_tex(result_table(plot_list, group_by='model'), out_file=opt.output.joinpath('ablation_embedding.tex'))
    contamination_plot(plot_list, out_file=opt.output.joinpath('ablation_embedding.png'))


def ablation_dropout(opt):
    experiments = FinishedExperiment.from_folders(opt.input.joinpath('ablation_dropout'), ds_root=opt.ds_root)
    df = FinishedExperiment.create_summary(experiments, merge_iterations=True)

    line_styles = {
        0.5: ':',
        0.2: '-.',
        0.1: '--',
        0.0: '-',
    }

    plot_list = []
    for (model, dropout), e in df.groupby(by=['experiment_model', 'experiment_dropout']):
        plot_list.append({
            'label': f'{model}-{dropout:.1f}',
            'ts': prepare_contamination_data_frame(e),
            'color': MODEL_COLOR[model](0.8),
            'line_style': line_styles[dropout],
            'model': model,
            'order': int(dropout * 100),
        })

    plot_list.sort(key=lambda e: (e['model'], e['order']))
    save_tex(result_table(plot_list, group_by='model'), out_file=opt.output.joinpath('ablation_dropout.tex'))
    contamination_plot(plot_list, out_file=opt.output.joinpath('ablation_dropout.png'))


def ablation_d_ff(opt):
    experiments = FinishedExperiment.from_folders(opt.input.joinpath('ablation_d_ff'), ds_root=opt.ds_root)
    df = FinishedExperiment.create_summary(experiments, merge_iterations=True)

    line_styles = {
        64: ':',
        128: '-.',
        256: '--',
        512: '-',
    }

    plot_list = []
    for (model, d_ff), e in df.groupby(by=['experiment_model', 'experiment_d_ff']):
        plot_list.append({
            'label': f'{model}-{d_ff}',
            'ts': prepare_contamination_data_frame(e),
            'color': MODEL_COLOR[model](0.8),
            'line_style': line_styles[d_ff],
            'model': model,
            'order': d_ff,
        })

    plot_list.sort(key=lambda e: (e['model'], e['order']))
    save_tex(result_table(plot_list, group_by='model'), out_file=opt.output.joinpath('ablation_d_ff.tex'))
    contamination_plot(plot_list, out_file=opt.output.joinpath('ablation_d_ff.png'))


def simple_baseline(opt):
    experiments = FinishedExperiment.from_folders(opt.input.joinpath('isolation_forest'), ds_root=opt.ds_root) + \
                  FinishedExperiment.from_folders(opt.input.joinpath('z_score'), ds_root=opt.ds_root)
    df = FinishedExperiment.create_summary(experiments, merge_iterations=True)

    plot_list = []
    for model, e in df.groupby(by='experiment_exp_type'):
        plot_list.append({
            'label': f'{model}',
            'ts': prepare_contamination_data_frame(e),
            'color': MODEL_COLOR[model](0.8),
            'model': model,
            'order': MODEL_ORDER[model],
        })

    plot_list.sort(key=lambda e: (e['model'], e['order']))
    save_tex(result_table(plot_list), out_file=opt.output.joinpath('simple_baseline.tex'))
    contamination_plot(plot_list, out_file=opt.output.joinpath('simple_baseline.png'))


def baseline(opt):
    experiments = FinishedExperiment.from_folders(opt.input.joinpath('baseline'), ds_root=opt.ds_root)
    df = FinishedExperiment.create_summary(experiments, merge_iterations=False)

    plot_list = []
    for model, e in df.groupby(by='experiment_model'):
        plot_list.append({
            'label': f'{model}',
            'ts': prepare_contamination_data_frame(e),
            'color': MODEL_COLOR[model](0.8),
            'model': model,
            'order': MODEL_ORDER[model],
        })

    plot_list.sort(key=lambda e: (e['model'], e['order']))
    save_tex(result_table(plot_list), out_file=opt.output.joinpath('baseline.tex'))
    contamination_plot(plot_list, out_file=opt.output.joinpath('baseline.png'))


def baseline_extended(opt):
    experiments = FinishedExperiment.from_folders(opt.input.joinpath('baseline_extended'), ds_root=opt.ds_root)
    df = FinishedExperiment.create_summary(experiments, merge_iterations=True)

    plot_list = []
    for model, e in df.groupby(by='experiment_model'):
        plot_list.append({
            'label': f'{model}',
            'ts': prepare_contamination_data_frame(e),
            'color': MODEL_COLOR[model](0.8),
            'model': model,
            'order': MODEL_ORDER[model],
        })

    plot_list.sort(key=lambda e: (e['model'], e['order']))
    save_tex(result_table(plot_list), out_file=opt.output.joinpath('baseline_extended.tex'))
    contamination_plot(plot_list, out_file=opt.output.joinpath('baseline_extended.png'))


def baseline_sam(opt):
    experiments = FinishedExperiment.from_folders(opt.input.joinpath('baseline_sam'), ds_root=opt.ds_root)
    df = FinishedExperiment.create_summary(experiments, merge_iterations=True)

    plot_list = []
    for model, e in df.groupby(by='experiment_model'):
        plot_list.append({
            'label': f'{model}',
            'ts': prepare_contamination_data_frame(e),
            'color': MODEL_COLOR[model](0.8),
            'model': model,
            'order': MODEL_ORDER[model],
        })

    plot_list.sort(key=lambda e: (e['model'], e['order']))
    save_tex(result_table(plot_list), out_file=opt.output.joinpath('baseline_sam.tex'))
    contamination_plot(plot_list, out_file=opt.output.joinpath('baseline_sam.png'))


def tasks(opt):
    for task in ('task_point', 'task_noise', 'task_quadratic'):
        experiments = FinishedExperiment.from_folders(opt.input.joinpath(task), ds_root=opt.ds_root)
        df = FinishedExperiment.create_summary(experiments, merge_iterations=True)

        plot_list = []
        for model, e in df.groupby(by='experiment_model'):
            plot_list.append({
                'label': f'{model}',
                'ts': prepare_contamination_data_frame(e),
                'color': MODEL_COLOR[model](0.8),
                'model': model,
                'order': MODEL_ORDER[model],
            })

        plot_list.sort(key=lambda e: (e['model'], e['order']))
        save_tex(result_table(plot_list), out_file=opt.output.joinpath(f'{task}.tex'))
        contamination_plot(plot_list, out_file=opt.output.joinpath(f'{task}.png'))


def main():
    opt = get_options()

    output_folder = opt.output
    output_folder.mkdir(parents=True, exist_ok=True)

    simple_baseline(opt)
    baseline(opt)
    baseline_extended(opt)
    baseline_sam(opt)
    tasks(opt)
    ablation_latent_size(opt)
    ablation_d_model(opt)
    ablation_embedding(opt)
    ablation_dropout(opt)
    ablation_d_ff(opt)


if __name__ == '__main__':
    main()
