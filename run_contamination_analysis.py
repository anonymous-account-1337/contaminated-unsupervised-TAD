import argparse
import numpy as np
import pandas as pd
import re

from pathlib import Path
from experiment import FinishedExperiment
from utils.plot import train_contamination_agg_plot


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+', required=True, help='a folder containing multiple finished experiments')
    parser.add_argument('-o', '--output', required=True, help='output folder')
    parser.add_argument('--ds-root', required=False, help='the root folder of the data sets', default=Path('.'))
    parser.add_argument('--max-ar-table', required=False, type=float, default=100)
    parser.add_argument('--max-ar-plot', required=False, type=float, default=10)
    parser.add_argument('-v', '--verbose', required=False, action='store_true', default=False)
    opt = parser.parse_args()

    opt.input = list(map(Path, opt.input))
    opt.output = Path(opt.output)

    return opt


def ds_def_name_to_base_name(s):
    return s.split('/')[1]


def sort_columns(cols):
    """Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(cols, key=alphanum_key)


def data_frame_to_latex_table(df):
    cols = df.columns.tolist()
    cols.insert(0, df.index.name)
    header_str = ' & '.join(cols) + r' \\'

    lines = [header_str, r'\hline']

    for idx, r in df.iterrows():
        r = r.tolist()
        r.insert(0, str(idx))
        row_str = ' & '.join(r) + r' \\'
        lines.append(row_str)

    return '\n'.join(lines)


def format_float(s):
    return f'{s:.2f}'


def keys_to_str(keys):
    if isinstance(keys, tuple):
        key_str = '-'.join(map(str, keys))
    else:
        key_str = keys

    return key_str


def create_result_tables(df, output_folder, max_ar, mark_best=True):
    for keys, group in df.groupby(by=['experiment_model']):
        key_str = keys_to_str(keys)

        res = pd.DataFrame(index=sorted(group['ds_def_anomaly_ratio_train'].unique()), data=None)
        res.index.name = 'sample_wise_anomaly_ratio_train'

        for group_keys, group in group.groupby(by=['base_dataset', 'ds_def_num_samples']):
            group_key_str = keys_to_str(group_keys)
            metrics = group['metrics_max-score_f1_score_mean'].map(format_float) + '\u00B1' + group['metrics_max-score_f1_score_std'].map(format_float)
            metrics.name = group_key_str
            metrics = metrics.to_frame()
            metrics['sample_wise_anomaly_ratio_train'] = group['ds_def_anomaly_ratio_train']
            metrics = metrics.loc[metrics['sample_wise_anomaly_ratio_train'] <= max_ar]

            if mark_best:
                metrics['f1-score'] = group['metrics_max-score_f1_score_mean']
                metrics = metrics.sort_values(by='f1-score', ascending=False)
                metrics['rank'] = range(1, len(metrics) + 1)

                best_result_predicate = metrics['rank'] == 1
                metrics.loc[best_result_predicate, group_key_str] = r'\textbf{' + metrics.loc[best_result_predicate, group_key_str] + '}'

                second_best_result_predicate = metrics['rank'] == 2
                metrics.loc[second_best_result_predicate, group_key_str] = r'\underline{' + metrics.loc[second_best_result_predicate, group_key_str] + '}'

                metrics = metrics.drop(['f1-score', 'rank'], axis=1)

            metrics = metrics.sort_values(by='sample_wise_anomaly_ratio_train')
            metrics = metrics.set_index('sample_wise_anomaly_ratio_train')

            res = res.join(metrics)

        res = res.reindex(sort_columns(res.columns), axis=1)
        res = res.set_index(res.index.map(lambda a: str(int(a * 100))))  # convert anomaly ratio from [0-1] to [0-100]

        with open(output_folder.joinpath(f'{key_str}_results.tex'), 'w', encoding='utf-8') as f:
            f.write(data_frame_to_latex_table(res))


def create_plots(df, output_folder, max_ar):
    for keys, group in df.groupby(by=['base_dataset', 'experiment_model']):
        key_str = keys_to_str(keys)

        res = {}
        for (ds_size), group in group.groupby(by=['ds_def_num_samples']):
            metrics = pd.DataFrame(index=group.index, data=None)
            metrics['sample_wise_anomaly_ratio_train'] = group['ds_def_anomaly_ratio_train'] * 100
            metrics['f1_score'] = group['metrics_max-score_f1_score_mean']
            metrics['f1_score_std'] = group['metrics_max-score_f1_score_std']
            metrics = metrics.loc[metrics['sample_wise_anomaly_ratio_train'] <= max_ar]

            metrics = metrics.sort_values(by='sample_wise_anomaly_ratio_train')
            metrics = metrics.set_index('sample_wise_anomaly_ratio_train')

            res[ds_size] = metrics

        train_contamination_agg_plot(res, title=key_str, out_file=output_folder.joinpath(f'{key_str}.png'))


def main():
    opt = get_options()

    output_folder = opt.output
    output_folder.mkdir(parents=True, exist_ok=True)
    experiments = FinishedExperiment.from_folders(opt.input, ds_root=opt.ds_root)
    df = FinishedExperiment.create_summary(experiments, merge_iterations=True)
    # df = pd.read_pickle(r'\\fit4ba3\time-series\output\temp.pkl')
    df['base_dataset'] = df['experiment_root_path'].map(ds_def_name_to_base_name)

    create_result_tables(df, output_folder, opt.max_ar_table)
    create_plots(df, output_folder, opt.max_ar_plot)


if __name__ == '__main__':
    main()
