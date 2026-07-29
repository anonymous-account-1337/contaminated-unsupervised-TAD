import argparse
import json
import math
import pandas as pd

from pathlib import Path
from collections import defaultdict
from experiment import MergedFinishedExperiment
from utils.plot import f1_train_contamination_plot, rel_f1_train_contamination_plot, retained_performance_plot


MODEL_COLOR_MAP = {
    'Autoencoder': 'yellow',
    'USAD': 'green',
    'Transformer': 'red',
    'TranAD': 'blue',
    'TimesNet': 'black',
    'isolation-forest': 'pink',
    'z-score': 'cyan',
}


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='results')
    parser.add_argument('-o', '--output', default='results/TSB-AD-Anomaly-summary')
    parser.add_argument('--ds-root', required=False, help='the root folder of the data sets', default=None)
    parser.add_argument('--seeds', required=False, type=int, nargs='+', default=None)
    opt = parser.parse_args()

    opt.input = Path(opt.input)
    opt.output = Path(opt.output)

    return opt


def mean_and_std(mean, std, digits=2):
    std = 0 if math.isnan(std) else std
    return f'{mean:.{digits}f} $\\pm$ {std:.{digits}f}'


def make_latex_table(results: dict, digits=2, metric_key='rel_f1', drop_zero_column=True, include_num=True, num_prepend_dummy=1):
    rows = {}

    for key, df in results.items():
        df = df.copy()
        rows[key] = {}

        for ar, row in df.iterrows():
            rows[key][ar] = mean_and_std(row[metric_key], row[f'{metric_key}_std'], digits)

        rows[key][-1] = mean_and_std(df.loc[0]["f1"], df.loc[0]["f1_std"], digits)
        if include_num:
            rows[key][-2] = len(df.iloc[0]['ds_name'])  # num dataset collections
            rows[key][-3] = len(df.iloc[0]['model'])  # num models
            # rows[key][-4] = int(df.loc[0]["group_size"])  # num time series

    table = pd.DataFrame.from_dict(rows, orient='index')
    table = table.reindex(sorted(table.columns), axis=1)
    table.columns = [f'{100 * c:g}' for c in table.columns]
    table = table.sort_index(ascending=True)

    if drop_zero_column:
        table = table.drop(columns=['0'])

    latex = table.reset_index()
    for i in range(num_prepend_dummy):
        latex.insert(0, f'dummy{i}', '')
    latex = latex.to_latex(
        index=False,
        escape=False,
        column_format='l' + 'c' * table.shape[1],
    )

    return latex


def create_final_table(results, key_labels=None, digits=2, dataset_max_f1=None):
    rows = {}

    for key, df in results.items():
        df = df.copy()
        rows[key] = {}

        if key_labels is not None:
            if isinstance(key, str):
                for k, v in zip(key_labels, str(key).split(':')):
                    rows[key][k] = v
            else:
                if len(key_labels) > 1:
                    raise ValueError
                else:
                    rows[key][key_labels[0]] = key

        for ar, row in df.iterrows():
            cell_value = mean_and_std(row['f1'], row['f1_std'], digits)
            if row['rank'] == 0:
                cell_value = r'\rbest{' + cell_value + '}'
            elif row['rank'] == 1:
                cell_value = r'\rsecond{' + cell_value + '}'

            if row['halved_f1']:
                # halved f1 wrt the row
                cell_value = r'\rmarked{' + cell_value + '}'

            if dataset_max_f1 is not None and 'dataset' in rows[key]:
                # halved f1 wrt the best f1 of a dataset across models
                pretty_bad = row['f1'] <= dataset_max_f1[rows[key]['dataset']] / 2
                if pretty_bad:
                    cell_value = r'\rbad' + cell_value

            rows[key][ar] = cell_value

    table = pd.DataFrame.from_dict(rows, orient='index')
    if key_labels is not None:
        table = table.sort_values(by=key_labels)

    latex = table.to_latex(
        index=False,
        escape=False,
        column_format='l' + 'c' * table.shape[1],
    )

    return latex


def save_latex_table(results: dict, out_file, **kwargs):
    if len(results.keys()) == 0:
        return

    with open(out_file, 'w') as f:
        f.write(make_latex_table(results, **kwargs))


def aggregate_experiments(r_list):
    df = pd.concat(r_list)

    agg_spec = {
        'f1': ('f1', 'mean'),
        'f1_std': ('f1_std', 'mean'),
        'rel_f1': ('rel_f1', 'mean'),
        'rel_f1_std': ('rel_f1', 'std'),
        'drop_f1': ('drop_f1', 'mean'),
        'drop_f1_std': ('drop_f1', 'std'),
        'model': ('model', 'unique'),
        'ds_name': ('ds_name', 'unique'),
    }

    if 'rel_cor_f1' in df.columns:
        agg_spec.update({
            'cor_f1': ('cor_f1', 'mean'),
            'cor_f1_std': ('cor_f1_std', 'mean'),
            'rel_cor_f1': ('rel_cor_f1', 'mean'),
            'rel_cor_f1_std': ('rel_cor_f1', 'std'),
            'drop_cor_f1': ('drop_cor_f1', 'mean'),
            'drop_cor_f1_std': ('drop_cor_f1', 'std'),
        })

    grouper = df.groupby(level=0)
    df = grouper.agg(**agg_spec).sort_index(ascending=True)
    df['group_size'] = grouper.size()
    df = df.round(decimals=2)

    if len(df) >= 2:
        best_df = df[['f1', 'f1_std']].copy()
        best_df = best_df.sort_values(by=['f1', 'f1_std'], ascending=[False, True])
        best_df['rank'] = list(range(len(best_df)))
        df['rank'] = best_df['rank']
        df['halved_f1'] = df['f1'] <= best_df.iloc[0]['f1'] / 2

    return df


def prep_dict(d: dict):
    for key in d.keys():
        res = aggregate_experiments(d[key])
        if len(res) > 0:
            d[key] = res
        else:
            del d[key]
    return d


def save_dict(d: dict, output_folder, prefix, save_corruption=True):
    save_latex_table(d, out_file=output_folder.joinpath(f'{prefix}.tex'))
    if save_corruption:
        save_latex_table(d, out_file=output_folder.joinpath(f'cor-{prefix}.tex'), metric_key='rel_cor_f1')
    for key, res_df in d.items():
        f1_train_contamination_plot(res_df, out_file=output_folder.joinpath(f'{prefix}-{key}.png'))
        # rel_f1_train_contamination_plot(res_df, out_file=output_folder.joinpath(f'{prefix}-{key}.png'))


def get_ds_id(ds):
    return '_'.join(ds.split('_')[:2])


class EligibilityException(Exception):
    pass


def check_eligibility(group, uncontaminated_f1):
    if uncontaminated_f1 < 0.6:
        raise EligibilityException('uncontaminated F1-score below 0.6')


def check_eligibility2(group, uncontaminated_f1):
    relevant_datasets = ['CATSv2', 'Exathlon', 'GHL', 'IOPS', 'MITDB', 'NAB', 'OPPORTUNITY', 'SMD', 'SVDB', 'WSD']
    is_relevant_dataset = group['ds_name'].isin(relevant_datasets).all()

    if not is_relevant_dataset:
        raise EligibilityException('irrelevant dataset')

    if uncontaminated_f1 < 0.01:
        raise EligibilityException('uncontaminated F1-score below 0.01')


def check_eligibility3(group, uncontaminated_f1):
    relevant_file_names = ['141_CATSv2_id_4_Sensor_tr_41727_1st_41827.csv', '142_CATSv2_id_5_Sensor_tr_30704_1st_30804.csv', '140_CATSv2_id_3_Sensor_tr_28307_1st_28407.csv', '263_IOPS_id_4_WebService_tr_3482_1st_3582.csv', '272_IOPS_id_13_WebService_tr_3442_1st_3542.csv', '265_IOPS_id_6_WebService_tr_6453_1st_6553.csv', '251_SVDB_id_15_Medical_tr_5421_1st_5521.csv', '111_SVDB_id_28_Medical_tr_1791_1st_1891.csv', '087_SVDB_id_4_Medical_tr_5421_1st_5521.csv', '846_OPPORTUNITY_id_5_HumanActivity_tr_2085_1st_2185.csv', '849_OPPORTUNITY_id_8_HumanActivity_tr_1100_1st_1200.csv', '862_OPPORTUNITY_id_21_HumanActivity_tr_500_1st_566.csv', '074_WSD_id_46_WebService_tr_990_1st_1090.csv', '134_WSD_id_106_WebService_tr_4559_1st_7990.csv', '090_WSD_id_62_WebService_tr_2281_1st_2381.csv', '204_SMD_id_27_Facility_tr_5000_1st_15144.csv', '208_SMD_id_31_Facility_tr_5000_1st_15144.csv', '078_SMD_id_22_Facility_tr_500_1st_326.csv', '200_Exathlon_id_27_Facility_tr_10766_1st_12590.csv', '197_Exathlon_id_24_Facility_tr_10766_1st_12590.csv', '195_Exathlon_id_22_Facility_tr_10766_1st_12590.csv']
    is_relevant_file_name = group['ds_file_name'].isin(relevant_file_names).all()

    if not is_relevant_file_name:
        raise EligibilityException('irrelevant file name')


def prepare_group(group):
    group = group.sort_values(by='ds_anomaly_ratio_train').copy()
    uncontaminated_f1 = group.iloc[0]['m_eval_f1_score']
    group['rel_f1'] = group['m_eval_f1_score'] / uncontaminated_f1

    if 'm_cor_eval_f1_score' in group.columns:
        group['rel_cor_f1'] = group['m_cor_eval_f1_score'] / uncontaminated_f1

    if len(group) != 8:
        raise ValueError('invalid group length')

    check_eligibility3(group, uncontaminated_f1)

    r = {
        'f1': group['m_eval_f1_score'].tolist(),
        'f1_std': group['m_eval_f1_score_std'].tolist(),
        'rel_f1': group['rel_f1'].tolist(),
        'drop_f1': (1 - group['rel_f1']).tolist(),
        'ar': group['ds_anomaly_ratio_train'].tolist(),
        'ds_name': group['ds_name'].tolist(),
        'model': group['exp_model'].tolist() if 'exp_model' in group.columns else group['exp_exp_type'].tolist(),
    }

    if 'm_cor_eval_f1_score' in group.columns:
        r['cor_f1'] = group['m_cor_eval_f1_score'].tolist()
        r['cor_f1_std'] = group['m_cor_eval_f1_score_std'].tolist()
        r['rel_cor_f1'] = group['rel_cor_f1'].tolist()
        r['drop_cor_f1'] = (1 - group['rel_cor_f1']).tolist()

    return pd.DataFrame(r).set_index('ar', drop=True)


def load(experiments_path, ds_root, cache_path=None):
    if cache_path is not None:
        cache_path = Path(cache_path)

        if cache_path.exists():
            return pd.read_pickle(cache_path)

    fe_list = MergedFinishedExperiment.from_folders(experiments_path, ds_root=ds_root)
    for fe in fe_list:
        fe.add_hash('setup_hash', excluded_keys=['id', 'root_path'])
    summary = MergedFinishedExperiment.to_data_frame(fe_list)

    if cache_path is not None:
        summary.to_pickle(cache_path)

    return summary


def baseline(opt):
    s1 = load(opt.input.joinpath('TSB-AD-Anomaly'), ds_root=opt.ds_root, cache_path=opt.input.joinpath('baseline.pkl'))
    s2 = load(opt.input.joinpath('TSB-AD-Anomaly-Classical'), ds_root=opt.ds_root, cache_path=opt.input.joinpath('baseline_classical.pkl'))
    summary = pd.concat([s1, s2], axis=0)

    output = opt.output
    output.mkdir(exist_ok=True)

    output_detail = output.joinpath('detail')
    output_detail.mkdir(exist_ok=True)

    model_dict = defaultdict(list)
    ds_dict = defaultdict(list)
    model_ds_dict = defaultdict(list)
    model_ds_detail_dict = defaultdict(list)
    avg_dict = defaultdict(list)

    stats_list = []

    for (ds, ds_collection, setup_hash), group in summary.groupby(by=['ds_file_name', 'ds_name', 'h_setup_hash']):
        ds_id = get_ds_id(ds)

        if group['exp_exp_type'].iloc[0] in ('z-score', 'isolation-forest'):
            model = group['exp_exp_type'].iloc[0]
        else:
            model = group['exp_model'].iloc[0]

        stats = {
            'model': model,
            'ds_collection': ds_collection,
            'ds': ds,
        }

        try:
            r = prepare_group(group)
            stats['status'] = 'ok'
            stats_list.append(stats)
        except EligibilityException as e:
            stats['status'] = str(e)
            stats_list.append(stats)
            continue

        model_dict[model].append(r)
        ds_dict[ds_collection].append(r)
        model_ds_dict[f'{model}:{ds_collection}'].append(r)
        model_ds_detail_dict[f'{model}-{ds_id}'].append(r)  # don't plot these for brevity
        avg_dict['avg'].append(r)

    stats_list = pd.DataFrame(stats_list)
    stats_list.to_excel(output.joinpath('stats.xlsx'))

    prep_dict(model_dict)
    prep_dict(ds_dict)
    prep_dict(model_ds_dict)
    prep_dict(model_ds_detail_dict)
    prep_dict(avg_dict)

    retained_performance_plot(model_dict, color_map=MODEL_COLOR_MAP, out_file=output.joinpath('overview.png'), fixed_y=False)

    save_dict(model_dict, output, 'model')
    # save_dict(ds_dict, output, 'ds')
    save_dict(model_ds_dict, output_detail, 'model-ds')
    save_dict(avg_dict, output, 'avg')

    ds_max_f1 = get_dataset_max_f1(model_ds_dict)
    with open(output.joinpath('ds_max_f1.json'), 'w') as f:
        json.dump(ds_max_f1, f)

    with open(output.joinpath('_final.tex'), 'w') as f:
        f.write(create_final_table(model_ds_dict, key_labels=['model', 'dataset'], dataset_max_f1=ds_max_f1))


def get_dataset_max_f1(d):
    max_f1 = defaultdict(lambda: 0.)

    for k, v in d.items():
        model, dataset = k.split(':')
        best_f1_row = float(v.loc[v['rank'] == 0, 'f1'].iloc[0])

        if max_f1[dataset] < best_f1_row:
            max_f1[dataset] = best_f1_row

    return dict(max_f1)


def baseline_classical(opt):
    summary = load(opt.input.joinpath('TSB-AD-Anomaly-Classical'), ds_root=opt.ds_root, cache_path=opt.input.joinpath('baseline_classical.pkl'))

    output = opt.output.joinpath('classical')
    output.mkdir(exist_ok=True)

    model_dict = defaultdict(list)
    model_ds_dict = defaultdict(list)
    avg_dict = defaultdict(list)

    for (ds, ds_collection, setup_hash), group in summary.groupby(by=['ds_file_name', 'ds_name', 'h_setup_hash']):
        model = group['exp_exp_type'].iloc[0]

        try:
            r = prepare_group(group)
        except EligibilityException:
            continue

        model_dict[model].append(r)
        model_ds_dict[f'{model}:{ds_collection}'].append(r)
        avg_dict['avg'].append(r)

    prep_dict(model_dict)
    prep_dict(model_ds_dict)
    prep_dict(avg_dict)

    retained_performance_plot(model_dict, color_map=MODEL_COLOR_MAP, out_file=output.joinpath('overview.png'))

    save_dict(model_dict, output, 'model', save_corruption=False)
    save_dict(avg_dict, output, 'avg', save_corruption=False)

    with open(output.joinpath('_final.tex'), 'w') as f:
        f.write(create_final_table(model_ds_dict, key_labels=['model', 'dataset']))


def ablation_latent_size(opt):
    summary = load(opt.input.joinpath('TSB-AD-Anomaly-ablation-latent-size'), ds_root=opt.ds_root, cache_path=opt.input.joinpath('ablation_latent_size.pkl'))

    output = opt.output.joinpath('ablation_latent_size')
    output.mkdir(exist_ok=True)

    latent_size_dict = defaultdict(list)
    autoencoder_dict = defaultdict(list)
    usad_dict = defaultdict(list)

    for (ds, ds_collection, latent_size, setup_hash), group in summary.groupby(by=['ds_file_name', 'ds_name', 'exp_latent_size', 'h_setup_hash']):
        model = group['exp_model'].iloc[0]

        try:
            r = prepare_group(group)
        except EligibilityException:
            continue

        if model == 'Autoencoder':
            autoencoder_dict[latent_size].append(r)
        elif model == 'USAD':
            usad_dict[latent_size].append(r)
        else:
            raise ValueError

        latent_size_dict[latent_size].append(r)

    prep_dict(latent_size_dict)
    prep_dict(autoencoder_dict)
    prep_dict(usad_dict)

    retained_performance_plot(latent_size_dict, title='latent_size', out_file=output.joinpath('latent_size.png'))
    retained_performance_plot(autoencoder_dict, title='latent_size', out_file=output.joinpath('latent_size_Autoencoder.png'))
    retained_performance_plot(usad_dict, title='latent_size', out_file=output.joinpath('latent_size_USAD.png'))

    save_latex_table(latent_size_dict, out_file=output.joinpath('latent_size.tex'))

    with open(output.joinpath('_final.tex'), 'w') as f:
        f.write(create_final_table(latent_size_dict, key_labels=['hyperparameter']))


def ablation_embed_size(opt):
    summary = load(opt.input.joinpath('TSB-AD-Anomaly-ablation-embed-size'), ds_root=opt.ds_root, cache_path=opt.input.joinpath('ablation_embed_size.pkl'))

    output = opt.output.joinpath('ablation_embed_size')
    output.mkdir(exist_ok=True)

    embed_size_dict = defaultdict(list)
    model_tranad_dict = defaultdict(list)
    model_transformer_dict = defaultdict(list)
    model_timesnet_dict = defaultdict(list)

    for (ds, ds_collection, embed_kernel_size, setup_hash), group in summary.groupby(by=['ds_file_name', 'ds_name', 'exp_embed_kernel_size', 'h_setup_hash']):
        model = group['exp_model'].iloc[0]

        try:
            r = prepare_group(group)
        except EligibilityException:
            continue

        if model == 'Transformer':
            model_transformer_dict[embed_kernel_size].append(r)
        elif model == 'TranAD':
            model_tranad_dict[embed_kernel_size].append(r)
        elif model == 'TimesNet':
            model_timesnet_dict[embed_kernel_size].append(r)

        embed_size_dict[embed_kernel_size].append(r)

    prep_dict(embed_size_dict)
    prep_dict(model_transformer_dict)
    prep_dict(model_tranad_dict)
    prep_dict(model_timesnet_dict)

    retained_performance_plot(embed_size_dict, title='embed_size', out_file=output.joinpath('embed_size.png'))
    retained_performance_plot(model_transformer_dict, title='embed_size', out_file=output.joinpath('embed_size_Transformer.png'))
    retained_performance_plot(model_tranad_dict, title='embed_size', out_file=output.joinpath('embed_size_TranAD.png'))
    retained_performance_plot(model_timesnet_dict, title='embed_size', out_file=output.joinpath('embed_size_TimesNet.png'))

    save_latex_table(embed_size_dict, out_file=output.joinpath('embed_size.tex'))

    with open(output.joinpath('_final.tex'), 'w') as f:
        f.write(create_final_table(embed_size_dict, key_labels=['hyperparameter']))


def ablation_d_model(opt):
    summary = load(opt.input.joinpath('TSB-AD-Anomaly-ablation-d-model'), ds_root=opt.ds_root, cache_path=opt.input.joinpath('ablation_d_model.pkl'))

    output = opt.output.joinpath('ablation_d_model')
    output.mkdir(exist_ok=True)

    d_model_dict = defaultdict(list)
    d_model_detail_dict = defaultdict(list)
    model_tranad_dict = defaultdict(list)
    model_transformer_dict = defaultdict(list)
    model_timesnet_dict = defaultdict(list)

    for (ds, ds_collection, d_model, setup_hash), group in summary.groupby(by=['ds_file_name', 'ds_name', 'exp_d_model', 'h_setup_hash']):
        model = group['exp_model'].iloc[0]

        try:
            r = prepare_group(group)
        except EligibilityException:
            continue

        if model == 'Transformer':
            model_transformer_dict[d_model].append(r)
        elif model == 'TranAD':
            model_tranad_dict[d_model].append(r)
        elif model == 'TimesNet':
            model_timesnet_dict[d_model].append(r)

        d_model_dict[d_model].append(r)
        d_model_detail_dict[f'{model}:{d_model}'].append(r)

    prep_dict(d_model_dict)
    prep_dict(d_model_detail_dict)
    prep_dict(model_transformer_dict)
    prep_dict(model_tranad_dict)
    prep_dict(model_timesnet_dict)

    retained_performance_plot(d_model_dict, title='d_model', out_file=output.joinpath('d_model.png'))
    retained_performance_plot(model_transformer_dict, title='d_model', out_file=output.joinpath('d_model_Transformer.png'))
    retained_performance_plot(model_tranad_dict, title='d_model', out_file=output.joinpath('d_model_TranAD.png'))
    retained_performance_plot(model_timesnet_dict, title='d_model', out_file=output.joinpath('d_model_TimesNet.png'))

    save_latex_table(d_model_dict, out_file=output.joinpath('d_model.tex'))

    with open(output.joinpath('_final.tex'), 'w') as f:
        f.write(create_final_table(d_model_dict, key_labels=['hyperparameter']))

    with open(output.joinpath('_final_detail.tex'), 'w') as f:
        f.write(create_final_table(d_model_detail_dict, key_labels=['model', 'hyperparameter']))


def ablation_e_layers(opt):
    summary = load(opt.input.joinpath('TSB-AD-Anomaly-ablation-e-layers'), ds_root=opt.ds_root, cache_path=opt.input.joinpath('ablation_e_layers.pkl'))

    output = opt.output.joinpath('ablation_e_layers')
    output.mkdir(exist_ok=True)

    e_layers_dict = defaultdict(list)
    model_transformer_dict = defaultdict(list)
    model_timesnet_dict = defaultdict(list)

    for (ds, ds_collection, e_layers, setup_hash), group in summary.groupby(by=['ds_file_name', 'ds_name', 'exp_e_layers', 'h_setup_hash']):
        model = group['exp_model'].iloc[0]

        try:
            r = prepare_group(group)
        except EligibilityException:
            continue

        if model == 'Transformer':
            model_transformer_dict[e_layers].append(r)
        elif model == 'TimesNet':
            model_timesnet_dict[e_layers].append(r)

        e_layers_dict[e_layers].append(r)

    prep_dict(e_layers_dict)
    prep_dict(model_transformer_dict)
    prep_dict(model_timesnet_dict)

    retained_performance_plot(e_layers_dict, title='e_layers', out_file=output.joinpath('e_layers.png'))
    retained_performance_plot(model_transformer_dict, title='e_layers', out_file=output.joinpath('e_layers_Transformer.png'))
    retained_performance_plot(model_timesnet_dict, title='e_layers', out_file=output.joinpath('e_layers_TimesNet.png'))

    save_latex_table(e_layers_dict, out_file=output.joinpath('e_layers.tex'))

    with open(output.joinpath('_final.tex'), 'w') as f:
        f.write(create_final_table(e_layers_dict, key_labels=['hyperparameter']))


def main():
    opt = get_options()
    opt.output.mkdir(exist_ok=True, parents=True)

    baseline(opt)
    # baseline_classical(opt)
    ablation_embed_size(opt)
    ablation_latent_size(opt)
    ablation_d_model(opt)
    ablation_e_layers(opt)


if __name__ == '__main__':
    main()
