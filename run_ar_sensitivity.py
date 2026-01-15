import argparse
import numpy as np
import multiprocessing

from pathlib import Path
from exp.exp_anomaly_detection import get_metrics, point_wise_anomaly_ratio
from experiment import FinishedExperiment
from utils.plot import ar_sensitivity_plot


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='a folder containing multiple finished experiments')
    parser.add_argument('-o', '--output', required=False, help='output folder', default='results/ar-sensitivity')
    parser.add_argument('--ds-root', required=False, help='the root folder of the data sets', default=None)
    opt = parser.parse_args()

    opt.input = Path(opt.input)
    opt.output = Path(opt.output)

    if opt.ds_root is not None:
        opt.ds_root = Path(opt.ds_root)

    return opt


def f1_score_for_estimated_ar(estimated_anomaly_ratio, e: FinishedExperiment):
    # calculate anomaly score using an estimated anomaly ratio
    threshold = float(np.percentile(e.val_energy, 100 - estimated_anomaly_ratio))

    # calculate f1-score for test data set
    energy = e.test_energy.reshape(-1)
    labels = e.test_labels.reshape(-1)

    pred = (energy > threshold).astype(int)
    return get_metrics(labels, pred, point_adjust=False)['f1_score']


def f1_scores_for_experiment(e: FinishedExperiment, anomaly_ratios):
    key = e.ds_def['basis_function']
    f1_scores = np.array([f1_score_for_estimated_ar(ar, e) for ar in anomaly_ratios])
    f1_scores = (f1_scores / np.max(f1_scores)) * 100
    return key, f1_scores.tolist()


def get_actual_data_set_anomaly_ratios(experiments: [FinishedExperiment], ds_root: Path):
    actual_ars = {}
    for e in experiments:
        base_function = e.ds_def['basis_function']
        name = e.ds_def['name']

        if base_function not in actual_ars:
            actual_ars[base_function] = {}

        if name not in actual_ars[base_function]:
            actual_ars[base_function][name] = []
            test_y = np.load(ds_root.joinpath(e.experiment['root_path']).joinpath('test_y.npy'))
            point_wise_ar = point_wise_anomaly_ratio(test_y.reshape(-1)) * 100

            actual_ars[base_function][name] = {
                'point_wise_ar': point_wise_ar,
                'sample_wise_ar': e.ds_def['anomaly_ratio_test'] * 100,
            }

    d = {}
    for key in actual_ars:
        point_wise_ar = [actual_ars[key][e]['point_wise_ar'] for e in actual_ars[key]]
        sample_wise_ar = [actual_ars[key][e]['sample_wise_ar'] for e in actual_ars[key]]

        point_wise_ar = sum(point_wise_ar) / len(point_wise_ar)
        sample_wise_ar = sum(sample_wise_ar) / len(sample_wise_ar)

        d[key] = {
            'point_wise_ar': point_wise_ar,
            'sample_wise_ar': sample_wise_ar,
        }

    return d


def main():
    opt = get_options()
    output_folder = opt.output
    output_folder.mkdir(parents=True, exist_ok=True)

    anomaly_ratios = np.arange(0, 6, 0.2)
    experiments = FinishedExperiment.from_folders(opt.input, ds_root=opt.ds_root)
    actual_ars = get_actual_data_set_anomaly_ratios(experiments, opt.ds_root)

    res = {}
    with multiprocessing.Pool() as p:
        futures = []
        for e in experiments:
            if e.ds_def['num_samples'] >= 100000 and e.ds_def['anomaly_ratio_train'] <= 0.05:
                futures.append(p.apply_async(f1_scores_for_experiment, (e, anomaly_ratios)))

        for future in futures:
            key, f1_scores = future.get()
            if key not in res:
                res[key] = []

            res[key].append(f1_scores)

    for key in res:
        ar_sensitivity_plot(anomaly_ratios, np.array(res[key]), point_wise_ar=actual_ars[key]['point_wise_ar'], sample_wise_ar=actual_ars[key]['sample_wise_ar'], out_file=output_folder.joinpath(f'ar_sensitivity_{key}.png'))


if __name__ == '__main__':
    main()
