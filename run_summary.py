import argparse

from pathlib import Path
from experiment import FinishedExperiment


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='a folder containing multiple finished experiments')
    parser.add_argument('-o', '--output', required=True, help='output summary file')
    parser.add_argument('--ds-root', required=False, help='the root folder of the data sets', default=None)
    parser.add_argument('--merge-iterations', action='store_true', default=False)
    parser.add_argument('--exclude-bulk-cols', action='store_true', default=False)
    opt = parser.parse_args()

    opt.input = Path(opt.input)
    opt.output = Path(opt.output)

    return opt


def main():
    opt = get_options()
    finished_experiments = FinishedExperiment.from_folders(opt.input, ds_root=opt.ds_root)
    summary = FinishedExperiment.create_summary(finished_experiments, merge_iterations=opt.merge_iterations)

    if opt.exclude_bulk_cols:
        del_columns = list(filter(lambda col: 'point_adjust' in col, summary.columns.tolist())) \
        + list(filter(lambda col: '_loss' in col, summary.columns.tolist()))
        summary.drop(del_columns, axis=1, inplace=True)

    if opt.output.name.endswith('.csv'):
        summary.to_csv(opt.output)
    elif opt.output.name.endswith('.xlsx'):
        summary.to_excel(opt.output)
    else:
        raise ValueError(f'invalid output file format: {opt.output}')


if __name__ == '__main__':
    main()
