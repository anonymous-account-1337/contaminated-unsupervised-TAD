import argparse
import shutil
from pathlib import Path

from experiment import FinishedExperiment


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_folder', help='a folder')
    parser.add_argument('-s', '--selection', nargs='*', default=[])
    parser.add_argument('-a', '--action', type=str, choices=['ls', 'mv', 'cp'], default='ls')
    parser.add_argument('-o', '--output-folder', type=str, default=None)
    parser.add_argument('--ds-root', type=str, default=None)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    opt = parser.parse_args()
    if opt.output_folder is not None:
        opt.output_folder = Path(opt.output_folder)
        opt.output_folder.mkdir(parents=True, exist_ok=True)

    if opt.ds_root is not None:
        opt.ds_root = Path(opt.ds_root)

    return opt


def resolve(obj, item, sep='/'):
    if obj is None:
        raise ValueError('obj is None')

    if isinstance(item, str):
        item = item.split(sep)

    if len(item) <= 0:
        raise ValueError('could not find item')
    elif len(item) == 1:
        try:
            return obj[item[0]]
        except KeyError as e:
            return ValueError(f'could not find item: {e}')

    return resolve(obj[item[0]], item[1:], sep=sep)


def satisfies_selection(obj, selection):
    for s in selection:
        key, expected = s.split('=')
        try:
            actual = str(resolve(obj, key))
        except ValueError:
            return False

        if expected != actual:
            return False

    return True


def main():
    opt = get_options()
    opt.results_folder = Path(opt.results_folder)

    if not opt.results_folder.is_dir():
        print(f'{opt.results_folder} is no directory')
        return

    if opt.action in ['mv', 'cp'] and opt.output_folder is None:
        print(f'action is {opt.action} but no output folder was specified')
        return

    l = []
    for p in opt.results_folder.iterdir():
        if not FinishedExperiment.is_valid_folder(p):
            if opt.verbose:
                print(f'skipping {p} as it is no finished experiment folder')
            continue

        try:
            e = FinishedExperiment.from_folder(p, ds_root=opt.ds_root)
        except ValueError as e:
            print(f'failed to load finished experiment: {e}')
            continue

        if not satisfies_selection(e, opt.selection):
            continue

        l.append((p, e))

    for p, e in l:
        if opt.action == 'ls':
            print(p)
        elif opt.action in ['mv', 'cp']:
            dst = opt.output_folder.joinpath(p.name)
            if dst.exists():
                print(f'finished experiment {p} exists already in {opt.output_folder}')
            else:
                print(f'archiving finished experiment {p} to {opt.output_folder}')
                if opt.action == 'mv':
                    shutil.move(p, dst)
                else:
                    shutil.copytree(p, dst)
        else:
            print(f'invalid action {opt.action}')
            return

    if opt.verbose:
        print(f'found {len(l)} finished experiment(s).')


if __name__ == '__main__':
    main()
