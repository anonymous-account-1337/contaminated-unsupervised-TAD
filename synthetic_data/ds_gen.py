import argparse
import itertools
import logging
import multiprocessing
import json

from pathlib import Path
from synthetic_data.ds_def import DatasetDefinition


logger = logging.getLogger(__name__)


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_definition', type=str, nargs='+', help='a dataset definition file or a folder containing dataset definition files')
    parser.add_argument('-o', '--output-folder', type=str, required=True)
    parser.add_argument('-r', '--recursive', action='store_true', default=False)
    parser.add_argument('-f', '--force', action='store_true', default=False)
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('-n', '--num-processes', type=int, required=False, default=1)
    parser.add_argument('--dry-run', type=int, default=None)
    args = parser.parse_args()
    args.dataset_definition = [Path(e) for e in args.dataset_definition]
    args.output_folder = Path(args.output_folder)
    return args


def main():
    opt = get_options()
    dataset_definitions = list(itertools.chain.from_iterable(DatasetDefinition.of(e, recursive=opt.recursive) for e in opt.dataset_definition))
    print(f'Loaded {len(dataset_definitions)} dataset definition(s).')

    if opt.dry_run is None:
        if opt.num_processes <= 0:
            print(f'number of processes must be gt 0')
        elif opt.num_processes == 1:
            for d in dataset_definitions:
                d.save_to(opt.output_folder, opt.force, opt.seed)
        else:
            with multiprocessing.Pool(processes=opt.num_processes) as pool:
                futures = [pool.apply_async(d.save_to, (opt.output_folder, opt.force, opt.seed)) for d in dataset_definitions]

                for future in futures:
                    try:
                        future.get()
                    except BaseException as ex:
                        logger.warning(ex)
    else:
        """dry runs can be used to enumerate multiple datasets for definitions in experiments conveniently"""
        output_paths = [Path(opt.output_folder).joinpath(d.name) for d in dataset_definitions]
        output_paths = list(map(lambda s: str(s).replace('\\', '/'), output_paths))
        step_size = len(output_paths) // max(opt.dry_run, 1)

        for e in [output_paths[i:i + step_size] for i in range(0, len(output_paths), step_size)]:
            print(json.dumps(e))


if __name__ == '__main__':
    main()
