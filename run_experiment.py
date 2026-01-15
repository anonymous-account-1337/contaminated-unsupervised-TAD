import argparse
import logging
import sys
import os
import traceback
import multiprocessing

from experiment import Experiment
from tqdm import tqdm


logger = logging.getLogger(__name__)


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_definition', nargs='+', help='a file, multiple files or a folder')
    parser.add_argument('-r', '--recursive', action='store_true', default=False)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-f', '--force', action='store_true', default=False)
    parser.add_argument('-n', '--num-processes', type=int, required=False, default=1)
    parser.add_argument('--dry-run', action='store_true', default=False)
    return parser.parse_args()


def conduct_experiment(exp: Experiment, force: bool):
    try:
        exp.run(force)
    except Exception as e:
        print(f'experiment {exp.id} failed: {e}')
        with open(exp.output_folder.joinpath('exception.txt'), 'w') as f:
            traceback.print_exc(file=f)
    else:
        print(f'experiment {exp.id} finished.')


def mute():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


def main():
    opt = get_options()
    experiments = Experiment.of(opt.experiment_definition, recursive=opt.recursive, verbose=opt.verbose)
    print(f'successfully loaded {len(experiments)} experiment(s) in total.')

    if opt.dry_run:
        print('dry run -> quitting.')
        return

    if opt.num_processes <= 0:
        print(f'number of processes must be gt 0')
    elif opt.num_processes == 1:
        for exp in experiments:
            conduct_experiment(exp, opt.force)
    else:
        with multiprocessing.Pool(processes=opt.num_processes, initializer=mute) as pool:
            futures = [pool.apply_async(conduct_experiment, (exp, opt.force)) for exp in experiments]

            with tqdm(total=len(futures), unit=' experiment(s)') as p_bar:
                while len(futures) > 0:
                    future = futures.pop(0)
                    try:
                        future.get(timeout=0.5)
                        p_bar.update(1)
                    except multiprocessing.TimeoutError:
                        futures.append(future)
                    except BaseException as ex:
                        logger.warning(ex)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  # requirement to use CUDA
    main()
