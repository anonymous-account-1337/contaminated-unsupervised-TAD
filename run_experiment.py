import argparse
import logging
import sys
import os
import traceback
import multiprocessing
import torch

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


def conduct_experiment(exp: Experiment, force: bool, use_stderr=False, device=None):
    if device is not None:
        exp._device = device

    try:
        exp.run(force)
    except Exception as e:
        print(f'experiment {exp.id} failed: {e}')
        with open(exp.output_folder.joinpath('exception.txt'), 'w') as f:
            traceback.print_exc(file=f)
        if use_stderr:
            traceback.print_exc(file=sys.stderr)
    else:
        print(f'experiment {exp.id} finished.')


def mute():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


class DeviceProcess(multiprocessing.Process):
    """a process tied to a specific (CUDA) device"""

    def __init__(self, q, force, device):
        super().__init__()
        self.q = q
        self.force = force
        self.device = device

    def run(self):
        mute()
        while (experiment := self.q.get()) is not None:
            conduct_experiment(experiment, force=self.force, device=self.device)


def get_devices():
    if torch.cuda.is_available():
        devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    else:
        devices = ['cpu']
    return devices


def main():
    opt = get_options()
    experiments = Experiment.of(opt.experiment_definition, recursive=opt.recursive, verbose=opt.verbose)
    print(f'successfully loaded {len(experiments)} experiment(s) in total.')

    if opt.dry_run:
        print('dry run -> quitting.')
        return

    devices = get_devices()
    if opt.num_processes <= 0:
        print(f'number of processes must be gt 0')
    elif opt.num_processes == 1:
        for exp in experiments:
            conduct_experiment(exp, force=opt.force, use_stderr=True, device=devices[0])
    else:
        exp_q = multiprocessing.Queue()
        for exp in experiments:
            exp_q.put(exp)

        processes = []
        for i in range(opt.num_processes):
            process = DeviceProcess(q=exp_q, force=opt.force, device=devices[i % len(devices)])
            process.start()
            processes.append(process)
            exp_q.put(None)  # for each worker process, add a sentinel values

        with tqdm(unit=' experiment(s)', total=len(experiments)) as p_bar:
            while len(processes) > 0:
                process = processes.pop(0)
                process.join(timeout=1)
                if process.is_alive():
                    processes.append(process)

                p_bar.update(n=len(experiments) - max(0, exp_q.qsize() - opt.num_processes) - p_bar.n)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  # requirement to use CUDA
    main()
