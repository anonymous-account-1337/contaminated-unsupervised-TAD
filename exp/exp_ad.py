import json
import random
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from datetime import datetime
from pathlib import Path

from calflops import calculate_flops
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_recall_curve

from utils.sam import SAMSGD
from utils.tools import adjustment
from utils.plot import ts_plot, loss_plot, f1_score_plot
from utils.training_monitor import TrainingMonitor
from data_provider.anomaly_data_loader import AnomalyDataset
from models.TranAD import Model as TranAD
from models.USAD import Model as USAD
from models.TimesNetModified import Model as TimesNet
from models.SimpleFormer import Model as Transformer


def init_seed(seed=47):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def use_deterministic_algorithms(flag):
    if flag:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.backends.cudnn.benchmark = not flag
    torch.use_deterministic_algorithms(mode=flag, warn_only=False)


class AnomalyDetectionExperiment:

    MODELS = {
        'TranAD': TranAD,
        'USAD': USAD,
        'TimesNet': TimesNet,
        'Transformer': Transformer,
    }

    def __init__(self, exp):
        self.exp = exp

        if hasattr(self.exp, 'seed'):
            init_seed(self.exp.seed)
            use_deterministic_algorithms(True)

        self._device = None
        self.results = {}
        self.model = self.create_model()
        self.calc_flops()
        self.epoch = 0

    @property
    def device(self):
        if self._device is None:
            if hasattr(self.exp, 'gpu'):
                self._device = torch.device(self.exp.gpu)
            elif hasattr(self.exp, 'gpu_name'):
                # Caution: nvidia-smi GPU-ID and CUDA GPU-ID do not match!
                # Therefore, setting an explicit GPU name may be desirable.
                # e.g. 'NVIDIA RTX A6000' or 'NVIDIA GeForce RTX 4090'
                for i in range(torch.cuda.device_count()):
                    if torch.cuda.get_device_properties(i).name == self.exp.gpu_name:
                        self._device = torch.device(f'cuda:{i}')
                        break

                if self._device is None:
                    raise ValueError(f'could not find gpu with name {self.exp.gpu_name}')
            else:
                self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        return self._device

    @property
    def checkpoint_path(self):
        path = Path(self.exp.output_folder).joinpath('checkpoints')
        path.mkdir(parents=False, exist_ok=True)
        return path

    @property
    def best_model_checkpoint_path(self):
        return self.checkpoint_path.joinpath('best.pt')

    @property
    def plot_path(self):
        plots = Path(self.exp.output_folder).joinpath('plots')
        plots.mkdir(parents=False, exist_ok=True)
        return plots

    def create_model(self):
        return self.MODELS[self.exp.model](self.exp).to(device=self.device, dtype=torch.float32)

    def calc_flops(self):
        dummy_sample = self.get_data_set('train').get_dummy_sample().to(self.device)  # ones with same shape as train_data
        flops, macs, num_parameters = calculate_flops(model=self.model, args=[dummy_sample], output_as_string=False, print_results=False, print_detailed=False)

        self.results['flops'] = flops
        self.results['macs'] = macs
        self.results['num_parameters'] = num_parameters

    def get_data_set(self, flag):
        return AnomalyDataset(self.exp.root_path, flag, device=self.device)

    def get_optimizer(self):
        if hasattr(self.exp, 'optimizer'):
            if self.exp.optimizer == 'SAMSGD':
                return SAMSGD(self.model.parameters(), lr=self.exp.learning_rate)
            else:
                raise ValueError(f'invalid optimizer {self.exp.optimizer}')

        return Adam(self.model.parameters(), lr=self.exp.learning_rate)

    def train(self):
        start_time = datetime.now()

        train_loader = DataLoader(self.get_data_set('train'), batch_size=self.exp.batch_size, shuffle=True)
        optimizer = self.get_optimizer()

        train_losses = []
        vali_losses = []

        monitor = TrainingMonitor(self.exp.patience)

        epoch_iterator = tqdm(range(1, self.exp.train_epochs + 1))
        for epoch in epoch_iterator:
            self.epoch = epoch
            batch_losses = []

            self.model.train()
            for i, (batch_x, _) in enumerate(train_loader):
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)

                if isinstance(optimizer, SAMSGD):
                    def closure():
                        optimizer.zero_grad()
                        _loss = self.model.train_step(batch_x, epoch=epoch)
                        _loss.backward()
                        return _loss

                    loss = optimizer.step(closure)
                else:
                    optimizer.zero_grad()
                    loss = self.model.train_step(batch_x, epoch=epoch)
                    loss.backward()
                    optimizer.step()

                batch_losses.append(loss.item())

            train_losses.append(np.average(batch_losses))
            vali_losses.append(self.vali())

            if monitor(vali_losses[-1]):
                torch.save(self.model.state_dict(), self.best_model_checkpoint_path)

            if monitor.should_early_stop:
                print('patience exhausted -> stopping early.')
                break

            postfix = {
                'train_loss': train_losses[-1],
                'vali_loss': vali_losses[-1],
                'lowest_vali_loss': monitor.lowest_loss,
            }

            if monitor.early_stopping_enabled:
                postfix['patience'] = monitor.current_patience

            epoch_iterator.set_postfix(postfix)

        self.model.load_state_dict(torch.load(self.best_model_checkpoint_path, weights_only=True, map_location=self.device))
        loss_plot(train_losses, vali_losses, out_file=self.exp.output_folder.joinpath('loss.png'))

        end_time = datetime.now()
        self.results['start_time'] = start_time.isoformat()
        self.results['end_time'] = end_time.isoformat()
        self.results['elapsed_time_sec'] = (end_time - start_time).total_seconds()
        self.results['epochs'] = self.epoch
        self.results['train_loss'] = train_losses
        self.results['val_loss'] = vali_losses
        self.results['optimizer'] = optimizer.__class__.__name__

    def vali(self):
        val_loader = DataLoader(self.get_data_set('val'), batch_size=self.exp.batch_size, shuffle=False)
        batch_losses = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(val_loader):
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)
                loss = self.model.train_step(batch_x, epoch=self.epoch)
                batch_losses.append(loss.item())

        return np.average(batch_losses)

    def get_scores_and_labels(self, data_loader, callback=None):
        scores = []
        labels = []

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(data_loader):
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)

                score = self.model.anomaly_score(batch_x)
                score = score.detach().cpu().numpy()
                scores.extend(score)

                label = batch_y.detach().cpu().numpy()
                labels.extend(label)

                if callback is not None:
                    callback(i, batch_x.detach().cpu().numpy(), label, score, None)

        scores = np.array(scores)
        labels = np.array(labels, dtype=int)

        return scores, labels

    def test(self):
        val_loader = DataLoader(self.get_data_set('val'), batch_size=self.exp.batch_size, shuffle=False)
        test_loader = DataLoader(self.get_data_set('test'), batch_size=self.exp.batch_size, shuffle=False)

        self.model.eval()
        # (1) statistic on the data sets
        val_energy, val_labels = self.get_scores_and_labels(val_loader)
        test_energy, test_labels = self.get_scores_and_labels(test_loader)

        np.save(self.exp.output_folder.joinpath('val_labels.npy'), val_labels)
        np.save(self.exp.output_folder.joinpath('val_energy.npy'), val_energy)

        np.save(self.exp.output_folder.joinpath('test_labels.npy'), test_labels)
        np.save(self.exp.output_folder.joinpath('test_energy.npy'), test_energy)

        # (2) find threshold which leads to the maximum f1-score using the validation set
        val_energy = val_energy.reshape(-1)
        val_labels = val_labels.reshape(-1)
        test_energy = test_energy.reshape(-1)
        test_labels = test_labels.reshape(-1)

        # reduce different anomaly labels to binary labels
        val_labels[val_labels > 0] = 1
        test_labels[test_labels > 0] = 1

        threshold_max_f1_score = get_max_f1_score_threshold(y_true=val_labels, y_score=val_energy)

        # (3) evaluation on the test set
        test_pred = (test_energy > threshold_max_f1_score).astype(int)

        metrics = {
            'results': self.results,
            'max-score': get_metrics(test_labels, test_pred, point_adjust=False, threshold=threshold_max_f1_score),
            'pa-max-score': get_metrics(test_labels, test_pred, point_adjust=True, threshold=threshold_max_f1_score),
        }

        test_f1_scores, test_thresholds = get_f1_scores_and_thresholds(y_true=test_labels, y_score=test_energy)
        f1_score_plot(test_thresholds, test_f1_scores, metrics=metrics, out_file=self.exp.output_folder.joinpath('f1-score.png'))

        metrics_json = json.dumps(metrics, indent=4)
        print(metrics_json)

        with open(self.exp.output_folder.joinpath('metrics.json'), 'w') as f:
            f.write(metrics_json)

        # (3.1) plot a few test data samples
        if self.exp.plot:
            y_score_min_max = (test_energy.min(), test_energy.max())

            def plot_callback(batch_idx, x_batch, y_batch, score_batch, reconstructed_batch):
                if reconstructed_batch is None:
                    reconstructed_batch = np.zeros(shape=x_batch.shape, dtype=np.float32)
                    reconstructed_batch.fill(np.nan)

                for i, (x, y, score, r) in enumerate(zip(x_batch, y_batch, score_batch, reconstructed_batch)):
                    x = x.reshape(-1)
                    y = y.reshape(-1)
                    r = r.reshape(-1)
                    r = None if np.isnan(r).all() else r

                    if len(np.nonzero(y)[0]) > 0:
                        ts_plot(x, y_true=y, y_score=score, reconstructed=r, y_score_min_max=y_score_min_max, out_file=self.plot_path.joinpath(f'batch_{batch_idx}-idx_{i}.png'))

            self.get_scores_and_labels(test_loader, callback=plot_callback)


def get_max_f1_score_threshold(y_true, y_score):
    f1_scores, thresholds = get_f1_scores_and_thresholds(y_true=y_true, y_score=y_score)
    return float(thresholds[f1_scores.argmax()])


def get_f1_scores_and_thresholds(y_true, y_score):
    precisions, recalls, thresholds = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    f1_scores = np.array([2 * (p * r) / (p + r) for p, r in zip(precisions[:-1], recalls[:-1])])
    np.nan_to_num(f1_scores, copy=False, nan=0.0, posinf=0.0, neginf=0.0)  # nan can happen if precision + recall = 0
    return f1_scores, thresholds


def get_metrics(gt, pred, point_adjust=False, **kwargs):
    pred = np.array(pred)
    gt = np.array(gt)

    if point_adjust:
        gt, pred = adjustment(gt, pred)

    accuracy = accuracy_score(gt, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(gt, pred, average='binary')

    metrics = {
        'point_adjust': point_adjust,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }

    return {
        **kwargs,
        **metrics
    }
