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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_recall_curve, average_precision_score, roc_auc_score

from utils.sam import SAMSGD
from utils.plot import ts_plot, loss_plot
from utils.training_monitor import TrainingMonitor
from data.anomaly_data_loader import AnomalyDataset, CorruptedAnomalyDataset
from models.TranAD import Model as TranAD
from models.Autoencoder import Model as Autoencoder
from models.USAD import Model as USAD
from models.TimesNet import Model as TimesNet
from models.SimpleFormer import Model as Transformer
from models.rnn import Model as RNNModel
from models.LatentFormer import Model as LatentFormer


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
        'Autoencoder': Autoencoder,
        'TranAD': TranAD,
        'USAD': USAD,
        'TimesNet': TimesNet,
        'Transformer': Transformer,
        'RNNModel': RNNModel,
        'LatentFormer': LatentFormer,
    }

    def __init__(self, exp):
        self.exp = exp

        if hasattr(self.exp, 'seed'):
            init_seed(self.exp.seed)
            use_deterministic_algorithms(True)

        self._device = None
        self.results = {}
        self.derive_experiment_parameters()
        self.model = self.create_model()
        self.calc_flops()
        self.epoch = 0

    def is_data_set_available(self, flag):
        try:
            self.get_data_set(flag)
            return True
        except FileNotFoundError:
            return False

    @property
    def device(self):
        if self._device is None:
            if hasattr(self.exp, 'device'):
                self._device = torch.device(self.exp.device)
            elif hasattr(self.exp, 'cuda_device_name'):
                # Caution: nvidia-smi GPU-ID and CUDA GPU-ID do not match!
                # Therefore, setting an explicit GPU name may be desirable.
                # e.g. 'NVIDIA RTX A6000' or 'NVIDIA GeForce RTX 4090'
                for i in range(torch.cuda.device_count()):
                    if torch.cuda.get_device_properties(i).name == self.exp.cuda_device_name:
                        self._device = torch.device(f'cuda:{i}')
                        break

                if self._device is None:
                    raise ValueError(f'could not find gpu with name {self.exp.cuda_device_name}')
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

    def derive_experiment_parameters(self):
        _, seq_len, num_features = self.get_data_set('train').get_dummy_sample().shape
        self.exp.seq_len = seq_len
        self.exp.enc_in = num_features
        self.exp.c_out = num_features

    def get_allocated_memory(self):
        """returns the allocated memory in GB"""
        return torch.cuda.memory_allocated(device=self.device) / 10 ** 9

    def create_model(self):
        return self.MODELS[self.exp.model](self.exp).to(device=self.device, dtype=torch.float32)

    def calc_flops(self):
        dummy_sample = self.get_data_set('train').get_dummy_sample().to(self.device)  # ones with same shape as train_data
        flops, macs, num_parameters = calculate_flops(model=self.model, args=[dummy_sample], output_as_string=False, print_results=False, print_detailed=False)

        self.results['flops'] = flops
        self.results['macs'] = macs
        self.results['num_parameters'] = num_parameters

    def get_data_set(self, flag):
        ds_path = os.getenv('DATASET_PATH')
        if ds_path is None:
            path = self.exp.root_path
        else:
            path = Path(ds_path).joinpath(self.exp.root_path)

        return AnomalyDataset(path, flag)

    def get_corrupted_data_set(self, flag):
        ds_path = os.getenv('DATASET_PATH')
        if ds_path is None:
            path = self.exp.root_path
        else:
            path = Path(ds_path).joinpath(self.exp.root_path)

        return CorruptedAnomalyDataset(path, flag)

    def get_fine_tuning_data_set(self, flag):
        return AnomalyDataset(self.exp.fine_tune_path, flag, load_clean=True)

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
        train_losses = []
        vali_losses = [] if self.is_data_set_available('val') else None

        monitor = TrainingMonitor(self.exp.patience)
        trainer = self.model.get_trainer()

        epoch_iterator = tqdm(range(1, self.exp.train_epochs + 1))
        for epoch in epoch_iterator:
            self.epoch = epoch
            batch_losses = []

            self.model.train()
            for i, (batch_x, _) in enumerate(train_loader):
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)
                loss = trainer.train_step(batch_x, epoch=epoch)
                batch_losses.append(loss if isinstance(loss, tuple) else (loss, ))

            if hasattr(self.exp, 'fine_tune_path') and self.exp.fine_tune_path is not None:
                self.fine_tune(trainer)

            train_losses.append(np.average(np.array(batch_losses), axis=0))

            if vali_losses is None:
                torch.save(self.model.state_dict(), self.best_model_checkpoint_path)
            else:
                vali_losses.append(self.vali(trainer))
                if monitor(vali_losses[-1]):
                    torch.save(self.model.state_dict(), self.best_model_checkpoint_path)

                if monitor.should_early_stop:
                    print('patience exhausted -> stopping early.')
                    break

            postfix = {
                'train_loss': ', '.join(map(lambda e: f'l{e[0]}={e[1]:.4f}', enumerate(train_losses[-1], start=1))),
                'vali_loss': None if vali_losses is None else ', '.join(map(lambda e: f'l{e[0]}={e[1]:.4f}', enumerate(vali_losses[-1], start=1))),
                'lowest_vali_loss': monitor.lowest_loss,
            }

            if monitor.early_stopping_enabled:
                postfix['patience'] = monitor.current_patience

            epoch_iterator.set_postfix(postfix)

        train_losses = np.array(train_losses)

        if vali_losses is not None:
            vali_losses = np.array(vali_losses)

        self.model.load_state_dict(torch.load(self.best_model_checkpoint_path, weights_only=True, map_location=self.device))
        loss_plot(train_losses, vali_losses, out_file=self.exp.output_folder.joinpath('loss.png'))

        end_time = datetime.now()
        self.results['start_time'] = start_time.isoformat()
        self.results['end_time'] = end_time.isoformat()
        self.results['elapsed_time_sec'] = (end_time - start_time).total_seconds()
        self.results['epochs'] = self.epoch
        self.results['train_loss'] = train_losses.tolist()
        self.results['val_loss'] = None if vali_losses is None else vali_losses.tolist()

    def fine_tune(self, trainer):
        train_loader = DataLoader(self.get_fine_tuning_data_set('train'), batch_size=self.exp.batch_size, shuffle=False)
        batch_losses = []

        self.model.train()
        for i, (batch_x_anom, batch_x_clean, _) in enumerate(train_loader):
            batch_x_anom = batch_x_anom.to(device=self.device, dtype=torch.float32)
            batch_x_clean = batch_x_clean.to(device=self.device, dtype=torch.float32)
            loss = trainer.guided_reconstruction(batch_x_anom, batch_x_clean)
            batch_losses.append(loss if isinstance(loss, tuple) else (loss, ))

        return np.average(np.array(batch_losses), axis=0)

    def vali(self, trainer):
        self.model.eval()
        val_loader = DataLoader(self.get_data_set('val'), batch_size=self.exp.batch_size, shuffle=False)
        batch_losses = []

        with torch.no_grad():
            for i, (batch_x, _) in enumerate(val_loader):
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)
                loss = trainer.validation_step(batch_x, epoch=self.epoch)
                batch_losses.append(loss if isinstance(loss, tuple) else (loss, ))

        return np.average(np.array(batch_losses), axis=0)

    def get_scores_and_labels(self, data_loader, reconstruct=False, plot=False):
        scores = []
        labels = []
        trainer = self.model.get_trainer()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(data_loader):
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)

                if reconstruct and hasattr(trainer, 'reconstruct'):
                    reconstructed = trainer.reconstruct(batch_x)
                    reconstructed = reconstructed.detach().cpu().numpy()
                else:
                    reconstructed = None

                score = trainer.anomaly_score(batch_x)
                score = score.detach().cpu().numpy()
                score = np.sum(score, axis=-1).reshape(score.shape[0], score.shape[1], 1)
                scores.extend(score)

                label = batch_y.detach().cpu().numpy()
                labels.extend(label)

                if plot:
                    self.plot_batch(i, batch_x.detach().cpu().numpy(), label, score, reconstructed, only_anomalous=True)

        scores = np.array(scores)
        labels = np.array(labels, dtype=int)

        return scores, labels

    def get_val_threshold(self):
        val_loader = DataLoader(self.get_data_set('val'), batch_size=self.exp.batch_size, shuffle=False)
        val_energy, val_labels = self.get_scores_and_labels(val_loader)
        val_energy = val_energy.reshape(-1)
        val_labels = val_labels.reshape(-1)
        val_labels[val_labels > 0] = 1

        threshold_max_f1_score = get_max_f1_score_threshold(y_true=val_labels, y_score=val_energy)
        return threshold_max_f1_score

    def get_test_metrics(self, ds, threshold_max_f1_score):
        test_loader = DataLoader(ds, batch_size=self.exp.batch_size, shuffle=False)
        test_energy, test_labels = self.get_scores_and_labels(test_loader)
        test_energy = test_energy.reshape(-1)
        test_labels = test_labels.reshape(-1)
        test_labels[test_labels > 0] = 1
        # threshold_max_f1_score = get_max_f1_score_threshold(y_true=test_labels, y_score=test_energy)

        auprc = average_precision_score(test_labels, test_energy)
        auroc = roc_auc_score(test_labels, test_energy)
        test_pred = (test_energy > threshold_max_f1_score).astype(int)

        metrics = get_metrics(test_labels, test_pred, point_adjust=False, threshold=threshold_max_f1_score, auprc=auprc, auroc=auroc)
        metrics_pa = get_metrics(test_labels, test_pred, point_adjust=True, threshold=threshold_max_f1_score)

        return metrics, metrics_pa

    def test(self):
        self.model.eval()

        threshold_max_f1_score = self.get_val_threshold()
        metrics = dict(self.results)
        m, m_pa = self.get_test_metrics(self.get_data_set('test'), threshold_max_f1_score)
        metrics['eval'] = m
        metrics['eval_pa'] = m_pa

        m, m_pa = self.get_test_metrics(self.get_corrupted_data_set('test'), threshold_max_f1_score)
        metrics['cor_eval'] = m
        metrics['cor_eval_pa'] = m_pa

        metrics_json = json.dumps(metrics, indent=4)
        print(metrics_json)

        with open(self.exp.output_folder.joinpath('metrics.json'), 'w') as f:
            f.write(metrics_json)

        if self.exp.plot:
            self.get_scores_and_labels(DataLoader(self.get_data_set('test'), batch_size=self.exp.batch_size, shuffle=False), reconstruct=True, plot=True)

    def plot_batch(self, batch_idx, x_batch, y_batch, score_batch, reconstructed_batch, only_anomalous=False):
        if reconstructed_batch is None:
            reconstructed_batch = np.zeros(shape=x_batch.shape, dtype=np.float32)
            reconstructed_batch.fill(np.nan)

        for i, (x, y, score, r) in enumerate(zip(x_batch, y_batch, score_batch, reconstructed_batch)):
            y = y.reshape(y.shape[0])
            score = score.reshape(score.shape[0])
            anomalous = len(np.nonzero(y)[0]) > 0

            for feature in range(x.shape[-1]):
                f_x = x[:, feature]
                f_r = r[:, feature]
                f_r = None if np.isnan(f_r).all() else f_r

                if not only_anomalous or anomalous:
                    plot_path = self.plot_path.joinpath(f'f{feature}')
                    plot_path.mkdir(exist_ok=True)
                    ts_plot(f_x, y_true=y, y_score=score, reconstructed=f_r, y_score_min_max=None, out_file=plot_path.joinpath(f'b{batch_idx}-i{i}-f{feature}.png'))


def get_max_f1_score_threshold(y_true, y_score):
    f1_scores, thresholds = get_f1_scores_and_thresholds(y_true=y_true, y_score=y_score)
    return float(thresholds[f1_scores.argmax()])


def get_f1_scores_and_thresholds(y_true, y_score):
    precisions, recalls, thresholds = precision_recall_curve(y_true=y_true, y_score=y_score)
    f1_scores = np.array([2 * (p * r) / (p + r) for p, r in zip(precisions[:-1], recalls[:-1])])
    np.nan_to_num(f1_scores, copy=False, nan=0.0, posinf=0.0, neginf=0.0)  # nan can happen if precision + recall = 0
    return f1_scores, thresholds


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


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
