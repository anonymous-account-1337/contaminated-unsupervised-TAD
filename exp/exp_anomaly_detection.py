from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_recall_curve, f1_score
from utils.plot import ts_plot, loss_plot, precision_recall_plot, f1_score_plot
from tqdm import tqdm
from calflops import calculate_flops
import torch.multiprocessing
from models.TranAD2 import Model as TranAD
from models.USAD import Model as USAD

torch.multiprocessing.set_sharing_strategy('file_system')
import json
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import os
import warnings
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)
        self.results = {}

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.supervised_anomaly_detection:
            criterion = nn.BCELoss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, epoch):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                if isinstance(self.model, (TranAD, USAD)):
                    loss = self.model.train_step(batch_x, epoch=epoch)
                else:
                    outputs = self.model(batch_x, None, None, None)

                    if self.args.supervised_anomaly_detection:
                        batch_y = batch_y.float().to(self.device)
                        loss = criterion(F.sigmoid(outputs), batch_y)
                    else:
                        loss = criterion(outputs, batch_x)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        start_time = datetime.now()
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        dummy_sample = train_data.get_dummy_sample().to(self.device)  # ones with same shape as train_data
        if isinstance(self.model, (TranAD, USAD)):
            flops, macs, num_parameters = calculate_flops(model=self.model, args=[dummy_sample], output_as_string=False, print_results=False, print_detailed=False)
        else:
            dummy = torch.zeros(size=(1,), device=self.device)  # necessary fill value -> not used
            flops, macs, num_parameters = calculate_flops(model=self.model, args=[dummy_sample, dummy, dummy, dummy], output_as_string=False, print_results=False, print_detailed=False)

        self.results['flops'] = flops
        self.results['macs'] = macs
        self.results['num_parameters'] = num_parameters

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=False)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        train_losses = []
        vali_losses = []
        epoch = 0

        with tqdm(unit='epoch', total=self.args.train_epochs) as progress:
            for epoch in range(1, self.args.train_epochs + 1):
                train_loss = []

                self.model.train()
                for i, (batch_x, batch_y) in enumerate(train_loader):
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)

                    if isinstance(self.model, (TranAD, USAD)):
                        loss = self.model.train_step(batch_x, epoch=epoch)
                    else:
                        outputs = self.model(batch_x, None, None, None)

                        if self.args.supervised_anomaly_detection:
                            batch_y = batch_y.float().to(self.device)
                            loss = criterion(F.sigmoid(outputs), batch_y)
                        else:
                            loss = criterion(outputs, batch_x)

                    train_loss.append(loss.item())
                    loss.backward()
                    model_optim.step()

                train_loss = np.average(train_loss)
                vali_loss = self.vali(vali_data, vali_loader, criterion, epoch)

                train_losses.append(train_loss)
                vali_losses.append(vali_loss)

                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print('stopping early.')
                    break

                progress.update(1)
                progress.set_postfix({
                    'train_loss': train_loss,
                    'vali_loss': vali_loss,
                    'early_stopping_in': early_stopping.current_patience,
                })

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        loss_plot(train_losses, vali_losses, out_file=self.args.output_folder.joinpath('loss.png'))

        end_time = datetime.now()
        self.results['start_time'] = start_time.isoformat()
        self.results['end_time'] = end_time.isoformat()
        self.results['elapsed_time_sec'] = (end_time - start_time).total_seconds()
        self.results['epochs'] = epoch
        self.results['train_loss'] = train_losses
        self.results['val_loss'] = vali_losses
        self.results['optimizer'] = model_optim.__class__.__name__
        self.results['criterion'] = criterion.__class__.__name__

        return self.model

    def get_scores_and_labels(self, data_loader, callback=None):
        scores = []
        labels = []

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)

                if isinstance(self.model, (TranAD, USAD)):
                    score = self.model.anomaly_score(batch_x)
                    score = score.detach().cpu().numpy()
                    outputs = None
                    # outputs = self.model(batch_x, focus_score=torch.zeros_like(batch_x))
                    # score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                    # score = score.detach().cpu().numpy()
                    # outputs = outputs.detach().cpu().numpy()
                else:
                    # reconstruction
                    outputs = self.model(batch_x, None, None, None)
                    # criterion
                    if self.args.supervised_anomaly_detection:
                        score = F.sigmoid(outputs).detach().cpu().numpy()
                    else:
                        score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                        score = score.detach().cpu().numpy()

                    outputs = outputs.detach().cpu().numpy()

                scores.extend(score)

                label = batch_y.detach().cpu().numpy()
                labels.extend(label)

                if callback is not None:
                    callback(i, batch_x.detach().cpu().numpy(), label, score, outputs)

        scores = np.array(scores)
        labels = np.array(labels, dtype=int)

        return scores, labels

    def test(self, setting, test=0):
        val_data, val_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)
        # (1) statistic on the data sets
        val_energy, val_labels = self.get_scores_and_labels(val_loader)
        test_energy, test_labels = self.get_scores_and_labels(test_loader)

        np.save(self.args.output_folder.joinpath('val_labels.npy'), val_labels)
        np.save(self.args.output_folder.joinpath('val_energy.npy'), val_energy)

        np.save(self.args.output_folder.joinpath('test_labels.npy'), test_labels)
        np.save(self.args.output_folder.joinpath('test_energy.npy'), test_energy)

        # (2) find different thresholds using the validation set
        sample_wise_ar = sample_wise_anomaly_ratio(val_labels) * 100

        val_energy = val_energy.reshape(-1)
        val_labels = val_labels.reshape(-1)
        test_energy = test_energy.reshape(-1)
        test_labels = test_labels.reshape(-1)

        point_wise_ar = point_wise_anomaly_ratio(val_labels) * 100

        threshold_sample_wise = float(np.percentile(val_energy, 100 - sample_wise_ar))
        threshold_point_wise = float(np.percentile(val_energy, 100 - point_wise_ar))
        val_f1_scores, val_thresholds = get_f1_scores_and_thresholds(y_true=val_labels, y_score=val_energy)
        threshold_max_f1_score = float(val_thresholds[val_f1_scores.argmax()])

        # (3) evaluation on the test set
        pred_sample_wise = (test_energy > threshold_sample_wise).astype(int)
        pred_point_wise = (test_energy > threshold_point_wise).astype(int)
        pred_max_f1_score = (test_energy > threshold_max_f1_score).astype(int)

        metrics = {
            'results': self.results,
            'sample-wise': get_metrics(test_labels, pred_sample_wise, point_adjust=False, threshold=threshold_sample_wise),
            'point-wise': get_metrics(test_labels, pred_point_wise, point_adjust=False, threshold=threshold_point_wise),
            'max-score': get_metrics(test_labels, pred_max_f1_score, point_adjust=False, threshold=threshold_max_f1_score),
        }

        test_precisions, test_recalls, _ = precision_recall_curve(y_true=test_labels, probas_pred=test_energy)
        test_f1_scores, test_thresholds = get_f1_scores_and_thresholds(y_true=test_labels, y_score=test_energy)

        precision_recall_plot(test_precisions, test_recalls, out_file=self.args.output_folder.joinpath('precision-recall.png'))
        f1_score_plot(test_thresholds, test_f1_scores, metrics=metrics, out_file=self.args.output_folder.joinpath('f1-score.png'))

        metrics_json = json.dumps(metrics, indent=4)
        print(metrics_json)

        with open(self.args.output_folder.joinpath(f'metrics.json'), 'w') as f:
            f.write(metrics_json)

        # (3.1) plot a few test data samples
        if self.args.plot:
            plot_folder = self.args.output_folder.joinpath('plots')
            plot_folder.mkdir(exist_ok=True, parents=True)
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
                        ts_plot(x, y_true=y, y_score=score, reconstructed=r, y_score_min_max=y_score_min_max, out_file=plot_folder.joinpath(f'batch_{batch_idx}-idx_{i}.png'))

            self.get_scores_and_labels(test_loader, callback=plot_callback)


def get_f1_scores_and_thresholds(y_true, y_score):
    precisions, recalls, thresholds = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    f1_scores = np.array([2 * (p * r) / (p + r) for p, r in zip(precisions[:-1], recalls[:-1])])
    np.nan_to_num(f1_scores, copy=False, nan=0.0, posinf=0.0, neginf=0.0)  # nan can happen if precision + recall = 0
    return f1_scores, thresholds


def point_wise_anomaly_ratio(gt):
    res = dict(zip(*np.unique(gt, return_counts=True)))

    normal_points = res[0] if 0 in res else 0
    anomalous_points = res[1] if 1 in res else 0
    number_of_points = normal_points + anomalous_points

    if number_of_points == 0:
        raise ValueError('no points available')

    return anomalous_points / number_of_points


def sample_wise_anomaly_ratio(gt):
    if len(gt.shape) <= 1:
        raise ValueError('gt must not be flattened')

    num_samples = gt.shape[0]
    if num_samples == 0:
        raise ValueError('no samples available')

    anomalous_samples = sum([1 if len(y[y == 1]) > 0 else 0 for y in gt])
    return anomalous_samples / num_samples


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
