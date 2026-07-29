import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

from synthetic_data.plf import PiecewiseLinearFunction
from synthetic_data.base import BaseFunction


# plt.rcParams['figure.titlesize'] = 32
# plt.rcParams['figure.labelsize'] = 48
#
# plt.rcParams['axes.titlesize'] = 32
# plt.rcParams['axes.labelsize'] = 48

plt.rcParams['xtick.labelsize'] = 42
plt.rcParams['ytick.labelsize'] = 42
plt.rcParams['font.size'] = 48

# plt.rcParams['legend.title_fontsize'] = 28
plt.rcParams['legend.fontsize'] = 48


def avg_curve(mean, std, out_file=None):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(range(len(mean)), mean, color='black')
    if std is not None:
        ax.fill_between(range(len(mean)), mean - std, mean + std, color='black', alpha=.1)
    plt.tight_layout()
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)
    plt.close(fig)


def _consecutive(a):
    if len(a) == 0:
        return []

    diffs = np.diff(a) != 1
    indexes = np.nonzero(diffs)[0] + 1
    groups = np.split(a, indexes)
    return list(map(lambda g: (g[0], g[-1]), groups))


def seg_plot(x, y, intervals=None, segment_names=None, out_file=None):
    if intervals is None:
        intervals = []

    if segment_names is None:
        segment_names = [f's{i}' for i in range(len(intervals))]

    if len(intervals) != len(segment_names):
        raise ValueError('shape mismatch')

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(x, y, color='black', marker='o', linewidth=2)

    for i, (interval, segment_name) in enumerate(zip(intervals, segment_names)):
        x_start = x[min(interval[0], len(x) - 1)]
        x_end = x[min(interval[1], len(x) - 1)]
        ax.vlines(x_start, np.min(y), np.max(y), colors='black')
        ax.vlines(x_end, np.min(y), np.max(y), colors='black')

        ax.text(x_start + (x_end - x_start) / 2, np.min(y) - 60 if i % 2 == 0 else np.max(y) + 40, segment_name, ha='center', va='center')

    plt.tight_layout()
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)
    plt.close(fig)


def f_plot(f: BaseFunction, segment_names=None, title=None, out_file=None):
    if segment_names is None:
        segment_names = []

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    if title is not None:
        ax.set_title(title)

    x = np.linspace(0, f.domain[1], 100)
    y = f(x)

    ax.plot(x, y, color='black', marker='o', linewidth=2)

    for i, segment_name in enumerate(segment_names):
        interval = f.get_segment_interval(segment_name)
        ax.vlines(interval.start, np.min(y), np.max(y), colors='black')
        ax.vlines(interval.stop, np.min(y), np.max(y), colors='black')
        ax.text(interval.start + (interval.stop - interval.start) / 2, np.min(y) - 60 if i % 2 == 0 else np.max(y) + 40, segment_name, ha='center', va='center')

    plt.tight_layout()
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)
    plt.close(fig)


def plf_plot(f: PiecewiseLinearFunction, scale=False, title=None, out_file=None):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    if title is not None:
        ax.set_title(title)

    x = np.array(sorted(list(set(np.array([[f.get_segment_interval(s.name).start, f.get_segment_interval(s.name).stop] for s in f.segments]).reshape(-1).tolist()))))
    y = f(x)

    if scale:
        _mean = np.mean(y)
        _std = np.std(y)
        y = (y - _mean) / _std

        x_scale = 100 / np.max(x)
        x *= x_scale
    else:
        x_scale = 1

    ax.plot(x, y, color='black', marker='o', linewidth=2)

    for i, s in enumerate(f.segments):
        start_x, end_x = f.get_segment_interval(s.name)
        start_x *= x_scale
        end_x *= x_scale
        ax.vlines(start_x, np.min(y), np.max(y), colors='black')

        # display_name = f's{i}' if len(s.name) > 10 else s.name
        display_name = f's{i}'
        ax.text(start_x + (end_x - start_x) / 2, np.min(y) - 0.05 * np.max(y) if i % 2 == 0 else np.max(y) + 0.04 * np.max(y), display_name, ha='center', va='center', fontsize=32)

    ax.set_xlabel('timestep')
    ax.set_ylabel('voltage')
    plt.subplots_adjust(left=0.15, bottom=0.14, top=0.975, right=0.99)
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)
    plt.close(fig)


def contamination_plot(d, title=None, out_file=None):
    with plt.rc_context({
        'font.size': 32,
        'axes.labelsize': 32,
        'axes.titlesize': 8,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 32,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }):
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))

        if title is not None:
            ax.set_title(title)

        for e in d:
            order = e['order'] if 'order' in e else None
            color = e['color'] if 'color' in e else None
            line_style = e['line_style'] if 'line_style' in e else None

            ax.plot(e['ts']['ar'], e['ts']['f1'], label=e['label'], marker='o', linestyle=line_style, linewidth=2, zorder=order, color=color)

            if 'f1std' in e['ts'].columns:
                ax.fill_between(e['ts']['ar'], e['ts']['f1'] - e['ts']['f1std'], e['ts']['f1'] + e['ts']['f1std'], color=color, alpha=.1, zorder=order)

        ax.legend()
        ax.set_xlabel('training data contamination [%]', labelpad=2)
        ax.set_ylabel('standard F1-score', labelpad=2)
        ax.margins(x=0, y=0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out', length=3, width=0.8, pad=2)

        ax.grid(True, linewidth=0.5, alpha=0.3)
        ax.legend(frameon=False)

        fig.subplots_adjust(
            left=0.08,
            right=0.96,
            bottom=0.1,
            top=0.96,
        )

        if out_file is None:
            plt.show()
        else:
            plt.savefig(out_file)
        plt.close(fig)


def train_contamination_agg_plot(d, title=None, out_file=None):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    if title is not None:
        ax.set_title(title)

    mapping = {
        10000: ('#0400ff', 0, 's'),
        50000: ('#ff0000', 1, 'm'),
        100000: ('#09ff00', 2, 'l'),
        150000: ('#000000', 3, 'xl'),
    }

    legend_info = []
    for ds_size, e in d.items():
        color, order, label, = mapping[ds_size]
        handles = ax.plot(e.index, e['f1_score'], color=color, marker='o', linewidth=2, zorder=10 + order)
        for handle in handles:
            legend_info.append((order, handle, label))
        if e['f1_score_std'] is not None:
            ax.fill_between(e.index, e['f1_score'] - e['f1_score_std'], e['f1_score'] + e['f1_score_std'], color=color, alpha=.1, zorder=5 + order)

    # ax.set_ylim(0, 1)
    ax.set_ylabel('f1 score')
    ax.set_xlabel('sample-wise anomaly ratio of training data[%]')

    legend_info.sort(key=lambda e: e[0])
    handles = list(map(lambda e: e[1], legend_info))
    labels = list(map(lambda e: e[2], legend_info))
    ax.legend(handles, labels)

    plt.tight_layout()
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)
    plt.close(fig)


def f1_train_contamination_plot(df, title=None, out_file=None):
    with plt.rc_context({
        'font.size': 8,
        'axes.labelsize': 11,
        'axes.titlesize': 8,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }):
        fig, ax = plt.subplots(figsize=(3.4, 2.4))

        if title is not None:
            ax.set_title(title)

        x = df.index.to_numpy()
        y = df['f1'].to_numpy()

        ax.plot(x, y, marker='o', label='baseline', color='blue')

        if 'f1_std' in df.columns:
            y_std = df['f1_std'].to_numpy()
            ax.fill_between(
                x,
                y - y_std,
                y + y_std,
                alpha=0.2,
                linewidth=0,
                color='blue'
            )

        ax.set_xlabel('contamination [%]', labelpad=2)
        ax.set_ylabel('F1-score', labelpad=2)

        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(0.0, 1.05)
        ax.margins(x=0, y=0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out', length=3, width=0.8, pad=2)

        ax.grid(True, linewidth=0.5, alpha=0.3)
        ax.legend(frameon=False)

        fig.subplots_adjust(left=0.155, bottom=0.165, top=0.975, right=0.96)

        if out_file is None:
            plt.show()
        else:
            plt.savefig(out_file)
        plt.close(fig)


def rel_f1_train_contamination_plot(df, title=None, out_file=None):
    with plt.rc_context({
        'font.size': 8,
        'axes.labelsize': 11,
        'axes.titlesize': 8,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }):
        fig, ax = plt.subplots(figsize=(3.4, 2.4))

        if title is not None:
            ax.set_title(title)
        else:
            title = f'uncontaminated f1: {df.iloc[0]["f1"]:.2f}'
            if 'f1_std' in df.columns and not math.isnan(df.iloc[0]['f1_std']):
                title += f'\u00B1{df.iloc[0]["f1_std"]:.2f}'

            # ax.set_title(title)

        x = df.index.to_numpy()
        y = df['rel_f1'].to_numpy()

        ax.plot(x, y, marker='o', label='baseline', color='blue')

        if 'rel_f1_std' in df.columns:
            y_std = df['rel_f1_std'].to_numpy()
            ax.fill_between(
                x,
                y - y_std,
                y + y_std,
                alpha=0.2,
                linewidth=0,
                color='blue'
            )

        if 'rel_cor_f1' in df.columns:
            y = df['rel_cor_f1'].to_numpy()
            ax.plot(x, y, marker='o', label='domain shift', color='red')

            if 'rel_cor_f1_std' in df.columns:
                y_std = df['rel_cor_f1_std'].to_numpy()
                ax.fill_between(
                    x,
                    y - y_std,
                    y + y_std,
                    alpha=0.2,
                    linewidth=0,
                    color='red'
                )

        ax.set_xlabel('contamination [%]', labelpad=2)
        ax.set_ylabel('retained performance [%]', labelpad=2)

        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(0.0, 1.05)
        ax.margins(x=0, y=0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out', length=3, width=0.8, pad=2)

        ax.grid(True, linewidth=0.5, alpha=0.3)
        ax.legend(frameon=False)

        fig.subplots_adjust(left=0.155, bottom=0.165, top=0.975, right=0.96)

        if out_file is None:
            plt.show()
        else:
            plt.savefig(out_file)
        plt.close(fig)


def retained_performance_plot(d, title=None, out_file=None, color_map=None, relative=False, fixed_y=True):
    with plt.rc_context({
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }):
        if isinstance(d, pd.DataFrame):
            d = {'baseline': d}
        elif not isinstance(d, dict):
            raise TypeError

        fig, ax = plt.subplots(figsize=(3.4, 2.4))

        if relative:
            y_key_mean, y_key_std = 'rel_f1', 'rel_f1_std'
            y_label = 'retained performance [%]'
        else:
            y_key_mean, y_key_std = 'f1', 'f1_std'
            y_label = 'F1-score'

        if title is not None:
            ax.set_title(title)

        for key, df in d.items():
            x = df.index.to_numpy()
            y = df[y_key_mean].to_numpy()

            color = None if color_map is None else color_map[key]
            line_2d_list = ax.plot(x, y, marker='o', label=key, color=color)
            color = line_2d_list[0].get_color()

            if y_key_std in df.columns:
                y_std = df[y_key_std].to_numpy()
                ax.fill_between(
                    x,
                    y - y_std,
                    y + y_std,
                    alpha=0.2,
                    linewidth=0,
                    color=color
                )

        ax.set_xlabel('contamination [%]', labelpad=2)
        ax.set_ylabel(y_label, labelpad=2)

        ax.set_xlim(x.min(), x.max())
        if isinstance(fixed_y, tuple):
            ax.set_ylim(*fixed_y)
        elif fixed_y:
            ax.set_ylim(0.0, 1.05)
        else:
            ax.set_ylim(math.floor(y.min() * 10) / 10, math.ceil(y.max() * 10) / 10)

        ax.margins(x=0, y=0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out', length=3, width=0.8, pad=2)

        ax.grid(True, linewidth=0.5, alpha=0.3)
        ax.legend(frameon=False)

        fig.subplots_adjust(
            left=0.13,
            right=0.96,
            bottom=0.15,
            top=0.90,
        )

        if out_file is None:
            plt.show()
        else:
            plt.savefig(out_file)
        plt.close(fig)


def ar_sensitivity_plot(anomaly_ratios, f1_scores, point_wise_ar=None, sample_wise_ar=None, out_file=None):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    _mean = f1_scores.mean(axis=0)
    _std = f1_scores.std(axis=0)

    ax.plot(anomaly_ratios, _mean, label='relative f1 score', color='black', marker='o', linewidth=2, zorder=10)
    ax.fill_between(anomaly_ratios, (_mean - _std), (_mean + _std), color='black', alpha=.1, zorder=9)

    if point_wise_ar:
        point_wise_ar_y = np.interp(np.array([point_wise_ar]), xp=anomaly_ratios, fp=_mean)[0]
        ax.scatter(point_wise_ar, point_wise_ar_y, color='#0400ff', label='point-wise', marker='o', zorder=13, s=128)

    if sample_wise_ar:
        sample_wise_ar_y = np.interp(np.array([sample_wise_ar]), xp=anomaly_ratios, fp=_mean)[0]
        ax.scatter(sample_wise_ar, sample_wise_ar_y, color='#ff0000', label='sample-wise', marker='o', zorder=13, s=128)

    ax.set_ylabel('relative f1 score [%]')
    ax.set_xlabel('hyper-parameter anomaly ratio [%]')

    ax.legend()
    plt.tight_layout()
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)
    plt.close(fig)


def ts_plot(x, y_true=None, y_score=None, y_threshold=None, reconstructed=None, y_score_min_max=None, out_file=None, anomaly_color='#ff7d83'):
    x = np.array(x)

    if y_score is None:
        fig, ts_ax = plt.subplots(1, 1, figsize=(15, 10))
    else:
        fig, (ts_ax, score_ax) = plt.subplots(2, 1, figsize=(15, 20))
        if y_score_min_max is not None:
            score_ax.set_ylim(*y_score_min_max)

        y_score = np.array(y_score)
        score_ax.plot(range(len(y_score)), y_score, color='black', label='score', zorder=10, marker='o')
        if y_threshold is not None and not math.isnan(y_threshold):
            score_ax.hlines(y=y_threshold, xmin=0, xmax=len(y_score), colors='black', label='threshold')
            y_ticks = list(score_ax.get_yticks())
            y_ticks.append(y_threshold)
            y_ticks.sort()
            score_ax.set_yticks(y_ticks)

    ts_ax.plot(range(len(x)), x, color='black', label='gt', zorder=10, marker='o')

    if reconstructed is not None:
        ts_ax.plot(range(len(reconstructed)), reconstructed, color='#0008ff', label='rec', zorder=11, marker='o')

    if y_true is not None:
        y_true = np.array(y_true)
        for xmin, xmax in _consecutive(np.nonzero(y_true)[0]):
            xmin = max(0, xmin - 1)
            xmax = xmax + 1
            ts_ax.axvspan(xmin=xmin, xmax=xmax, facecolor=anomaly_color, label='anomaly')
            if 'score_ax' in locals():
                score_ax.axvspan(xmin=xmin, xmax=xmax, facecolor=anomaly_color, label='anomaly', zorder=0)

    ts_ax.set_xlabel('timestep')
    ts_ax.set_ylabel('voltage')
    ts_ax.legend()
    if 'score_ax' in locals():
        score_ax.legend()

    plt.subplots_adjust(left=0.15, bottom=0.14, top=0.975, right=0.99)
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)
    plt.close(fig)


def loss_plot(train_loss, val_loss=None, out_file=None):
    train_loss = np.array(train_loss)
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    train_colors = ['#001aff', '#0392ff']
    for i in range(train_loss.shape[1]):
        ax.plot(range(1, len(train_loss) + 1), train_loss[:, i], label=f'train loss {i + 1}', color=train_colors[i % len(train_colors)], linewidth=2)

    val_colors = ['#ff0303', '#fc00f0']
    if val_loss is not None:
        val_loss = np.array(val_loss)
        for i in range(train_loss.shape[1]):
            ax.plot(range(1, len(val_loss) + 1), val_loss[:, i], label=f'val loss {i + 1}', color=val_colors[i % len(val_colors)], linewidth=2)

    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')

    ax.legend()
    plt.tight_layout()
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)
    plt.close(fig)


def precision_recall_plot(precision, recall, out_file=None):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    ax.plot(recall, precision, color='black', zorder=10, linewidth=2)

    ax.set_ylabel('precision')
    ax.set_xlabel('recall')

    plt.tight_layout()
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)
    plt.close(fig)


def f1_score_plot(threshold, f1_score, metrics=None, out_file=None):
    if metrics is None:
        metrics = {}

    metric_map = {
        'max-score': ('black', 'o', 255, 11),
        'sample-wise': ('#ff0000', 'o', 128, 13),
        'point-wise': ('#0400ff', 'o', 128, 13),
    }

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    ax.set_ylim(0, 1)
    ax.plot(threshold, f1_score, color='black', zorder=10, linewidth=2)

    for key, val in metrics.items():
        if key in metric_map:
            color, marker, size, zorder = metric_map.get(key, ('black', 'x', 128, 13))
            ax.scatter(val['threshold'], val['f1_score'], s=size, marker=marker, color=color, zorder=zorder, label=key)

    ax.set_xlabel('threshold')
    ax.set_ylabel('f1-score')
    ax.legend()

    plt.tight_layout()
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)
    plt.close(fig)
