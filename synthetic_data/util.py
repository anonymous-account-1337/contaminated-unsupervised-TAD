import numpy as np
from functools import wraps


def array_wrapper_function(f):

    @wraps(f)
    def new_f(x):
        if isinstance(x, np.ndarray):
            return np.array([new_f(e) for e in x])
        elif isinstance(x, list):
            return [new_f(e) for e in x]
        elif isinstance(x, (float, int, np.int32, np.int64, np.float32, np.float64)):
            return f(x)
        else:
            raise TypeError(f'{type(x)} not supported')

    return new_f


def array_wrapper_method(f):

    @wraps(f)
    def new_f(self, x):
        if isinstance(x, np.ndarray):
            return np.array([new_f(self, e) for e in x])
        elif isinstance(x, list):
            return [new_f(self, e) for e in x]
        elif isinstance(x, (float, int, np.int32, np.int64, np.float32, np.float64)):
            return f(self, x)
        else:
            raise TypeError(f'{type(x)} not supported')

    return new_f


def shift(a, offset):
    new = np.zeros(shape=a.shape)
    if offset == 0:
        return a
    elif offset > 0:
        new[offset:] = a[:-offset]
    else:
        new[:offset] = a[-offset:]

    return new


def pad_or_cut(a, num_dp):
    if len(a) < num_dp:
        pad_left = (num_dp - len(a)) // 2
        pad_right = num_dp - pad_left - len(a)
        a = np.pad(a, pad_width=(pad_left, pad_right), mode='constant', constant_values=0)
    elif len(a) > num_dp:
        a = a[:num_dp]

    return a


def cosine_similarity(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return np.dot(a, b) / (np.linalg.norm(a, ord=2) * np.linalg.norm(b, ord=2))


def euclidean_distance(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return np.linalg.norm(a - b, ord=2)


def contamination(a, b, batch_first=False):
    if not batch_first:
        a, b = a.transpose(1, 0, 2), b.transpose(1, 0, 2)

    a = a.reshape(-1, a.shape[1] * a.shape[2])
    b = b.reshape(-1, b.shape[1] * b.shape[2])

    distance = np.linalg.norm(a - b, ord=2, axis=1)
    return np.mean(distance), np.var(distance)


def point_wise_anomaly_ratio(gt):
    res = dict(zip(*np.unique(gt, return_counts=True)))

    normal_points = res[0] if 0 in res else 0
    del res[0]
    anomalous_points = 0
    for k, v in res.items():
        anomalous_points += v
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
