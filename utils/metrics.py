import numpy as np


def sample_wise_ar(a, batch_first=True):
    a[a > 1] = 1
    if not batch_first:
        a = a.transpose(1, 0, 2)

    sample_is_anomalous = np.any(a != 0, axis=(1, 2))
    return np.mean(sample_is_anomalous)


def point_wise_ar(a):
    a[a > 1] = 1
    return np.mean(a)
