import numpy as np
import math
from dataclasses import dataclass
from synthetic_data.util import array_wrapper_function
from synthetic_data.base import Interval, IntervalType


class QuadraticFunction:

    def __init__(self, a, z, c):
        self.a = a
        self.z = z
        self.c = c

    def __call__(self, x):
        return self.a * (x - self.z) ** 2 + self.c

    def __str__(self):
        return f'f(x) = {self.a} * (x - {self.z}) + {self.c}'


class LinearFunction:

    def __init__(self, k, d):
        self.k = k
        self.d = d

    def __call__(self, x):
        return self.k * x + self.d

    def __str__(self):
        return f'f(x) = {self.k} * x + {self.d}'


def find_linear_anomaly_function(f, anomaly_range):
    x1 = anomaly_range.start
    y1 = f(x1)

    x2 = anomaly_range.stop
    y2 = f(x2)

    k = (y1 - y2) / (x1 - x2)
    d = y2 - k * x2

    return LinearFunction(k, d)


def find_linear_drop_anomaly_function(f, c, anomaly_range):
    x1 = anomaly_range.start
    y1 = f(x1)

    x2 = anomaly_range.stop
    y2 = f(x2)

    x_mid = x1 + (x2 - x1) / 2
    y_mid = c

    k1 = (y1 - y_mid) / (x1 - x_mid)
    d1 = y_mid - k1 * x_mid

    k2 = (y_mid - y2) / (x_mid - x2)
    d2 = y2 - k2 * x2

    lin1 = LinearFunction(k1, d1)
    lin2 = LinearFunction(k2, d2)

    @array_wrapper_function
    def piecewise(x):
        if x <= x_mid:
            return lin1(x)
        else:
            return lin2(x)

    return piecewise


def solve_quadratic_equation(a, b, c):
    if math.isclose(a, 0):
        raise ValueError('a must not be zero')

    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        raise ValueError('discriminant must not be negative')

    x1 = (-b + math.sqrt(discriminant)) / (2 * a)
    x2 = (-b - math.sqrt(discriminant)) / (2 * a)
    return x1, x2


def find_quadratic_anomaly_function(f, c, anomaly_range):
    x1 = anomaly_range.start
    y1 = f(x1)

    x2 = anomaly_range.stop
    y2 = f(x2)

    z1, z2 = solve_quadratic_equation(y1 - y2 + 1e-5, -2 * c * x1 + 2 * c * x2 + 2 * x1 * y2 - 2 * x2 * y1, c * x1 ** 2 - c * x2 ** 2 - x1 ** 2 * y2 + x2 ** 2 * y1)

    if math.isclose(x2 - z1, 0) or math.isclose(x2 - z2, 0):
        raise ValueError('div by zero')

    a1 = (y2 - c) / (x2 - z1) ** 2
    a2 = (y2 - c) / (x2 - z2) ** 2

    return QuadraticFunction(a1, z1, c), QuadraticFunction(a2, z2, c)


def find_constant_anomaly_function(c):
    return lambda _: c


@dataclass
class Anomaly:

    id: int
    label: int
    duration_mean: float
    duration_std: float
    noise_mean: float
    noise_std: float
    segments: [str]
    anomaly_center: float
    weight: float

    @staticmethod
    def parse(**kwargs):
        if kwargs['label'] <= 0:
            raise ValueError('label must be gt 0')

        _type = kwargs['type']
        del kwargs['type']

        if kwargs['anomaly_center'] is not None and kwargs['segments'] is not None:
            raise ValueError('both anomaly_center and segments definition are not allowed')

        if _type == 'identity':
            return IdentityAnomaly(**kwargs)
        elif _type == 'linear':
            return LinearAnomaly(**kwargs)
        elif _type == 'point':
            return PointAnomaly(**kwargs)
        elif _type == 'quadratic':
            return QuadraticAnomaly(**kwargs)
        else:
            raise ValueError(f'invalid type anomaly type "{_type}"')

    def get_random_segment(self):
        if self.segments is None:
            return 'all'
        else:
            return self.segments[np.random.randint(0, len(self.segments))]

    def get_random_duration(self):
        return np.random.normal(loc=self.duration_mean, scale=self.duration_std)

    def get_random_anomaly_interval(self, f):
        segment = self.get_random_segment()
        segment_interval = f.get_segment_interval(segment)

        anomaly_duration_half = self.get_random_duration() / 2

        if self.anomaly_center is None:
            anomaly_center = np.random.uniform(segment_interval.start + anomaly_duration_half,
                                               segment_interval.stop - anomaly_duration_half)
        else:
            anomaly_center = self.anomaly_center

        anomaly_start = max(segment_interval.start, anomaly_center - anomaly_duration_half)
        anomaly_stop = min(segment_interval.stop, anomaly_center + anomaly_duration_half)

        return Interval(anomaly_start, anomaly_stop, IntervalType.CLOSED)

    def get_anomaly_function(self, f, anomaly_interval):
        raise NotImplementedError()

    def apply(self, f, x, y, gt=None):
        anomaly_interval = self.get_random_anomaly_interval(f)
        anomaly_function = self.get_anomaly_function(f, anomaly_interval)
        anomaly_indices = anomaly_interval.included_indices(x)

        noise = np.random.normal(loc=self.noise_mean, scale=self.noise_std, size=len(anomaly_indices))
        y[anomaly_indices] = anomaly_function(x[anomaly_indices]) + noise

        if gt is None:
            gt = np.zeros(shape=y.shape)

        gt[anomaly_indices] = self.label

        return gt


@dataclass
class IdentityAnomaly(Anomaly):

    def get_anomaly_function(self, f, anomaly_interval):
        return f


@dataclass
class LinearAnomaly(Anomaly):

    def get_anomaly_function(self, f, anomaly_interval):
        return find_linear_anomaly_function(f, anomaly_interval)


@dataclass
class PointAnomaly(Anomaly):

    deviation_mean: float
    deviation_std: float

    def get_anomaly_function(self, f, anomaly_interval):
        base = min(f(anomaly_interval.start), f(anomaly_interval.stop)) + abs(f(anomaly_interval.start) - f(anomaly_interval.stop)) / 2
        deviation = np.random.normal(loc=self.deviation_mean, scale=self.deviation_std)
        return find_linear_drop_anomaly_function(f, base + deviation, anomaly_interval)


@dataclass
class QuadraticAnomaly(Anomaly):

    deviation_mean: float
    deviation_std: float

    def get_anomaly_function(self, f, anomaly_interval):
        deviation = np.random.normal(loc=self.deviation_mean, scale=self.deviation_std)

        if deviation < 0:
            base = min(f(anomaly_interval.start), f(anomaly_interval.stop))
        else:
            base = max(f(anomaly_interval.start), f(anomaly_interval.stop))

        f1, f2 = find_quadratic_anomaly_function(f, base + deviation, anomaly_interval)

        if abs(f1.a) >= abs(f2.a):
            return f1
        else:
            return f2


class AnomalySet:

    @staticmethod
    def parse(anomalies):
        return AnomalySet([Anomaly.parse(**{**{'id': idx}, **a}) for idx, a in enumerate(anomalies)])

    def __init__(self, anomalies: [Anomaly]):
        for a in anomalies:
            if not isinstance(a, Anomaly):
                raise TypeError('a must be of type anomaly')

        self.anomalies = anomalies

    @property
    def weight_sum(self):
        return sum(map(lambda a: a.weight, self.anomalies))

    @property
    def probabilities(self):
        weight_sum = self.weight_sum
        if math.isclose(0, weight_sum):
            return [1 / len(self.anomalies)] * len(self.anomalies)
        else:
            return list(map(lambda a: a.weight / weight_sum, self.anomalies))

    def get_random_anomaly(self):
        return self.anomalies[np.random.choice(len(self.anomalies), p=self.probabilities)]
