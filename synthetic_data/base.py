import math
import numpy as np
from enum import Enum
from synthetic_data.util import array_wrapper_method


class IntervalType(Enum):
    OPEN = 1
    CLOSED = 2
    LEFT_OPEN = 3
    RIGHT_OPEN = 4


class Interval:

    def __init__(self, start, stop, _type):
        if not isinstance(start, (int, float)) or not isinstance(stop, (int, float)):
            raise ValueError('invalid data type')

        if not isinstance(_type, IntervalType):
            raise ValueError('invalid data type')

        self.start = start
        self.stop = stop
        self.type = _type

    @array_wrapper_method
    def includes(self, x):
        if self.type == IntervalType.OPEN:
            return self.start < x < self.stop
        elif self.type == IntervalType.CLOSED:
            return self.start <= x <= self.stop
        elif self.type == IntervalType.LEFT_OPEN:
            return self.start < x <= self.stop
        elif self.type == IntervalType.RIGHT_OPEN:
            return self.start <= x < self.stop
        else:
            raise ValueError('invalid interval type')

    def included_indices(self, x):
        return np.nonzero(self.includes(x))[0]

    def __getitem__(self, item):
        if item == 0:
            return self.start
        elif item == 1:
            return self.stop

        raise IndexError(f'invalid index {item}')

    def __str__(self):
        if self.type == IntervalType.OPEN:
            return f'({self.start}, {self.stop})'
        elif self.type == IntervalType.CLOSED:
            return f'[{self.start}, {self.stop}]'
        elif self.type == IntervalType.LEFT_OPEN:
            return f'({self.start}, {self.stop}]'
        elif self.type == IntervalType.RIGHT_OPEN:
            return f'[{self.start}, {self.stop})'

        raise ValueError('invalid interval type')


class BaseFunction:

    def __init__(self, *args, **kwargs):
        pass

    @property
    def domain(self):
        return Interval(float('-inf'), float('inf'), IntervalType.OPEN)

    def get_segment_interval(self, name: str):
        if name is None or name == 'all':
            return self.domain

        raise ValueError(f'invalid segment {name}')

    @array_wrapper_method
    def __call__(self, x):
        raise NotImplementedError()


class SineLikeFunction(BaseFunction):

    SEGMENTS = {
        'q1': (0, math.pi / 2),
        'q2': (math.pi / 2, math.pi),
        'q3': (math.pi, math.pi * 3 / 2),
        'q4': (math.pi * 3 / 2, 2 * math.pi),
    }

    def __init__(self, phase_shift_mean=0., phase_shift_std=0., freq_mean=1., freq_std=0., amplitude_mean=1., amplitude_std=0., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_shift = np.random.normal(phase_shift_mean, phase_shift_std)
        self.freq = np.random.normal(freq_mean, freq_std)
        self.amplitude = np.random.normal(amplitude_mean, amplitude_std)

    @property
    def domain(self):
        return Interval(0, 2 * math.pi, IntervalType.CLOSED)

    def _transform(self, a):
        return ((a - self.phase_shift) / self.freq) % self.domain[1]

    def get_segment_interval(self, name: str):
        if name in self.SEGMENTS:
            segment = self.SEGMENTS[name]
            start = self._transform(segment[0])
            stop = self._transform(segment[1])

            if start > stop:
                stop = self.domain[1] + stop

            return Interval(start, stop, IntervalType.RIGHT_OPEN)

        return super().get_segment_interval(name)


class SineFunction(SineLikeFunction):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @array_wrapper_method
    def __call__(self, x):
        return self.amplitude * math.sin(self.freq * x + self.phase_shift)


class CosineFunction(SineLikeFunction):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @array_wrapper_method
    def __call__(self, x):
        return self.amplitude * math.cos(self.freq * x + self.phase_shift)
