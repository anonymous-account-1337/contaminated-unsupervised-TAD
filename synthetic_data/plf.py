import numpy as np
from dataclasses import dataclass
from synthetic_data.util import array_wrapper_method
from synthetic_data.base import Interval, IntervalType, BaseFunction


class LinearFunction:

    def __init__(self, k, d):
        self.k = k
        self.d = d

    @staticmethod
    def solve(k, x, y):
        d = y - k * x
        return LinearFunction(k, d)

    def __call__(self, x):
        return self.k * x + self.d

    def __str__(self):
        return f'f(x) = {self.k} * x + {self.d}'


@dataclass
class LinearFunctionSegmentDefinition:

    slope: float
    duration: float
    name: str = None


@dataclass
class StochasticLinearFunctionSegmentDefinition:

    slope_mean: float
    duration_mean: float
    name: str = None
    slope_std: float = 0
    duration_std: float = 0

    def draw_segment_definition(self) -> LinearFunctionSegmentDefinition:
        return LinearFunctionSegmentDefinition(slope=np.random.normal(loc=self.slope_mean, scale=self.slope_std), duration=np.random.normal(loc=self.duration_mean, scale=self.duration_std), name=self.name)


class LinearFunctionSegment(Interval):

    def __init__(self, function: LinearFunction, start, stop, i_type: IntervalType, name: str = None):
        super().__init__(start, stop, i_type)
        self.function = function
        self.name = name

    @classmethod
    def from_stochastic_linear_function_segment_definitions(cls, definitions, start_x = 0, start_y = 0):
        return cls.from_linear_function_segment_definitions([definition.draw_segment_definition() for definition in definitions], start_x=start_x, start_y=start_y)

    @staticmethod
    def from_linear_function_segment_definitions(definitions, start_x = 0, start_y = 0):
        x = start_x
        y = start_y

        linear_function_segments = []
        for idx, definition in enumerate(definitions):
            f = LinearFunction.solve(definition.slope, x, y)
            old_x = x
            x += definition.duration
            y = f(x)

            if len(definitions) == 1 or idx == len(definitions) - 1:
                interval_type = IntervalType.CLOSED
            else:
                interval_type = IntervalType.RIGHT_OPEN

            linear_function_segments.append(LinearFunctionSegment(f, old_x, x, interval_type, definition.name))

        return linear_function_segments


class LinearFunctionSegmentNotFoundError(RuntimeError):
    pass


class PiecewiseLinearFunction(BaseFunction):

    def __init__(self, segments: [LinearFunctionSegment], *args, **kwargs):
        super().__init__(*args, **kwargs)

        if len(segments) == 0:
            raise ValueError('at least one segment is required')

        self.segments = segments

    def find_segment_by_name(self, name: str):
        for segment in self.segments:
            if segment.name is None:
                continue
            elif segment.name == name:
                return segment

        raise LinearFunctionSegmentNotFoundError()

    def get_segment_interval(self, name: str):
        try:
            return self.find_segment_by_name(name)
        except LinearFunctionSegmentNotFoundError:
            return super().get_segment_interval(name)

    @property
    def domain(self):
        return Interval(self.segments[0].start, self.segments[-1].stop, IntervalType.CLOSED)

    @array_wrapper_method
    def __call__(self, x):
        for segment in self.segments:
            if segment.includes(x):
                return segment.function(x)

        raise ValueError(f'x {x:.1f} out of domain {self.domain}')
