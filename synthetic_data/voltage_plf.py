from synthetic_data.plf import PiecewiseLinearFunction, StochasticLinearFunctionSegmentDefinition, LinearFunction, LinearFunctionSegment


def _fix_segments(segments: [LinearFunctionSegment]):
    """make sure that deceleration of move-back-stage stops at 0 [V]"""

    dec_function = segments[-2].function
    stop_function = segments[-1].function

    new_threshold = -dec_function.d / dec_function.k
    segments[-2].stop = new_threshold
    segments[-1].start = new_threshold
    segments[-1].function = LinearFunction.solve(stop_function.k, new_threshold, dec_function(new_threshold))


class VoltageFunction(PiecewiseLinearFunction):

    VARIABLE_SEGMENTS = [
        StochasticLinearFunctionSegmentDefinition(slope_mean=0.02, duration_mean=75, slope_std=0.01, duration_std=10, name='start'),
        StochasticLinearFunctionSegmentDefinition(slope_mean=1.73, duration_mean=65, slope_std=0.5, duration_std=5, name='pre_acceleration'),
        StochasticLinearFunctionSegmentDefinition(slope_mean=8.76, duration_mean=80, slope_std=0.25, duration_std=5, name='acceleration'),
        StochasticLinearFunctionSegmentDefinition(slope_mean=0, duration_mean=580, duration_std=35, name='plateau'),
        StochasticLinearFunctionSegmentDefinition(slope_mean=-11, duration_mean=152, slope_std=0.05, duration_std=5, name='deceleration'),
        StochasticLinearFunctionSegmentDefinition(slope_mean=1.6, duration_mean=48, slope_std=0.65, duration_std=5, name='deceleration_stop'),
        StochasticLinearFunctionSegmentDefinition(slope_mean=0.02, duration_mean=530, slope_std=0.01, duration_std=20, name='move_back'),
        StochasticLinearFunctionSegmentDefinition(slope_mean=10.4, duration_mean=75, slope_std=0.25, duration_std=5, name='move_back_deceleration'),
        StochasticLinearFunctionSegmentDefinition(slope_mean=0.02, duration_mean=75, slope_std=0.01, duration_std=10, name='stop'),
    ]

    def __init__(self, *args, **kwargs):
        segments = LinearFunctionSegment.from_stochastic_linear_function_segment_definitions(self.VARIABLE_SEGMENTS)
        _fix_segments(segments)
        super().__init__(segments, *args, **kwargs)
