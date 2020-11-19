from collections.abc import Sequence
import numpy as np

import openmdao.api as om
from ....options import options as dymos_options


class RungeKuttaStepsizeComp(om.ExplicitComponent):
    """
    Given the duration of the phase and the segment relative lengths, compute the duration of
    each segment (the step size) for each segment (step).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):
        self.options.declare('num_segments', types=int,
                             desc='The number of segments (timesteps) in the phase')

        self.options.declare('seg_rel_lengths', types=(np.ndarray, Sequence),
                             desc='The relative lengths of the segments in the phase.')

        self.options.declare('time_units', default=None, allow_none=True, types=str,
                             desc='Units of the integration variable')

    def configure_io(self):
        """
        I/O creation is delayed until configure so that we can determine the shape and units for
        the states.
        """
        self._var_names = {}

        num_seg = self.options['num_segments']
        seg_rel_lengths = self.options['seg_rel_lengths']

        self._norm_seg_rel_lengths = seg_rel_lengths / np.sum(seg_rel_lengths)

        self.add_input('t_duration', val=1.0, units=self.options['time_units'])

        self.add_output('h', val=np.ones(num_seg), units=self.options['time_units'],
                        desc='step size for current Runge-Kutta segment.')

        self.declare_partials(of='h', wrt='t_duration', rows=np.arange(num_seg, dtype=int),
                              cols=np.zeros(num_seg, dtype=int),
                              val=self._norm_seg_rel_lengths)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs['h'] = inputs['t_duration'] * self._norm_seg_rel_lengths
