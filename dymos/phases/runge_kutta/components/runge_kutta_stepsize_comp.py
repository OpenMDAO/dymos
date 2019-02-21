from __future__ import print_function, division, absolute_import

from collections import Sequence
from six import string_types

import numpy as np

from openmdao.api import ExplicitComponent


class RungeKuttaStepsizeComp(ExplicitComponent):
    """
    Given the duration of the phase and the segment relative lengths, compute the duration of
    each segment (the step size) for each segment (step).
    """

    def initialize(self):
        self.options.declare('num_segments', types=int,
                             desc='The number of segments (timesteps) in the phase')

        self.options.declare('seg_rel_lengths', types=(np.ndarray, Sequence),
                             desc='The relative lengths of the segments in the phase.')

        self.options.declare('time_units', default=None, allow_none=True, types=string_types,
                             desc='Units of the integration variable')

        self.options.declare('direction', default='forward', values=('forward', 'backward'),
                             desc='Whether the numerical propagation occurs forwards or backwards '
                                  'in time.  This poses restrictions on whether states can have '
                                  'fixed initial/final values.')

    def setup(self):

        self._var_names = {}

        num_seg = self.options['num_segments']
        seg_rel_lengths = self.options['seg_rel_lengths']

        self._norm_seg_rel_lengths = seg_rel_lengths / np.sum(seg_rel_lengths)

        self._sign = 1 if self.options['direction'] == 'forward' else -1

        self.add_input('t_duration', val=1.0, units=self.options['time_units'])

        self.add_output('h', val=np.ones(num_seg), units=self.options['time_units'],
                        desc='step size for current Runge-Kutta segment.')

        self.declare_partials(of='h', wrt='t_duration', rows=np.arange(num_seg, dtype=int),
                              cols=np.zeros(num_seg, dtype=int),
                              val=self._sign * self._norm_seg_rel_lengths)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs['h'] = self._sign * inputs['t_duration'] * self._norm_seg_rel_lengths
