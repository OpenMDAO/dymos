from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent

from dymos.phases.options import TimeOptionsDictionary


class StepSizeComp(ExplicitComponent):
    """
    Computes the values of the time to pass to the ODE for a given stage
    """
    def initialize(self):
        self.options.declare('time_options', types=TimeOptionsDictionary)
        self.options.declare('num_steps', types=(int,),
                             desc='Number of steps to take within the segment.')

    def setup(self):
        num_steps = self.options['num_steps']
        time_options = self.options['time_options']

        self.add_input(name='seg_t0_tf',
                       val=np.array([0.0, 1.0]),
                       desc='initial time in the segment',
                       units=time_options['units'])

        self.add_output(name='h',
                        val=1.0,
                        desc='size of the steps within the segment',
                        units=time_options['units'])

        self.declare_partials(of='h',
                              wrt='seg_t0_tf',
                              val=np.array([1.0, -1.0]) / num_steps)

    def compute(self, inputs, outputs):
        num_steps = self.options['num_steps']
        seg_t0_tf = inputs['seg_t0_tf']
        outputs['h'] = (seg_t0_tf[1] - seg_t0_tf[0]) / num_steps