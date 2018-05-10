from __future__ import print_function, division

import numpy as np

from openmdao.api import ExplicitComponent


class LinearTangentGuidanceComp(ExplicitComponent):
    """ Compute pitch angle from static controls governing linear expression for
        pitch angle tangent as function of time.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('a_ctrl',
                       val=np.zeros(nn),
                       desc='linear tangent slope',
                       units='1/s')

        self.add_input('b_ctrl',
                       val=np.zeros(nn),
                       desc='tangent of theta at t=0',
                       units=None)

        self.add_input('time',
                       val=np.zeros(nn),
                       desc='time',
                       units='s')

        self.add_output('theta',
                        val=np.zeros(nn),
                        desc='pitch angle',
                        units='rad')

        # Setup partials
        arange = np.arange(self.options['num_nodes'])

        self.declare_partials(of='theta', wrt='a_ctrl', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='theta', wrt='b_ctrl', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='theta', wrt='time', rows=arange, cols=arange, val=1.0)

    def compute(self, inputs, outputs):
        a = inputs['a_ctrl']
        b = inputs['b_ctrl']
        t = inputs['time']
        outputs['theta'] = np.arctan(a * t + b)

    def compute_partials(self, inputs, jacobian):
        a = inputs['a_ctrl']
        b = inputs['b_ctrl']
        t = inputs['time']

        x = a * t + b
        denom = x**2 + 1.0

        jacobian['theta', 'a_ctrl'] = t / denom
        jacobian['theta', 'b_ctrl'] = 1.0 / denom
        jacobian['theta', 'time'] = a / denom
