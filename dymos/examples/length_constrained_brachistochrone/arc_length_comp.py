from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent


class ArcLengthComp(ExplicitComponent):

    def initialize(self):

        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('x', val=np.ones(nn), units='m', desc='x at points along the trajectory')
        self.add_input('theta', val=np.ones(nn), units='rad',
                       desc='wire angle with vertical along the trajectory')

        self.add_output('S', val=1.0, units='m', desc='arclength of wire')

        self.declare_partials(of='S', wrt='*', method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        x = inputs['x']
        theta = inputs['theta']

        dy_dx = -1.0 / np.tan(theta)
        dx = np.diff(x)
        f = np.sqrt(1 + dy_dx**2)

        # trapezoidal rule
        fxm1 = f[:-1]
        fx = f[1:]
        outputs['S'] = 0.5 * np.dot(fxm1 + fx, dx)
