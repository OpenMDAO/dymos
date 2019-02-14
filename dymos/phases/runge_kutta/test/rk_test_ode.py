from __future__ import print_function, division, absolute_import

import numpy as np
from openmdao.api import ExplicitComponent
from dymos import declare_time, declare_state

@declare_time(targets=['t'], units='s')
@declare_state('y', targets=['y'], rate_source='ydot', units='m')
class TestODE(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        self.add_input('t', val=np.ones(self.options['num_nodes']), units='s')
        self.add_input('y', val=np.ones(self.options['num_nodes']), units='m')
        self.add_output('ydot', val=np.ones(self.options['num_nodes']), units='m/s')

        ar = np.arange(self.options['num_nodes'])
        self.declare_partials(of='ydot', wrt='t', rows=ar, cols=ar)
        self.declare_partials(of='ydot', wrt='y', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        t = inputs['t']
        y = inputs['y']
        outputs['ydot'] = y - t ** 2 + 1

    def compute_partials(self, inputs, partials):
        partials['ydot', 't'] = -2 * inputs['t']