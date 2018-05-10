from __future__ import print_function, division, absolute_import

import numpy as np
from openmdao.api import ExplicitComponent
from dymos import declare_time, declare_state, declare_parameter


@declare_time(units='s')
@declare_state('x', rate_source='xdot', units='m')
@declare_state('v', rate_source='vdot', targets='v', units='m/s')
@declare_parameter('u', targets='u', units='m/s**2')
class DoubleIntegratorODE(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('v', val=np.ones(nn), desc='velocity', units='m/s')
        self.add_input('u', val=np.ones(nn), desc='acceleration control', units='m/s/s')
        self.add_output('vdot', val=np.ones(nn), desc='velocity time-derivative', units='m/s/s')
        self.add_output('xdot', val=np.ones(nn), desc='position time-derivative', units='m/s')

        # Setup partials
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='vdot', wrt='u', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='xdot', wrt='v', rows=arange, cols=arange, val=1.0)

    def compute(self, inputs, outputs):
        v = inputs['v']
        u = inputs['u']

        outputs['xdot'] = v
        outputs['vdot'] = u
