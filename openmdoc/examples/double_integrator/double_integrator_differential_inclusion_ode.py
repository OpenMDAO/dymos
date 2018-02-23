from __future__ import print_function, division, absolute_import

import numpy as np
from openmdao.api import ExplicitComponent
from openmdoc import declare_time, declare_state, declare_parameter


@declare_time(units='s')
@declare_state('x', rate_source='xdot', units='m')
@declare_parameter('v', targets='v', units='m/s')
class DoubleIntegratorDifferentialInclusionODE(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

    def setup(self):
        nn = self.metadata['num_nodes']

        # Inputs
        self.add_input('v', val=np.ones(nn), desc='velocity control', units='m/s')
        self.add_output('xdot', val=np.ones(nn), desc='position time-derivative', units='m/s')

        # Setup partials
        arange = np.arange(self.metadata['num_nodes'])
        self.declare_partials(of='xdot', wrt='v', rows=arange, cols=arange, val=1.0)

    def compute(self, inputs, outputs):
        outputs['xdot'] = inputs['v']

