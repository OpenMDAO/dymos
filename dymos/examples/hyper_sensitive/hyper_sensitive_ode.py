from __future__ import print_function, division, absolute_import

import numpy as np
import openmdao.api as om

class HyperSensitiveODE(om.ExplicitComponent):

    states = {'x': {'rate_source': 'xdot',
                    'units': 'm'}}

    parameters = {'u': {'targets': 'u'}}

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # inputs
        self.add_input('x', val=np.zeros(nn), desc='state')

        self.add_input('u', val=np.zeros(nn), desc='control')

        self.add_output('x_dot', val=np.zeros(nn), desc='state rate')

        # Setup partials
        self.declare_partials(of='x_dot', wrt='x', rows=np.arange(nn), cols=np.arange(nn))

        self.declare_partials(of='x_dot', wrt='u', rows=np.arange(nn), cols=np.arange(nn))

    def compute(self, inputs, outputs):
        x = inputs['x']
        u = inputs['u']

        outputs['x_dot'] = -x + u

    def compute_partials(self, inputs, jacobian):
        x = inputs['x']
        u = inputs['u']

        jacobian['x_dot', 'x'] = -1
        jacobian['x_dot', 'u'] = 1

