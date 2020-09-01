import numpy as np
import openmdao.api as om


class HyperSensitiveODE(om.ExplicitComponent):
    states = {'x': {'rate_source': 'x_dot'},
              'xL': {'rate_source': 'L'}}

    parameters = {'u': {'targets': 'u'}}

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # inputs
        self.add_input('x', val=np.zeros(nn), desc='state')
        self.add_input('xL', val=np.zeros(nn), desc='cost_state')

        self.add_input('u', val=np.zeros(nn), desc='control')

        self.add_output('x_dot', val=np.zeros(nn), desc='state rate', units='1/s')
        self.add_output('L', val=np.zeros(nn), desc='Lagrangian', units='1/s')

        # Setup partials
        self.declare_partials(of='x_dot', wrt='x', rows=np.arange(nn), cols=np.arange(nn), val=-1)
        self.declare_partials(of='x_dot', wrt='u', rows=np.arange(nn), cols=np.arange(nn), val=1)

        self.declare_partials(of='L', wrt='x', rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials(of='L', wrt='u', rows=np.arange(nn), cols=np.arange(nn))

    def compute(self, inputs, outputs):
        x = inputs['x']
        u = inputs['u']

        outputs['x_dot'] = -x + u
        outputs['L'] = (x ** 2 + u ** 2) / 2

    def compute_partials(self, inputs, jacobian):
        x = inputs['x']
        u = inputs['u']

        jacobian['L', 'x'] = x
        jacobian['L', 'u'] = u
