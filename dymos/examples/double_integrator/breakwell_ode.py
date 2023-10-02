import openmdao.api as om
import numpy as np


class BreakwellODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('u', val=np.zeros(nn))
        self.add_output('J_dot', val=np.zeros(nn))

        ar = np.arange(nn)
        self.declare_partials(of='J_dot', wrt='u', rows=ar, cols=ar)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        u = inputs['u']
        outputs['J_dot'] = 0.5*u**2

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        u = inputs['u']
        partials['J_dot', 'u'] = u
