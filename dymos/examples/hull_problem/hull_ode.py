import numpy as np
import openmdao.api as om


class HullProblemODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('u', val=np.zeros(nn), desc='control')

        self.add_output('x_dot', val=np.zeros(nn), desc='state rate', units='1/s')
        self.add_output('L', val=np.zeros(nn), desc='Lagrangian', units='1/s')

        # Setup partials
        self.declare_partials(of='x_dot', wrt='u', rows=np.arange(nn), cols=np.arange(nn), val=1)
        self.declare_partials(of='L', wrt='u', rows=np.arange(nn), cols=np.arange(nn))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        u = inputs['u']

        outputs['x_dot'] = u
        outputs['L'] = 0.5 * u ** 2

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        u = inputs['u']
        partials['L', 'u'] = u
