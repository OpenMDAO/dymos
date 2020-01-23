import numpy as np
import openmdao.api as om
import dymos as dm


# @dm.declare_time(targets=['t'], units='s')
# @dm.declare_state('y', targets=['y'], rate_source='ydot', units='m')
class TestODE(om.ExplicitComponent):

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


def _test_ode_solution(y, t):
    return t**2 + 2*t + 1 - 0.5 * np.exp(t)
