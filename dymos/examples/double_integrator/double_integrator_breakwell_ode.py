import openmdao.api as om
import numpy as np


class DoubleIntegratorBreakwellODE(om.ExplicitComponent):
    """
    The double integrator is a special case where the state rates are all set to other states
    or parameters.  Unlike in the standard version of the problem, here we need compute a dummy
    state in order to obtain the objective function.
    Note we still have to declare the num_nodes option in initialize so that Dymos can instantiate
    the ODE.

    Also note that neither time, states, nor parameters have targets, since there are no inputs
    in the ODE system.
    """

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
