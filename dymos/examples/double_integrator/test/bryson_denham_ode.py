import openmdao.api as om
import numpy as np

class BrysonDenhamODE(om.ExplicitComponent):
    """
    The double integrator is a special case where the state rates are all set to other states
    or parameters.  Since we aren't computing any other outputs, the ODE doesn't actually
    need to compute anything.  OpenMDAO will warn us that the component has no outputs, but
    Dymos will solve the problem just fine.

    Note we still have to declare the num_nodes option in initialize so that Dymos can instantiate
    the ODE.

    Also note that neither time, states, nor parameters have targets, since there are no inputs
    in the ODE system.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('u', shape=(nn,), units='m/s**2')

        self.add_output('usq', shape=(nn,), units='m**2/s**4')

        ar = np.arange(nn, dtype=int)

        self.declare_partials(of='usq', wrt='u', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['usq'] = 0.5 * inputs['u'] ** 2

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials['usq', 'u'] = 2 * inputs['u']

