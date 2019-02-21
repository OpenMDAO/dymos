from __future__ import print_function, division, absolute_import

from openmdao.api import ExplicitComponent
from dymos import declare_time, declare_state, declare_parameter


@declare_time(units='s')
@declare_state('x', rate_source='v', units='m')
@declare_state('v', rate_source='u', units='m/s')
@declare_parameter('u', units='m/s**2')
class DoubleIntegratorODE(ExplicitComponent):
    """
    The double integrator is a special case where the state rates are all set to other states
    or parameters.  Since we aren't computing any other outputs, the ODE doesn't actually
    need to compute anything.  OpenMDAO will warn us that the component has no outputs, but
    Dymos will solve the problem just fine.

    Note we still have to declare the num_nodes option in initialize so that Dymos can instantiate
    the ODE.

    Also note that neither time, states, nor paramters have targets, since there are no inputs
    in the ODE system.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
