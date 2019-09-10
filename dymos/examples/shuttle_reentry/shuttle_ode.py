from openmdao.api import Problem, Group
from dymos.examples.shuttle_reentry.heating_comp import AerodynamicHeating
from dymos.examples.shuttle_reentry.flight_dynamics_comp import FlightDynamics
from dymos.examples.shuttle_reentry.aerodynamics_comp import Aerodynamics
from dymos.examples.shuttle_reentry.atmosphere_comp import Atmosphere


class ShuttleODE(Group):
    """
    The ODE for the Shuttle reentry problem.

    References
    ----------
    .. [1] Betts, John T., Practical Methods for Optimal Control and Estimation Using Nonlinear
           Programming, p. 248, 2010.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('atmosphere', subsys=Atmosphere(num_nodes=nn),
                           promotes_inputs=['h'], promotes_outputs=['rho'])
        self.add_subsystem('aerodynamics', subsys=Aerodynamics(num_nodes=nn),
                           promotes_inputs=['alpha', 'v', 'rho'],
                           promotes_outputs=['lift', 'drag'])
        self.add_subsystem('heating', subsys=AerodynamicHeating(num_nodes=nn),
                           promotes_inputs=['rho', 'v', 'alpha'], promotes_outputs=['q'])
        self.add_subsystem('eom', subsys=FlightDynamics(num_nodes=nn),
                           promotes_inputs=['beta', 'gamma', 'h', 'psi', 'theta', 'v', 'lift',
                                            'drag'],
                           promotes_outputs=['hdot', 'gammadot', 'phidot', 'psidot', 'thetadot',
                                             'vdot'])
