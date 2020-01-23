import openmdao.api as om

from .log_atmosphere_comp import LogAtmosphereComp
from .launch_vehicle_2d_eom_comp import LaunchVehicle2DEOM


class LaunchVehicleODE(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

        self.options.declare('central_body', values=['earth', 'moon'], default='earth',
                             desc='The central gravitational body for the launch vehicle.')

    def setup(self):
        nn = self.options['num_nodes']
        cb = self.options['central_body']

        if cb == 'earth':
            rho_ref = 1.225
            h_scale = 8.44E3
        elif cb == 'moon':
            rho_ref = 0.0
            h_scale = 1.0
        else:
            raise RuntimeError('Unrecognized value for central_body: {0}'.format(cb))

        self.add_subsystem('atmos',
                           LogAtmosphereComp(num_nodes=nn, rho_ref=rho_ref, h_scale=h_scale))

        self.add_subsystem('eom', LaunchVehicle2DEOM(num_nodes=nn, central_body=cb))

        self.connect('atmos.rho', 'eom.rho')
