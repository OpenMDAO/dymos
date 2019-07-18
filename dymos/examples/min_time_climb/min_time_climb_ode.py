from __future__ import print_function, division, absolute_import

import openmdao.api as om
import dymos as dm
from ...models.atmosphere import USatm1976Comp
from .aero import AeroGroup
from .prop import PropGroup
from ...models.eom import FlightPathEOM2D


@dm.declare_time(units='s')
@dm.declare_state('r', units='m', rate_source='flight_dynamics.r_dot')
@dm.declare_state('h', units='m', rate_source='flight_dynamics.h_dot', targets=['h'])
@dm.declare_state('v', units='m/s', rate_source='flight_dynamics.v_dot', targets=['v'])
@dm.declare_state('gam', units='rad', rate_source='flight_dynamics.gam_dot', targets=['gam'])
@dm.declare_state('m', units='kg', rate_source='prop.m_dot', targets=['m'])
@dm.declare_parameter('alpha', targets=['alpha'], units='rad')
@dm.declare_parameter('Isp', targets=['Isp'], units='s')
@dm.declare_parameter('S', targets=['S'], units='m**2')
@dm.declare_parameter('throttle', targets=['throttle'], units=None)
class MinTimeClimbODE(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='atmos',
                           subsys=USatm1976Comp(num_nodes=nn),
                           promotes_inputs=['h'])

        self.add_subsystem(name='aero',
                           subsys=AeroGroup(num_nodes=nn),
                           promotes_inputs=['v', 'alpha', 'S'])

        self.connect('atmos.sos', 'aero.sos')
        self.connect('atmos.rho', 'aero.rho')

        self.add_subsystem(name='prop',
                           subsys=PropGroup(num_nodes=nn),
                           promotes_inputs=['h', 'Isp', 'throttle'])

        self.connect('aero.mach', 'prop.mach')

        self.add_subsystem(name='flight_dynamics',
                           subsys=FlightPathEOM2D(num_nodes=nn),
                           promotes_inputs=['m', 'v', 'gam', 'alpha'])

        self.connect('aero.f_drag', 'flight_dynamics.D')
        self.connect('aero.f_lift', 'flight_dynamics.L')
        self.connect('prop.thrust', 'flight_dynamics.T')
