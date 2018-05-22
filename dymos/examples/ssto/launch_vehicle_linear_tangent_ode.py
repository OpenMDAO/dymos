from __future__ import print_function, division, absolute_import

from openmdao.api import Group

from dymos import declare_time, declare_state, declare_parameter

from .log_atmosphere_comp import LogAtmosphereComp
from .launch_vehicle_2d_eom_comp import LaunchVehicle2DEOM
from .linear_tangent_guidance_comp import LinearTangentGuidanceComp


@declare_time(units='s', targets=['guidance.time'])
@declare_state('x', rate_source='eom.xdot', units='m')
@declare_state('y', rate_source='eom.ydot', targets=['atmos.y'], units='m')
@declare_state('vx', rate_source='eom.vxdot', targets=['eom.vx'], units='m/s')
@declare_state('vy', rate_source='eom.vydot', targets=['eom.vy'], units='m/s')
@declare_state('m', rate_source='eom.mdot', targets=['eom.m'], units='kg')
@declare_parameter('thrust', targets=['eom.thrust'], units='N')
@declare_parameter('a_ctrl', targets=['guidance.a_ctrl'], units='1/s')
@declare_parameter('b_ctrl', targets=['guidance.b_ctrl'], units=None)
@declare_parameter('Isp', targets=['eom.Isp'], units='s')
class LaunchVehicleLinearTangentODE(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

        self.options.declare('central_body', values=['earth', 'moon'], default='earth',
                             desc='The central graviational body for the launch vehicle.')

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

        self.add_subsystem('guidance', LinearTangentGuidanceComp(num_nodes=nn))

        self.add_subsystem('eom', LaunchVehicle2DEOM(num_nodes=nn, central_body=cb))

        self.connect('atmos.rho', 'eom.rho')
        self.connect('guidance.theta', 'eom.theta')
