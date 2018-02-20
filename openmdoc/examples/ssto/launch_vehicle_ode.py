from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import Group, IndepVarComp

from openmdoc import ODEFunction

from .log_atmosphere_comp import LogAtmosphereComp
from .launch_vehicle_2d_eom_comp import LaunchVehicle2DEOM


class LaunchVehicle2DModel(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int,
                              desc='Number of nodes to be evaluated in the RHS')

        self.metadata.declare('central_body', values=['earth', 'moon'], default='earth',
                              desc='The central graviational body for the launch vehicle.')

    def setup(self):
        nn = self.metadata['num_nodes']

        if self.metadata['central_body'] == 'earth':
            self.add_subsystem('atmos',
                               LogAtmosphereComp(num_nodes=self.metadata['num_nodes']))
        else:
            self.add_subsystem('atmos',
                               IndepVarComp('rho', val=np.zeros(nn)))

        self.add_subsystem('eom',
                           LaunchVehicle2DEOM(num_nodes=self.metadata['num_nodes'],
                                              central_body=self.metadata['central_body']))

        self.connect('atmos.rho', 'eom.rho')


class LaunchVehicleODE(ODEFunction):
    """
    Time Variable
    -------------
    t : float
        time values at the nodes (s)

    EOM State Variables
    -------------------
    m : float
        vehicle mass (kg)
    t : float
        time values at the nodes (s)
    vx : float
        horizontal component of velocity (m/s)
    vy : float
        vertical component of velocity (m/s)
    x : float
        horizontal component of position (m)
    y : float
        altitude (m)

    Control Parameters
    ------------------
    theta : float
        pitch angle w.r.t. horizontal (rad)

    Unknowns
    --------
    dXdt:m : float
        Time rate of change of m (kg/s)
    dXdt:vx : float
        Time rate of change of vx (m/s/s)
    dXdt:vy : float
        Time rate of change of vy (m/s/s)
    dXdt:x : float
        Time rate of change of x (m/s)
    dXdt:y : float
        Time rate of change of y (m/s)
    """
    def __init__(self, central_body='earth'):
        super(LaunchVehicleODE, self).__init__(system_class=LaunchVehicle2DModel,
                                               system_init_kwargs={'central_body': central_body})

        self.declare_time(units='s')

        y_targets = ['atmos.y'] if central_body == 'earth' else None

        self.declare_state('x', rate_source='eom.xdot', units='m')
        self.declare_state('y', rate_source='eom.ydot', targets=y_targets, units='m')
        self.declare_state('vx', rate_source='eom.vxdot', targets=['eom.vx'], units='m/s')
        self.declare_state('vy', rate_source='eom.vydot', targets=['eom.vy'], units='m/s')
        self.declare_state('m', rate_source='eom.mdot', targets=['eom.m'], units='kg')

        self.declare_parameter('thrust', targets=['eom.thrust'], units='N')
        self.declare_parameter('theta', targets=['eom.theta'], units='rad')
