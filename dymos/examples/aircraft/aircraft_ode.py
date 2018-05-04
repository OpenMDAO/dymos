from __future__ import print_function, division, absolute_import

from openmdao.api import Group

from dymos import declare_time, declare_state, declare_parameter
from dymos.models.atmosphere import StandardAtmosphereGroup

from .flight_path_angle_comp import FlightPathAngleComp
from .dynamic_pressure_comp import DynamicPressureComp
from .aero.aerodynamics_group import AerodynamicsGroup
from .flight_dynamics.flight_dynamics_group import FlightDynamicsGroup
from .propulsion.propulsion_group import PropulsionGroup
from .range_rate_comp import RangeRateComp
from .velocity_comp import VelocityComp


@declare_time(units='s')
@declare_state('x', rate_source='eom.xdot', units='m')
@declare_state('y', rate_source='eom.ydot', targets=['atmos.y'], units='m')
@declare_state('vx', rate_source='eom.vxdot', targets=['eom.vx'], units='m/s')
@declare_state('vy', rate_source='eom.vydot', targets=['eom.vy'], units='m/s')
@declare_state('m', rate_source='eom.mdot', targets=['eom.m'], units='kg')
@declare_parameter('thrust', targets=['eom.thrust'], units='N')
@declare_parameter('theta', targets=['eom.theta'], units='rad')
@declare_parameter('Isp', targets=['eom.Isp'], units='s')
class AircraftMissionODE(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int,
                              desc='Number of nodes to be evaluated in the RHS')

    def setup(self):
        nn = self.metadata['num_nodes']

        self.add_subsystem(name='atmos',
                           subsys=StandardAtmosphereGroup(num_nodes=nn))

        self.add_subsystem(name='vel_comp',
                           subsys=VelocityComp(num_nodes=nn))

        self.add_subsystem(name='gam_comp',
                           subsys=FlightPathAngleComp(num_nodes=nn))

        self.add_subsystem(name='q_comp',
                           subsys=DynamicPressureComp(num_nodes=nn))

        self.add_subsystem(name='flight_dynamics',
                           subsys=FlightDynamicsGroup(num_nodes=nn))

        self.add_subsystem(name='propulsion', system=PropulsionGroup(num_nodes=nn))

        self.add_subsystem(name='range_rate_comp',
                           subsys=RangeRateComp(num_nodes=nn))
