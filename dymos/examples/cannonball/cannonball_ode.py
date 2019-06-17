from __future__ import print_function, division, absolute_import

import openmdao.api as om
import dymos as dm
from dymos.examples.min_time_climb.aero.dynamic_pressure_comp import DynamicPressureComp
from dymos.examples.min_time_climb.aero.lift_drag_force_comp import LiftDragForceComp
from dymos.models.atmosphere import USatm1976Comp
from dymos.models.eom import FlightPathEOM2D
from .kinetic_energy_comp import KineticEnergyComp


@dm.declare_time(units='s')
@dm.declare_state(name='r', rate_source='eom.r_dot', units='m')
@dm.declare_state(name='h', rate_source='eom.h_dot', targets=['atmos.h'], units='m')
@dm.declare_state(name='gam', rate_source='eom.gam_dot', targets=['eom.gam'], units='rad')
@dm.declare_state(name='v', rate_source='eom.v_dot',
                  targets=['dynamic_pressure.v', 'eom.v', 'kinetic_energy.v'], units='m/s')
@dm.declare_parameter(name='CD', targets=['aero.CD'], units=None)
@dm.declare_parameter(name='CL', targets=['aero.CL'], units=None)
@dm.declare_parameter(name='T', targets=['eom.T'], units='N')
@dm.declare_parameter(name='alpha', targets=['eom.alpha'], units='deg')
@dm.declare_parameter(name='m', targets=['eom.m', 'kinetic_energy.m'], units='kg')
@dm.declare_parameter(name='S', targets=['aero.S'], units='m**2')
class CannonballODE(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='kinetic_energy',
                           subsys=KineticEnergyComp(num_nodes=nn))

        self.add_subsystem(name='atmos',
                           subsys=USatm1976Comp(num_nodes=nn))

        self.add_subsystem(name='dynamic_pressure',
                           subsys=DynamicPressureComp(num_nodes=nn))

        self.add_subsystem(name='aero',
                           subsys=LiftDragForceComp(num_nodes=nn))

        self.add_subsystem(name='eom',
                           subsys=FlightPathEOM2D(num_nodes=nn))

        self.connect('atmos.rho', 'dynamic_pressure.rho')
        self.connect('dynamic_pressure.q', 'aero.q')

        self.connect('aero.f_drag', 'eom.D')
        self.connect('aero.f_lift', 'eom.L')
