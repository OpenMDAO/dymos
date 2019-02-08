from __future__ import print_function, division, absolute_import

from openmdao.api import Group, DirectSolver, NewtonSolver, BoundsEnforceLS

from dymos import declare_time, declare_state, declare_parameter
from dymos.models.atmosphere import USatm1976Comp

from .steady_flight_path_angle_comp import SteadyFlightPathAngleComp
from .dynamic_pressure_comp import DynamicPressureComp
from .flight_equlibrium.steady_flight_equilibrium_group import SteadyFlightEquilibriumGroup
from .propulsion.propulsion_group import PropulsionGroup
from .range_rate_comp import RangeRateComp
from .true_airspeed_comp import TrueAirspeedComp
from .mass_comp import MassComp


@declare_time(units='s')
@declare_state('range', rate_source='range_rate_comp.dXdt:range', units='m')
@declare_state('mass_fuel', targets=['mass_comp.mass_fuel'],
               rate_source='propulsion.dXdt:mass_fuel', units='kg')
@declare_state('alt', targets=['atmos.h', 'aero.alt', 'propulsion.alt'],
               rate_source='climb_rate', units='m')
# @declare_parameter('alt', targets=['atmos.h', 'aero.alt', 'propulsion.alt'], units='m')
@declare_parameter('climb_rate', targets=['gam_comp.climb_rate'], units='m/s')
@declare_parameter('mach', targets=['tas_comp.mach', 'aero.mach'], units='m/s')
@declare_parameter('S', targets=['aero.S', 'flight_equilibrium.S', 'propulsion.S'], units='m**2')
@declare_parameter('mass_empty', targets=['mass_comp.mass_empty'], units='kg')
@declare_parameter('mass_payload', targets=['mass_comp.mass_payload'], units='kg')
class AircraftODE(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='mass_comp',
                           subsys=MassComp(num_nodes=nn))

        self.connect('mass_comp.W_total', 'flight_equilibrium.W_total')

        self.add_subsystem(name='atmos',
                           subsys=USatm1976Comp(num_nodes=nn))

        self.connect('atmos.pres', 'propulsion.pres')
        self.connect('atmos.sos', 'tas_comp.sos')
        self.connect('atmos.rho', 'q_comp.rho')

        self.add_subsystem(name='tas_comp',
                           subsys=TrueAirspeedComp(num_nodes=nn))

        self.connect('tas_comp.TAS',
                     ('gam_comp.TAS', 'q_comp.TAS', 'range_rate_comp.TAS'))

        self.add_subsystem(name='gam_comp',
                           subsys=SteadyFlightPathAngleComp(num_nodes=nn))

        self.connect('gam_comp.gam', ('flight_equilibrium.gam', 'range_rate_comp.gam'))

        self.add_subsystem(name='q_comp',
                           subsys=DynamicPressureComp(num_nodes=nn))

        self.connect('q_comp.q', ('aero.q', 'flight_equilibrium.q', 'propulsion.q'))

        self.add_subsystem(name='flight_equilibrium',
                           subsys=SteadyFlightEquilibriumGroup(num_nodes=nn),
                           promotes_inputs=['aero.*'],
                           promotes_outputs=['aero.*'])

        self.connect('flight_equilibrium.CT', 'propulsion.CT')

        self.add_subsystem(name='propulsion', subsys=PropulsionGroup(num_nodes=nn))

        self.add_subsystem(name='range_rate_comp', subsys=RangeRateComp(num_nodes=nn))
