from __future__ import print_function, division, absolute_import

from openmdao.api import Group

from dymos import declare_time, declare_state, declare_parameter
from dymos.models.atmosphere import StandardAtmosphereGroup

from .unsteady_flight_path_angle_comp import UnsteadyFlightPathAngleComp
from .dynamic_pressure_comp import DynamicPressureComp
from .flight_equlibrium.flight_equilibrium_group import FlightEquilibriumGroup
from .propulsion.propulsion_group import PropulsionGroup
from .range_rate_comp import RangeRateComp
from .mach_comp import MachComp
from .mass_comp import MassComp


@declare_time(units='s')
@declare_state('range', rate_source='range_rate_comp.dXdt:range', units='m')
@declare_state('mass_fuel', targets=['mass_comp.mass_fuel'],
               rate_source='propulsion.dXdt:mass_fuel', units='kg')
@declare_parameter('alt', targets=['atmos.h', 'aero.alt', 'propulsion.alt'], units='m')
@declare_parameter('climb_rate', targets=['gam_comp.climb_rate'], units='m/s')
@declare_parameter('climb_rate2', targets=['gam_comp.climb_rate2'], units='m/s**2')
@declare_parameter('TAS', targets=['gam_comp.TAS', 'q_comp.TAS', 'range_rate_comp.TAS',
                                   'mach_comp.TAS', 'flight_dynamics.TAS'], units='m/s')
@declare_parameter('TAS_rate', targets=['gam_comp.TAS_rate', 'flight_equilibrium.TAS_rate'],
                   units='m/s**2')
@declare_parameter('S', targets=['aero.S'], units='m**2')
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

        self.connect('mass_comp.mass_total', 'flight_dynamics.mass')

        self.add_subsystem(name='atmos',
                           subsys=StandardAtmosphereGroup(num_nodes=nn))

        self.connect('atmos.pres', 'propulsion.pres')
        self.connect('atmos.sos', 'mach_comp.sos')
        self.connect('atmos.rho', 'q_comp.rho')

        self.add_subsystem(name='mach_comp',
                           subsys=MachComp(num_nodes=nn))

        self.connect('mach_comp.mach', ['aero.mach'])

        self.add_subsystem(name='gam_comp',
                           subsys=UnsteadyFlightPathAngleComp(num_nodes=nn))

        self.connect('gam_comp.gam', ('flight_dynamics.gam', 'range_rate_comp.gam'))

        self.connect('gam_comp.gam_rate', 'flight_equilibrium.gam_rate')

        self.add_subsystem(name='q_comp',
                           subsys=DynamicPressureComp(num_nodes=nn))

        self.connect('q_comp.q', ('aero.q'))

        self.add_subsystem(name='flight_equilibrium',
                           subsys=FlightEquilibriumGroup(num_nodes=nn),
                           promotes_inputs=['aero.*', 'flight_dynamics.*'],
                           promotes_outputs=['aero.*', 'flight_dynamics.*'])

        self.connect('flight_equilibrium.thrust', 'propulsion.thrust')

        self.add_subsystem(name='propulsion', subsys=PropulsionGroup(num_nodes=nn))

        self.add_subsystem(name='range_rate_comp',
                           subsys=RangeRateComp(num_nodes=nn))
