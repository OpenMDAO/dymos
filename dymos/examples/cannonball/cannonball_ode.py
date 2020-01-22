import openmdao.api as om
from dymos.examples.min_time_climb.aero.dynamic_pressure_comp import DynamicPressureComp
from dymos.examples.min_time_climb.aero.lift_drag_force_comp import LiftDragForceComp
from dymos.models.atmosphere import USatm1976Comp
from dymos.models.eom import FlightPathEOM2D
from .kinetic_energy_comp import KineticEnergyComp


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
