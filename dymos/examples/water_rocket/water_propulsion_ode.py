import numpy as np

import openmdao.api as om
from dymos.examples.min_time_climb.aero.dynamic_pressure_comp import DynamicPressureComp
from dymos.examples.min_time_climb.aero.lift_drag_force_comp import LiftDragForceComp
from dymos.models.atmosphere import USatm1976Comp
from dymos.models.eom import FlightPathEOM2D
from dymos.examples.cannonball.kinetic_energy_comp import KineticEnergyComp
from .water_engine_comp import WaterEngine


class WaterPropulsionODE(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='atmos',
                           subsys=USatm1976Comp(num_nodes=nn))

        self.add_subsystem(name='kinetic_energy',
                           subsys=KineticEnergyComp(num_nodes=nn))

        self.add_subsystem(name='water_engine',
                           subsys=WaterEngine(num_nodes=nn))

        self.add_subsystem(name='mass_adder',
                           subsys=_MassAdder(num_nodes=nn))

        self.add_subsystem(name='dynamic_pressure',
                           subsys=DynamicPressureComp(num_nodes=nn))

        self.add_subsystem(name='aero',
                           subsys=LiftDragForceComp(num_nodes=nn))

        self.add_subsystem(name='eom',
                           subsys=FlightPathEOM2D(num_nodes=nn))

        self.connect('atmos.rho', 'dynamic_pressure.rho')
        self.connect('atmos.pres', 'water_engine.p_a')
        self.connect('dynamic_pressure.q', 'aero.q')

        self.connect('aero.f_drag', 'eom.D')
        self.connect('aero.f_lift', 'eom.L')
        self.connect('water_engine.F', 'eom.T')
        self.connect('mass_adder.m', 'eom.m')


class _MassAdder(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('m_empty', val=np.zeros(nn), desc='empty mass', units='kg')
        self.add_input('V_w', val=1e-3*np.ones(nn), desc='water volume', units='m**3')
        self.add_input('rho_w', val=1e3*np.ones(nn), desc="water density", units='kg/m**3')

        self.add_output('m', val=np.zeros(nn), desc='total mass', units='kg')

        ar = np.arange(nn)
        self.declare_partials('*', '*', cols=ar, rows=ar)

    def compute(self, inputs, outputs):
        outputs['m'] = inputs['m_empty'] + inputs['rho_w']*inputs['V_w']

    def compute_partials(self, inputs, jacobian):
        jacobian['m', 'm_empty'] = 1
        jacobian['m', 'rho_w'] = inputs['V_w']
        jacobian['m', 'V_w'] = inputs['rho_w']


if __name__ == '__main__':
    p = om.Problem()
    p.model = WaterPropulsionODE(num_nodes=1)
    p.setup()
    p.check_config(checks=['unconnected_inputs'], out_file=None)
    p.final_setup()
