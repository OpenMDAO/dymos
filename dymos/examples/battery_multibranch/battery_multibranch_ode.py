"""
ODE for example that shows how to use multiple phases in Dymos to model failure of a battery cell
in a simple electrical system.
"""
import numpy as np
import openmdao.api as om

from dymos.examples.battery_multibranch.batteries import Battery
from dymos.examples.battery_multibranch.motors import Motors


class BatteryODE(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', default=1)
        self.options.declare('num_battery', default=3)
        self.options.declare('num_motor', default=3)

    def setup(self):
        num_nodes = self.options['num_nodes']
        num_battery = self.options['num_battery']
        num_motor = self.options['num_motor']

        self.add_subsystem(name='pwr_balance',
                           subsys=om.BalanceComp(name='I_Li', val=1.0*np.ones(num_nodes),
                                                 rhs_name='pwr_out_batt',
                                                 lhs_name='P_pack',
                                                 units='A', eq_units='W', lower=0.0, upper=50.))

        self.add_subsystem('battery', Battery(num_nodes=num_nodes, n_parallel=num_battery),
                           promotes_inputs=['SOC'],
                           promotes_outputs=['dXdt:SOC'])

        self.add_subsystem('motors', Motors(num_nodes=num_nodes, n_parallel=num_motor))

        self.connect('battery.P_pack', 'pwr_balance.P_pack')
        self.connect('motors.power_in_motor', 'pwr_balance.pwr_out_batt')
        self.connect('pwr_balance.I_Li', 'battery.I_Li')
        self.connect('battery.I_pack', 'motors.current_in_motor')

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=20)
        self.linear_solver = om.DirectSolver()
