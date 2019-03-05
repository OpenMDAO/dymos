"""
ODE for example that shows how to use multiple phases in Dymos to model failure of a battery cell
in a simple electrical system.
"""
from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import Group, BalanceComp, NewtonSolver, DirectSolver

from dymos import declare_time, declare_state

from dymos.examples.battery_multibranch.batteries import Battery
from dymos.examples.battery_multibranch.motors import Motors


@declare_time(units='s')
@declare_state('state_of_charge', targets=['SOC'], rate_source='dXdt:SOC')
class BatteryODE(Group):

    def initialize(self):
        self.options.declare('num_nodes', default=1)
        self.options.declare('num_battery', default=3)
        self.options.declare('num_motor', default=3)

    def setup(self):
        num_nodes = self.options['num_nodes']
        num_battery = self.options['num_battery']
        num_motor = self.options['num_motor']

        self.add_subsystem(name='pwr_balance',
                           subsys=BalanceComp(name='I_Li', val=1.0*np.ones(num_nodes),
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

        self.nonlinear_solver = NewtonSolver()
        self.nonlinear_solver.options['maxiter'] = 20
        self.linear_solver = DirectSolver()


if __name__ == '__main__':

    from openmdao.api import Problem, IndepVarComp
    num_nodes = 1

    prob = Problem(model=BatteryODE(num_nodes=num_nodes))
    model = prob.model

    prob.setup()

    prob.run_model()

    derivs = prob.check_partials(compact_print=True)

    prob.model.list_outputs()

    print('done')
