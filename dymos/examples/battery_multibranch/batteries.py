"""
Simple dynamic model of a LI battery.
"""
from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent


class Battery(ExplicitComponent):
    """
    Model of a Lithium Ion battery.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1)
        self.options.declare('n_series', default=1, desc='number of cells in series')
        self.options.declare('n_parallel', default=3, desc='number of cells in parallel')
        self.options.declare('Q_max', default=3.0,
                             desc='Max Energy Capacity of a battery cell in A*h')
        self.options.declare('R_0', default=.009,
                             desc='Internal resistance of the battery (ohms)')

    def setup(self):
        num_nodes = self.options['num_nodes']

        # Inputs
        self.add_input('I_Li', val=3.25*np.ones(num_nodes), units='A',
                       desc='Current demanded per cell')

        # State Variables
        self.add_input('SOC', val=np.ones(num_nodes), units=None, desc='State of charge')

        # Outputs
        self.add_output('U_L',
                        val=np.ones(num_nodes),
                        units='V',
                        desc='Terminal voltage of the battery')

        self.add_output('dXdt:SOC',
                        val=np.ones(num_nodes),
                        units='1/s',
                        desc='Time derivative of state of charge')

        self.add_output('I_pack', val=0.1*np.ones(num_nodes), units='A',
                        desc='Total Pack Current')
        self.add_output('U_pack', val=9.0*np.ones(num_nodes), units='V',
                        desc='Total Pack Voltage')
        self.add_output('P_pack', val=1.0*np.ones(num_nodes), units='W',
                        desc='Total Pack Power')

        # Derivatives
        row_col = np.arange(num_nodes)

        self.declare_partials(of='U_L', wrt=['SOC'], rows=row_col, cols=row_col, val=0.7)
        self.declare_partials(of='U_L', wrt=['I_Li'], rows=row_col, cols=row_col)
        self.declare_partials(of='dXdt:SOC', wrt=['I_Li'], rows=row_col, cols=row_col)
        self.declare_partials(of='I_pack', wrt=['I_Li'], rows=row_col, cols=row_col)
        self.declare_partials(of='U_pack', wrt=['SOC', 'I_Li'], rows=row_col, cols=row_col)
        self.declare_partials(of='P_pack', wrt=['SOC', 'I_Li'], rows=row_col, cols=row_col)


    def compute(self, inputs, outputs):
        opt = self.options
        I_Li = inputs['I_Li']
        SOC = inputs['SOC']

        # Simple linear curve fit for open circuit voltage as a function of state of charge.
        U_oc = 3.5 + 0.7*SOC

        outputs['U_L'] = U_oc - (I_Li * opt['R_0'])
        outputs['dXdt:SOC'] = -I_Li / (3600.0 * opt['Q_max'])

        outputs['I_pack'] = I_Li * opt['n_parallel']
        outputs['U_pack'] = outputs['U_L'] * opt['n_series']
        outputs['P_pack'] = outputs['I_pack'] * outputs['U_pack']

    def compute_partials(self, inputs, partials):
        opt = self.options
        I_Li = inputs['I_Li']
        SOC = inputs['SOC']

        partials['U_L', 'I_Li'] = -opt['R_0']

        partials['dXdt:SOC', 'I_Li'] = -1./(3600.0*opt['Q_max'])

        n_parallel = opt['n_parallel']
        n_series = opt['n_series']
        U_oc = 3.5 + 0.7*SOC
        U_L = U_oc - (I_Li * opt['R_0'])

        partials['I_pack', 'I_Li'] = n_parallel
        partials['U_pack', 'I_Li'] = -opt['R_0']
        partials['U_pack', 'SOC'] = n_series * 0.7
        partials['P_pack', 'I_Li'] = n_parallel * n_series * (U_L - I_Li * opt['R_0'])
        partials['P_pack', 'SOC'] = n_parallel * I_Li * n_series * 0.7


if __name__ == '__main__':

    from openmdao.api import Problem, IndepVarComp
    num_nodes = 1

    prob = Problem(model=Battery(num_nodes=num_nodes))
    model = prob.model

    prob.setup()
    prob.set_solver_print(level=2)

    prob.run_model()

    derivs = prob.check_partials(compact_print=True)

    print('done')