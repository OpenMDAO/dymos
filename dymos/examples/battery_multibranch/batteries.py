"""
Simple dynamic model of a LI battery.
"""
import numpy as np
from scipy.interpolate import Akima1DInterpolator

import openmdao.api as om
# Data for open circuit voltage model.
train_SOC = np.array([0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
train_V_oc = np.array([3.5, 3.55, 3.65, 3.75, 3.9, 4.1, 4.2])


class Battery(om.ExplicitComponent):
    """
    Model of a Lithium Ion battery.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1)
        self.options.declare('n_series', default=1, desc='number of cells in series')
        self.options.declare('n_parallel', default=3, desc='number of cells in parallel')
        self.options.declare('Q_max', default=1.05,
                             desc='Max Energy Capacity of a battery cell in A*h')
        self.options.declare('R_0', default=.025,
                             desc='Internal resistance of the battery (ohms)')

    def setup(self):
        num_nodes = self.options['num_nodes']

        # Inputs
        self.add_input('I_Li', val=np.ones(num_nodes), units='A',
                       desc='Current demanded per cell')

        # State Variables
        self.add_input('SOC', val=np.ones(num_nodes), units=None, desc='State of charge')

        # Outputs
        self.add_output('V_L',
                        val=np.ones(num_nodes),
                        units='V',
                        desc='Terminal voltage of the battery')

        self.add_output('dXdt:SOC',
                        val=np.ones(num_nodes),
                        units='1/s',
                        desc='Time derivative of state of charge')

        self.add_output('V_oc', val=np.ones(num_nodes), units='V',
                        desc='Open Circuit Voltage')
        self.add_output('I_pack', val=0.1*np.ones(num_nodes), units='A',
                        desc='Total Pack Current')
        self.add_output('V_pack', val=9.0*np.ones(num_nodes), units='V',
                        desc='Total Pack Voltage')
        self.add_output('P_pack', val=1.0*np.ones(num_nodes), units='W',
                        desc='Total Pack Power')

        # Derivatives
        row_col = np.arange(num_nodes)

        self.declare_partials(of='V_oc', wrt=['SOC'], rows=row_col, cols=row_col)
        self.declare_partials(of='V_L', wrt=['SOC'], rows=row_col, cols=row_col)
        self.declare_partials(of='V_L', wrt=['I_Li'], rows=row_col, cols=row_col)
        self.declare_partials(of='dXdt:SOC', wrt=['I_Li'], rows=row_col, cols=row_col)
        self.declare_partials(of='I_pack', wrt=['I_Li'], rows=row_col, cols=row_col)
        self.declare_partials(of='V_pack', wrt=['SOC', 'I_Li'], rows=row_col, cols=row_col)
        self.declare_partials(of='P_pack', wrt=['SOC', 'I_Li'], rows=row_col, cols=row_col)

        self.voltage_model = Akima1DInterpolator(train_SOC, train_V_oc)
        self.voltage_model_derivative = self.voltage_model.derivative()

    def compute(self, inputs, outputs):
        opt = self.options
        I_Li = inputs['I_Li']
        SOC = inputs['SOC']

        V_oc = self.voltage_model(SOC, extrapolate=True)

        outputs['V_oc'] = V_oc
        outputs['V_L'] = V_oc - (I_Li * opt['R_0'])
        outputs['dXdt:SOC'] = -I_Li / (3600.0 * opt['Q_max'])

        outputs['I_pack'] = I_Li * opt['n_parallel']
        outputs['V_pack'] = outputs['V_L'] * opt['n_series']
        outputs['P_pack'] = outputs['I_pack'] * outputs['V_pack']

    def compute_partials(self, inputs, partials):
        opt = self.options
        I_Li = inputs['I_Li']
        SOC = inputs['SOC']

        dV_dSOC = self.voltage_model_derivative(SOC, extrapolate=True)
        partials['V_oc', 'SOC'] = dV_dSOC
        partials['V_L', 'SOC'] = dV_dSOC

        partials['V_L', 'I_Li'] = -opt['R_0']

        partials['dXdt:SOC', 'I_Li'] = -1./(3600.0*opt['Q_max'])

        n_parallel = opt['n_parallel']
        n_series = opt['n_series']
        V_oc = self.voltage_model(SOC, extrapolate=True)
        V_L = V_oc - (I_Li * opt['R_0'])

        partials['I_pack', 'I_Li'] = n_parallel
        partials['V_pack', 'I_Li'] = -opt['R_0']
        partials['V_pack', 'SOC'] = n_series * dV_dSOC
        partials['P_pack', 'I_Li'] = n_parallel * n_series * (V_L - I_Li * opt['R_0'])
        partials['P_pack', 'SOC'] = n_parallel * I_Li * n_series * dV_dSOC


if __name__ == '__main__':

    import openmdao.api as om
    num_nodes = 1

    prob = om.Problem(model=Battery(num_nodes=num_nodes))
    model = prob.model

    prob.setup()
    prob.set_solver_print(level=2)

    prob.run_model()

    derivs = prob.check_partials(compact_print=True)

    print('done')
