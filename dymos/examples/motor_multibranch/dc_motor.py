"""
Simple model for brushless DC motor, based on Micromo DC motor tutorial.
"""
from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import Problem, Group, ExplicitComponent, MetaModelStructuredComp

train_rpm = np.linspace(11247.65, 175.25, 25)[::-1]
train_eta = np.array([0.10, 71.87, 75.27, 74.99, 73.25, 70.78, 67.89, 64.73, 61.40, 57.95, 54.41,
                      50.80, 47.14, 43.44, 39.71, 35.95, 32.17, 28.37, 24.56, 20.74, 16.90, 13.05,
                      9.0, 5.34, 1.47])[::-1] * .01


class DCMotorCurrent(ExplicitComponent):
    """
    """
    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('rpm', val=0.1*np.ones(nn), units='1/min',
                        desc='Shaft speed.')
        self.add_input('voltage', val=0.1*np.ones(nn), units='V',
                       desc='total supplied voltage')

        # Outputs
        self.add_output('current', val=0.1*np.ones(nn), units='A',
                       desc='applied current')

        # Derivatives
        row_col = np.arange(nn)
        self.declare_partials(of='*', wrt='*', rows=row_col, cols=row_col)

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        # 11.0/(11709 * 3.14159 / 30)
        self.options.declare('K_e', default=.0075,
                             desc='Back EMF constant (V*sec/rad).')
        self.options.declare('resistance', default=15.0,
                             desc='Terminal Resistance (Ohm)')

    def compute(self, inputs, outputs):
        opts = self.options
        V = inputs['voltage']
        omega = inputs['rpm'] * np.pi / 30.0

        outputs['current'] = (V - opts['K_e'] * omega) / opts['resistance']

    def compute_partials(self, inputs, partials):
        opts = self.options

        partials['current', 'voltage'][:] = 1.0 / opts['resistance']
        partials['current', 'rpm'][:] = -opts['K_e'] * (np.pi / (30.0 * opts['resistance']))


class DCMotorPower(ExplicitComponent):
    """
    """
    def setup(self):
        num_nodes = self.options['num_nodes']

        # Inputs
        self.add_input('current', val=0.1*np.ones(num_nodes), units='A',
                       desc='applied current')
        self.add_input('efficiency', val=0.1*np.ones(num_nodes),
                       desc='total supplied voltage')
        self.add_input('rpm', val=0.1*np.ones(num_nodes), units='1/min',
                        desc='Shaft speed.')
        self.add_input('voltage', val=0.1*np.ones(num_nodes), units='V',
                       desc='total supplied voltage')

        # Outputs
        self.add_output('torque', val=0.1*np.ones(num_nodes), units='N*m',
                        desc='Motor output torque.')
        self.add_output('power', val=0.1*np.ones(num_nodes), units='W',
                        desc='Motor output power.')

        # Derivatives
        row_col = np.arange(nn)
        self.declare_partials(of='power', wrt=['voltage', 'current', 'efficiency'],
                              rows=row_col, cols=row_col)
        self.declare_partials(of='torque', wrt='*', rows=row_col, cols=row_col)

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def compute(self, inputs, outputs):
        I = inputs['current']
        eff = inputs['efficiency']
        omega = inputs['rpm'] * np.pi / 30.0
        V = inputs['voltage']

        power = eff * I * V
        outputs['power'] = power
        outputs['torque'] = power / omega

    def compute_partials(self, inputs, partials):
        I = inputs['current']
        eff = inputs['efficiency']
        omega = inputs['rpm'] * np.pi / 30.0
        V = inputs['voltage']

        dpower_dV = eff * I
        dpower_deff = I * V
        dpower_dVI= eff * V

        partials['power', 'voltage'] = dpower_dV
        partials['power', 'efficiency'] = dpower_deff
        partials['power', 'current'] = dpower_dVI

        partials['torque', 'voltage'] = dpower_dV / omega
        partials['torque', 'efficiency'] = dpower_deff / omega
        partials['torque', 'current'] = dpower_dVI / omega
        partials['torque', 'rpm'] = -eff * I * V  * np.pi / (30.0 * omega**2)


class MMComp(MetaModelStructuredComp):

    def setup(self):
        nn = self.options['vec_size']
        self.add_input(name='rpm', val=np.ones(nn), units='1/min', training_data=train_rpm)

        self.add_output(name='efficiency', val=np.ones(nn), units=None, training_data=train_eta)


class DCMotor(Group):

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_subsystem('motor_current', DCMotorCurrent(num_nodes=num_nodes),
                           promotes_inputs=['rpm', 'voltage'],
                           promotes_outputs=['current'])

        self.add_subsystem('motor_efficiency', MMComp(vec_size=num_nodes, method='slinear', extrapolate=True),
                           promotes_inputs=['rpm'],
                           promotes_outputs=['efficiency'])

        self.add_subsystem('motor_power', DCMotorPower(num_nodes=num_nodes),
                           promotes_inputs=['current', 'efficiency', 'rpm', 'voltage'],
                           promotes_outputs=['torque', 'power'])

    def initialize(self):
        self.options.declare('num_nodes', types=int)


if __name__ == '__main__':

    from openmdao.api import Problem, IndepVarComp

    nn = len(train_rpm)

    prob = Problem(model=DCMotor(num_nodes=nn))
    model = prob.model

    ivc = IndepVarComp()
    ivc.add_output('rpm', train_rpm, units='1/min')
    ivc.add_output('voltage', 9.0*np.ones((nn, )), units='V')

    model.add_subsystem('ivc', ivc, promotes=['*'])

    prob.setup()

    prob.run_model()

    do_plot = True

    if do_plot:
        import matplotlib.pyplot as plt
        plt.plot(prob['rpm'], prob['power'])
        plt.xlabel('RPM')
        plt.ylabel('Output Power')
        plt.show()

    print('done')
