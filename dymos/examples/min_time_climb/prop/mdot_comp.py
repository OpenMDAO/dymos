import numpy as np

import openmdao.api as om


class MassFlowRateComp(om.ExplicitComponent):
    """ Computes mass flow rate for the F4's 2 J79 engines at full throttle. """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('thrust', shape=(nn,), desc='engine thrust', units='N')
        self.add_input('Isp', shape=(nn,), desc='engine specific impulse', units='s')

        # Outputs
        self.add_output(name='m_dot', val=np.zeros(nn),
                        desc='vehicle mass accumulation rate due to fuel consumption',
                        units='kg/s')

        # Jacobian
        ar = np.arange(nn)
        self.declare_partials(of='m_dot', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='m_dot', wrt='Isp', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['m_dot'] = -inputs['thrust'] / (9.80665 * inputs['Isp'])

    def compute_partials(self, inputs, partials):
        mdot = inputs['thrust'] / (9.80665 * inputs['Isp'])
        partials['m_dot', 'thrust'] = -1.0 / (9.80665 * inputs['Isp'])
        partials['m_dot', 'Isp'] = mdot / inputs['Isp']
