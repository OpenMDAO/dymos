import numpy as np

import openmdao.api as om


class DynamicPressureComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='rho', shape=(nn,), desc='atmospheric density', units='kg/m**3')
        self.add_input(name='v', shape=(nn,), desc='air-relative velocity', units='m/s')

        self.add_output(name='q', shape=(nn,), desc='dynamic pressure', units='N/m**2')

        ar = np.arange(nn)
        self.declare_partials(of='q', wrt='rho', rows=ar, cols=ar)
        self.declare_partials(of='q', wrt='v', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['q'] = 0.5 * inputs['rho'] * inputs['v'] ** 2

    def compute_partials(self, inputs, partials):
        partials['q', 'rho'] = 0.5 * inputs['v'] ** 2
        partials['q', 'v'] = inputs['rho'] * inputs['v']
