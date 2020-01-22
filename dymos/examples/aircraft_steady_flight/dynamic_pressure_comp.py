import numpy as np
import openmdao.api as om


class DynamicPressureComp(om.ExplicitComponent):
    """ Compute the dynamic pressure based on the velocity and the atmospheric density. """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input(name='rho', shape=(nn,), desc='Atmospheric density', units='kg/m**3')
        self.add_input(name='TAS', shape=(nn,), desc='True airspeed', units='m/s')

        # Outputs
        self.add_output(name='q', val=np.zeros(nn), desc='Dynamic pressure', units='Pa')

        # Partials
        ar = np.arange(nn)
        self.declare_partials(of='q', wrt='rho', rows=ar, cols=ar)
        self.declare_partials(of='q', wrt='TAS', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['q'] = 0.5 * inputs['rho'] * inputs['TAS'] ** 2

    def compute_partials(self, inputs, partials):
        partials['q', 'rho'] = 0.5 * inputs['TAS'] ** 2
        partials['q', 'TAS'] = inputs['rho'] * inputs['TAS']
