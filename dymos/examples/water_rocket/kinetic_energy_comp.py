import numpy as np

import openmdao.api as om


class KineticEnergyComp(om.ExplicitComponent):
    """
    Computes the kinetic energy of a particle based on its speed and mass

    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='v', val=np.ones(nn), desc='cannonball speed', units='m/s')
        self.add_input(name='m', val=np.ones(nn), desc='cannonball mass', units='kg')

        self.add_output(name='ke', shape=(nn,), desc='kinetic energy', units='J')

        ar = np.arange(nn)

        self.declare_partials(of='ke', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='ke', wrt='m', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        m = inputs['m']
        v = inputs['v']

        outputs['ke'] = 0.5 * m * v ** 2

    def compute_partials(self, inputs, partials):
        m = inputs['m']
        v = inputs['v']

        partials['ke', 'm'] = 0.5 * v ** 2
        partials['ke', 'v'] = m * v
